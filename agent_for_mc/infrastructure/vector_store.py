from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import lancedb
import pyarrow as pa

from agent_for_mc.domain.errors import StartupValidationError
from agent_for_mc.domain.models import RetrievedDoc, VectorStoreStats
from agent_for_mc.infrastructure.observability import record_counter, trace_operation


REQUIRED_FIELDS = {
    "id",
    "content",
    "plugin_chinese_name",
    "plugin_english_name",
    "embedding",
}


@dataclass(slots=True)
class _NameMatch:
    name_length: int
    row: dict


class LancePluginVectorStore:
    def __init__(
        self,
        db_dir: Path,
        table_name: str,
        *,
        expected_embedding_dimension: int = 1024,
    ):
        self._db_dir = Path(db_dir)
        self._table_name = table_name
        self._expected_embedding_dimension = expected_embedding_dimension
        self._db = None
        self._table = None
        self._stats: VectorStoreStats | None = None
        self._rows: list[dict] | None = None
        self._bm25_index_ready = False

    def validate(self) -> VectorStoreStats:
        with trace_operation(
            "vector_store.validate",
            attributes={"component": "vector_store", "table": self._table_name},
            metric_name="rag_vector_store_validate_seconds",
        ):
            record_counter("rag_vector_store_validate_requests_total")
            if self._stats is not None:
                return self._stats

            if not self._db_dir.exists():
                raise StartupValidationError(f"Lance 数据库目录不存在: {self._db_dir}")

            try:
                self._db = lancedb.connect(str(self._db_dir))
                self._table = self._db.open_table(self._table_name)
            except Exception as exc:
                raise StartupValidationError(
                    f"无法打开 Lance 表 {self._table_name}: {exc}"
                ) from exc

            try:
                arrow_table = self._table.to_arrow()
            except Exception as exc:
                raise StartupValidationError(f"读取 Lance 表失败: {exc}") from exc

            schema = arrow_table.schema
            actual_fields = {field.name for field in schema}
            missing_fields = sorted(REQUIRED_FIELDS - actual_fields)
            if missing_fields:
                raise StartupValidationError(
                    f"Lance 表缺少必要字段: {', '.join(missing_fields)}"
                )

            embedding_field = schema.field("embedding")
            if not pa.types.is_fixed_size_list(embedding_field.type):
                raise StartupValidationError("embedding 字段不是 fixed_size_list 类型。")
            if embedding_field.type.list_size != self._expected_embedding_dimension:
                raise StartupValidationError(
                    "embedding 维度不匹配，"
                    f"期望 {self._expected_embedding_dimension}，"
                    f"实际 {embedding_field.type.list_size}。"
                )

            self._rows = arrow_table.to_pylist()
            self._stats = VectorStoreStats(
                db_dir=self._db_dir,
                table_name=self._table_name,
                record_count=len(self._rows),
                embedding_dimension=embedding_field.type.list_size,
            )
            return self._stats

    def find_name_matches(self, search_query: str) -> list[RetrievedDoc]:
        with trace_operation(
            "vector_store.find_name_matches",
            attributes={"component": "vector_store", "query.length": len(search_query.strip())},
            metric_name="rag_vector_store_name_match_seconds",
        ):
            record_counter("rag_vector_store_name_match_requests_total")
            self.validate()
            normalized_search_query = search_query.lower()
            matches: list[_NameMatch] = []

            for row in self._rows or []:
                best_name_length = 0
                chinese_name = str(row.get("plugin_chinese_name") or "").strip()
                english_name = str(row.get("plugin_english_name") or "").strip()

                if chinese_name and chinese_name in search_query:
                    best_name_length = max(best_name_length, len(chinese_name))
                if english_name and english_name.lower() in normalized_search_query:
                    best_name_length = max(best_name_length, len(english_name))

                if best_name_length > 0:
                    matches.append(_NameMatch(name_length=best_name_length, row=row))

            matches.sort(
                key=lambda item: (
                    item.name_length,
                    len(str(item.row.get("content") or "")),
                    str(item.row.get("plugin_chinese_name") or ""),
                ),
                reverse=True,
            )
            return [
                self._row_to_doc(item.row, distance=0.0, match_reason="name-boost")
                for item in matches
            ]

    def search_by_embedding(
        self,
        embedding: list[float],
        *,
        top_k: int,
    ) -> list[RetrievedDoc]:
        with trace_operation(
            "vector_store.search_by_embedding",
            attributes={"component": "vector_store", "top_k": top_k},
            metric_name="rag_vector_store_search_seconds",
        ):
            record_counter("rag_vector_store_search_requests_total")
            self.validate()
            try:
                rows = self._table.search(embedding).limit(top_k).to_list()
            except Exception as exc:
                raise StartupValidationError(f"Lance 向量检索失败: {exc}") from exc

            docs: list[RetrievedDoc] = []
            for row in rows:
                docs.append(
                    self._row_to_doc(
                        row,
                        distance=float(row.get("_distance", 0.0)),
                        match_reason="vector",
                    )
            )
            return docs

    def ensure_bm25_index(self, *, replace: bool = False) -> None:
        with trace_operation(
            "vector_store.ensure_bm25_index",
            attributes={"component": "vector_store", "table": self._table_name},
            metric_name="rag_vector_store_bm25_index_seconds",
        ):
            record_counter("rag_vector_store_bm25_index_requests_total")
            self.validate()
            if self._bm25_index_ready and not replace:
                return

            if not replace and _has_content_fts_index(self._table.list_indices()):
                self._bm25_index_ready = True
                return

            try:
                self._table.create_fts_index(
                    "content",
                    replace=replace,
                    base_tokenizer="ngram",
                    ngram_min_length=2,
                    ngram_max_length=3,
                    stem=False,
                    remove_stop_words=False,
                )
            except Exception as exc:
                raise StartupValidationError(f"Lance BM25 索引创建失败: {exc}") from exc
            self._bm25_index_ready = True

    def search_by_bm25(
        self,
        search_query: str,
        *,
        top_k: int,
        auto_create_index: bool = True,
    ) -> list[RetrievedDoc]:
        with trace_operation(
            "vector_store.search_by_bm25",
            attributes={"component": "vector_store", "top_k": top_k},
            metric_name="rag_vector_store_bm25_search_seconds",
        ):
            record_counter("rag_vector_store_bm25_search_requests_total")
            normalized_query = " ".join(str(search_query).split()).strip()
            if not normalized_query or top_k < 1:
                return []

            self.validate()
            if auto_create_index:
                self.ensure_bm25_index()
            elif not self._bm25_index_ready:
                self._bm25_index_ready = _has_content_fts_index(self._table.list_indices())

            try:
                rows = (
                    self._table.search(
                        normalized_query,
                        query_type="fts",
                        fts_columns="content",
                    )
                    .limit(top_k)
                    .to_list()
                )
            except Exception as exc:
                raise StartupValidationError(f"Lance BM25 检索失败: {exc}") from exc

            docs: list[RetrievedDoc] = []
            for row in rows:
                docs.append(
                    self._row_to_doc(
                        row,
                        distance=float(row.get("_score", 0.0)),
                        match_reason="bm25",
                    )
                )
            return docs

    def _row_to_doc(
        self,
        row: dict,
        *,
        distance: float,
        match_reason: str,
    ) -> RetrievedDoc:
        return RetrievedDoc(
            id=int(row["id"]),
            plugin_chinese_name=str(row.get("plugin_chinese_name") or "原文未明确说明"),
            plugin_english_name=str(row.get("plugin_english_name") or "原文未明确说明"),
            content=str(row.get("content") or ""),
            distance=distance,
            match_reason=match_reason,
        )


def _has_content_fts_index(indices) -> bool:
    for index in indices:
        index_type = str(getattr(index, "index_type", "")).upper()
        columns = [str(column) for column in getattr(index, "columns", [])]
        if index_type == "FTS" and "content" in columns:
            return True
    return False
