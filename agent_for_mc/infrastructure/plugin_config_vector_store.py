from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import lancedb
import pyarrow as pa

from agent_for_mc.domain.errors import StartupValidationError
from agent_for_mc.domain.models import PluginConfigDoc, VectorStoreStats
from agent_for_mc.infrastructure.observability import record_counter, trace_operation


REQUIRED_FIELDS = {
    "id",
    "content",
    "plugin_chinese_name",
    "plugin_english_name",
    "file_path",
    "embedding",
}


@dataclass(slots=True)
class _PathMatch:
    score: int
    row: dict


class LancePluginConfigVectorStore:
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

    def validate(self) -> VectorStoreStats:
        with trace_operation(
            "plugin_config_vector_store.validate",
            attributes={"component": "plugin_config_vector_store", "table": self._table_name},
            metric_name="rag_plugin_config_vector_store_validate_seconds",
        ):
            record_counter("rag_plugin_config_vector_store_validate_requests_total")
            if self._stats is not None:
                return self._stats

            if not self._db_dir.exists():
                raise StartupValidationError(f"配置向量数据库目录不存在: {self._db_dir}")

            try:
                self._db = lancedb.connect(str(self._db_dir))
                self._table = self._db.open_table(self._table_name)
            except Exception as exc:
                raise StartupValidationError(
                    f"无法打开配置向量表 {self._table_name}: {exc}"
                ) from exc

            try:
                arrow_table = self._table.to_arrow()
            except Exception as exc:
                raise StartupValidationError(f"读取配置向量表失败: {exc}") from exc

            schema = arrow_table.schema
            actual_fields = {field.name for field in schema}
            missing_fields = sorted(REQUIRED_FIELDS - actual_fields)
            if missing_fields:
                raise StartupValidationError(
                    f"配置向量表缺少必要字段: {', '.join(missing_fields)}"
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

    def find_name_matches(self, search_query: str) -> list[PluginConfigDoc]:
        with trace_operation(
            "plugin_config_vector_store.find_name_matches",
            attributes={"component": "plugin_config_vector_store", "query.length": len(search_query.strip())},
            metric_name="rag_plugin_config_vector_store_name_match_seconds",
        ):
            record_counter("rag_plugin_config_vector_store_name_match_requests_total")
            self.validate()
            normalized_search_query = search_query.lower()
            matches: list[_PathMatch] = []

            for row in self._rows or []:
                score = 0
                file_path = str(row.get("file_path") or "").strip()
                chinese_name = str(row.get("plugin_chinese_name") or "").strip()
                english_name = str(row.get("plugin_english_name") or "").strip()
                basename = Path(file_path).name if file_path else ""
                stem = Path(file_path).stem if file_path else ""

                for candidate in (chinese_name, english_name, file_path, basename, stem):
                    if candidate and candidate.lower() in normalized_search_query:
                        score = max(score, len(candidate))

                if score > 0:
                    matches.append(_PathMatch(score=score, row=row))

            matches.sort(
                key=lambda item: (
                    item.score,
                    len(str(item.row.get("content") or "")),
                    str(item.row.get("file_path") or ""),
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
    ) -> list[PluginConfigDoc]:
        with trace_operation(
            "plugin_config_vector_store.search_by_embedding",
            attributes={"component": "plugin_config_vector_store", "top_k": top_k},
            metric_name="rag_plugin_config_vector_store_search_seconds",
        ):
            record_counter("rag_plugin_config_vector_store_search_requests_total")
            self.validate()
            try:
                rows = self._table.search(embedding).limit(top_k).to_list()
            except Exception as exc:
                raise StartupValidationError(f"配置向量检索失败: {exc}") from exc

            docs: list[PluginConfigDoc] = []
            for row in rows:
                docs.append(
                    self._row_to_doc(
                        row,
                        distance=float(row.get("_distance", 0.0)),
                        match_reason="vector",
                    )
                )
            return docs

    def _row_to_doc(
        self,
        row: dict,
        *,
        distance: float,
        match_reason: str,
    ) -> PluginConfigDoc:
        file_path = str(row.get("file_path") or row.get("config_path") or "").strip()
        return PluginConfigDoc(
            id=int(row["id"]),
            plugin_chinese_name=str(row.get("plugin_chinese_name") or "原文未明确说明"),
            plugin_english_name=str(row.get("plugin_english_name") or "原文未明确说明"),
            file_path=file_path,
            content=str(row.get("content") or ""),
            distance=distance,
            match_reason=match_reason,
        )
