from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import lancedb
import pyarrow as pa

from agent_for_mc.domain.errors import StartupValidationError
from agent_for_mc.domain.models import SemanticMemoryDoc, SemanticMemoryEntry, VectorStoreStats
from agent_for_mc.infrastructure.observability import record_counter, trace_operation


REQUIRED_FIELDS = {
    "server_id",
    "plugin_name",
    "memory_type",
    "relation_type",
    "memory_text",
    "embedding",
}


ALLOWED_MEMORY_TYPES = {
    "topology",
    "plugin_config",
    "default",
    "override",
    "dependency",
    "fact",
}

ALLOWED_RELATION_TYPES = {
    "belongs_to",
    "contains",
    "located_in",
    "overrides",
    "depends_on",
    "affects",
    "uses",
    "controls",
}


@dataclass(slots=True)
class _NameMatch:
    score: int
    row: dict


class LanceSemanticMemoryVectorStore:
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
        self._rows: list[dict] | None = None
        self._stats: VectorStoreStats | None = None

    def validate(self) -> VectorStoreStats:
        with trace_operation(
            "semantic_memory_vector_store.validate",
            attributes={
                "component": "semantic_memory_vector_store",
                "table": self._table_name,
            },
            metric_name="rag_semantic_memory_vector_store_validate_seconds",
        ):
            record_counter("rag_semantic_memory_vector_store_validate_requests_total")
            if self._stats is not None:
                return self._stats

            if not self._db_dir.exists():
                self._rows = []
                self._stats = VectorStoreStats(
                    db_dir=self._db_dir,
                    table_name=self._table_name,
                    record_count=0,
                    embedding_dimension=self._expected_embedding_dimension,
                )
                return self._stats

            self._db = lancedb.connect(str(self._db_dir))
            table = self._open_table()
            if table is None:
                self._rows = []
                self._stats = VectorStoreStats(
                    db_dir=self._db_dir,
                    table_name=self._table_name,
                    record_count=0,
                    embedding_dimension=self._expected_embedding_dimension,
                )
                return self._stats

            self._table = table
            try:
                arrow_table = self._table.to_arrow()
            except Exception as exc:
                raise StartupValidationError(f"读取语义记忆表失败: {exc}") from exc

            schema = arrow_table.schema
            actual_fields = {field.name for field in schema}
            missing_fields = sorted(REQUIRED_FIELDS - actual_fields)
            if missing_fields:
                raise StartupValidationError(
                    f"语义记忆表缺少必要字段: {', '.join(missing_fields)}"
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

    def upsert_bundle_entries(
        self,
        *,
        server_id: str,
        plugin_name: str,
        entries: list[SemanticMemoryEntry],
        embeddings: list[list[float]],
    ) -> None:
        if len(entries) != len(embeddings):
            raise ValueError("entries 和 embeddings 数量不一致。")
        if not entries:
            self.delete_bundle(server_id=server_id, plugin_name=plugin_name)
            return

        self._db_dir.mkdir(parents=True, exist_ok=True)
        self._db = lancedb.connect(str(self._db_dir))
        self._table = self._open_table()

        rows = _build_rows(entries, embeddings)
        self.delete_bundle(server_id=server_id, plugin_name=plugin_name)
        if self._table is None:
            self._table = self._db.create_table(
                self._table_name,
                schema=self._schema(),
                data=rows,
                mode="create",
            )
        else:
            self._table.add(pa.Table.from_pylist(rows, schema=self._schema()))
        self._invalidate_cache()

    def delete_bundle(self, *, server_id: str, plugin_name: str) -> None:
        if not self._db_dir.exists():
            return
        self._db = lancedb.connect(str(self._db_dir))
        self._table = self._open_table()
        if self._table is None:
            return
        where_clause = (
            f"server_id = '{_escape_sql_literal(server_id)}' AND "
            f"plugin_name = '{_escape_sql_literal(plugin_name)}'"
        )
        try:
            self._table.delete(where_clause)
        except Exception:
            return
        self._invalidate_cache()

    def find_name_matches(self, search_query: str) -> list[SemanticMemoryDoc]:
        with trace_operation(
            "semantic_memory_vector_store.find_name_matches",
            attributes={
                "component": "semantic_memory_vector_store",
                "query.length": len(search_query.strip()),
            },
            metric_name="rag_semantic_memory_vector_store_name_match_seconds",
        ):
            record_counter("rag_semantic_memory_vector_store_name_match_requests_total")
            rows = self._load_rows()
            if not rows:
                return []
            normalized_query = search_query.lower()
            matches: list[_NameMatch] = []

            for row in rows:
                score = 0
                for candidate in (
                    str(row.get("server_id") or "").strip(),
                    str(row.get("plugin_name") or "").strip(),
                    str(row.get("memory_text") or "").strip(),
                ):
                    if candidate and candidate.lower() in normalized_query:
                        score = max(score, len(candidate))
                if score > 0:
                    matches.append(_NameMatch(score=score, row=row))

            matches.sort(
                key=lambda item: (
                    item.score,
                    len(str(item.row.get("memory_text") or "")),
                    str(item.row.get("plugin_name") or ""),
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
        server_id: str | None = None,
        plugin_name: str | None = None,
    ) -> list[SemanticMemoryDoc]:
        with trace_operation(
            "semantic_memory_vector_store.search_by_embedding",
            attributes={
                "component": "semantic_memory_vector_store",
                "top_k": top_k,
            },
            metric_name="rag_semantic_memory_vector_store_search_seconds",
        ):
            record_counter("rag_semantic_memory_vector_store_search_requests_total")
            table = self._ensure_table_for_search()
            if table is None:
                return []
            query = table.search(embedding)
            where_clause = self._build_where_clause(
                server_id=server_id,
                plugin_name=plugin_name,
            )
            if where_clause:
                query = query.where(where_clause)
            try:
                rows = query.limit(top_k).to_list()
            except Exception as exc:
                raise StartupValidationError(f"语义记忆向量检索失败: {exc}") from exc

            return [
                self._row_to_doc(
                    row,
                    distance=float(row.get("_distance", 0.0)),
                    match_reason="vector",
                )
                for row in rows
            ]

    def _load_rows(self) -> list[dict]:
        if self._rows is not None:
            return self._rows
        stats = self.validate()
        if stats.record_count == 0:
            return []
        return self._rows or []

    def _ensure_table_for_search(self):
        if self._db_dir.exists():
            self._db = lancedb.connect(str(self._db_dir))
            self._table = self._open_table()
        return self._table

    def _open_table(self):
        try:
            return self._db.open_table(self._table_name)
        except Exception:
            return None

    def _invalidate_cache(self) -> None:
        self._rows = None
        self._stats = None

    def _row_to_doc(
        self,
        row: dict,
        *,
        distance: float,
        match_reason: str,
    ) -> SemanticMemoryDoc:
        return SemanticMemoryDoc(
            server_id=str(row.get("server_id") or ""),
            plugin_name=str(row.get("plugin_name") or ""),
            memory_type=str(row.get("memory_type") or ""),
            relation_type=str(row.get("relation_type") or ""),
            memory_text=str(row.get("memory_text") or ""),
            distance=distance,
            match_reason=match_reason,
        )

    def _schema(self) -> pa.Schema:
        return pa.schema(
            [
                pa.field("server_id", pa.string()),
                pa.field("plugin_name", pa.string()),
                pa.field("memory_type", pa.string()),
                pa.field("relation_type", pa.string()),
                pa.field("memory_text", pa.string()),
                pa.field(
                    "embedding",
                    pa.list_(pa.float32(), self._expected_embedding_dimension),
                ),
            ]
        )

    def _build_where_clause(
        self,
        *,
        server_id: str | None,
        plugin_name: str | None,
    ) -> str:
        clauses: list[str] = []
        if server_id:
            clauses.append(f"server_id = '{_escape_sql_literal(server_id)}'")
        if plugin_name:
            clauses.append(f"plugin_name = '{_escape_sql_literal(plugin_name)}'")
        return " AND ".join(clauses)


def _build_rows(
    entries: list[SemanticMemoryEntry],
    embeddings: list[list[float]],
) -> list[dict]:
    return [
        {
            "server_id": entry.server_id,
            "plugin_name": entry.plugin_name,
            "memory_type": entry.memory_type,
            "relation_type": entry.relation_type,
            "memory_text": entry.memory_text,
            "embedding": embedding,
        }
        for entry, embedding in zip(entries, embeddings, strict=True)
    ]


def _escape_sql_literal(value: str) -> str:
    return str(value).replace("'", "''")
