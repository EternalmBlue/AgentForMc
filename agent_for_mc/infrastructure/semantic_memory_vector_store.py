from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import lancedb
import pyarrow as pa

from agent_for_mc.domain.errors import StartupValidationError
from agent_for_mc.domain.models import SemanticMemoryDoc, SemanticMemoryEntry
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

    def validate(self) -> None:
        with trace_operation(
            "semantic_memory_vector_store.validate",
            attributes={
                "component": "semantic_memory_vector_store",
                "table": self._table_name,
            },
            metric_name="rag_semantic_memory_vector_store_validate_seconds",
        ):
            record_counter("rag_semantic_memory_vector_store_validate_requests_total")
            if not self._db_dir.exists():
                raise StartupValidationError(
                    f"语义记忆向量库目录不存在: {self._db_dir}"
                )

            try:
                self._db = lancedb.connect(str(self._db_dir))
                self._table = self._db.open_table(self._table_name)
            except Exception as exc:
                raise StartupValidationError(
                    f"无法打开语义记忆表 {self._table_name}: {exc}"
                ) from exc

            try:
                arrow_table = self._table.to_arrow()
            except Exception as exc:
                raise StartupValidationError(
                    f"读取语义记忆表失败: {exc}"
                ) from exc

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

    def replace_entries(
        self,
        entries: list[SemanticMemoryEntry],
        embeddings: list[list[float]],
    ) -> None:
        if len(entries) != len(embeddings):
            raise ValueError("entries 与 embeddings 数量不一致")

        self._db_dir.mkdir(parents=True, exist_ok=True)
        rows = [
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

        schema = self._schema()
        self._db = lancedb.connect(str(self._db_dir))
        self._db.drop_table(self._table_name, ignore_missing=True)
        self._table = self._db.create_table(
            self._table_name,
            schema=schema,
            data=rows,
            mode="create",
        )
        self._rows = rows

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
            self.validate()
            query = self._table.search(embedding)
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
        self.validate()
        return self._rows or []

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


def _escape_sql_literal(value: str) -> str:
    return str(value).replace("'", "''")
