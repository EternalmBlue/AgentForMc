from __future__ import annotations

import re
from dataclasses import dataclass

from agent_for_mc.domain.models import SemanticMemoryDoc
from agent_for_mc.infrastructure.clients import JinaEmbeddingClient
from agent_for_mc.infrastructure.observability import record_counter, trace_operation
from agent_for_mc.infrastructure.semantic_memory_vector_store import (
    LanceSemanticMemoryVectorStore,
)


WHITESPACE_RE = re.compile(r"\s+")


def normalize_semantic_query(search_query: str) -> str:
    return WHITESPACE_RE.sub(" ", search_query).strip()


@dataclass(slots=True)
class SemanticMemoryRetriever:
    vector_store: LanceSemanticMemoryVectorStore
    embedding_client: JinaEmbeddingClient

    def retrieve(
        self,
        search_query: str,
        *,
        top_k: int = 8,
        server_id: str | None = None,
        plugin_name: str | None = None,
    ) -> list[SemanticMemoryDoc]:
        with trace_operation(
            "semantic_memory.retrieve",
            attributes={
                "component": "semantic_memory",
                "query.length": len(search_query.strip()),
            },
            metric_name="rag_semantic_memory_retrieve_seconds",
        ):
            record_counter("rag_semantic_memory_retrieve_requests_total")
            normalized_search_query = normalize_semantic_query(search_query)
            if not normalized_search_query:
                return []

            try:
                boosted_docs = self.vector_store.find_name_matches(normalized_search_query)
            except Exception:
                boosted_docs = []

            try:
                query_embedding = self.embedding_client.embed_query(normalized_search_query)
                vector_docs = self.vector_store.search_by_embedding(
                    query_embedding,
                    top_k=top_k,
                    server_id=server_id,
                    plugin_name=plugin_name,
                )
            except Exception:
                vector_docs = []

            return _merge_semantic_docs(boosted_docs, vector_docs, top_k=top_k)


def _merge_semantic_docs(
    boosted_docs: list[SemanticMemoryDoc],
    vector_docs: list[SemanticMemoryDoc],
    *,
    top_k: int,
) -> list[SemanticMemoryDoc]:
    merged: list[SemanticMemoryDoc] = []
    seen: set[tuple[str, str, str, str]] = set()

    for doc in [*boosted_docs, *vector_docs]:
        key = (
            doc.server_id,
            doc.plugin_name,
            doc.memory_type,
            doc.relation_type,
        )
        if key in seen:
            continue
        seen.add(key)
        merged.append(doc)

    merged.sort(
        key=lambda doc: (
            doc.match_reason != "name-boost",
            doc.distance,
            len(doc.memory_text),
        ),
        reverse=False,
    )
    return merged[:top_k]
