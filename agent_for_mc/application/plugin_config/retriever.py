from __future__ import annotations

import re
from dataclasses import dataclass

from agent_for_mc.application.retrieval import merge_retrieved_docs
from agent_for_mc.domain.models import PluginConfigDoc
from agent_for_mc.infrastructure.clients import JinaEmbeddingClient
from agent_for_mc.infrastructure.observability import record_counter, trace_operation
from agent_for_mc.infrastructure.plugin_config_vector_store import (
    LancePluginConfigVectorStore,
)
from agent_for_mc.infrastructure.ranker import BceRanker


WHITESPACE_RE = re.compile(r"\s+")


def normalize_search_query(search_query: str) -> str:
    return WHITESPACE_RE.sub(" ", search_query).strip()


@dataclass(slots=True)
class PluginConfigRetriever:
    vector_store: LancePluginConfigVectorStore
    embedding_client: JinaEmbeddingClient
    ranker: BceRanker | None = None

    def retrieve(self, search_query: str, *, top_k: int = 8) -> list[PluginConfigDoc]:
        with trace_operation(
            "plugin_config.retrieve",
            attributes={"component": "plugin_config", "query.length": len(search_query.strip())},
            metric_name="rag_plugin_config_retrieve_seconds",
        ):
            record_counter("rag_plugin_config_retrieve_requests_total")
            normalized_search_query = normalize_search_query(search_query)
            boosted_docs = self.vector_store.find_name_matches(normalized_search_query)
            query_embedding = self.embedding_client.embed_query(normalized_search_query)
            vector_docs = self.vector_store.search_by_embedding(
                query_embedding,
                top_k=top_k,
            )
            merged_docs = merge_retrieved_docs(boosted_docs, vector_docs)
            if self.ranker is not None:
                reranked_docs = self.ranker.rank_docs(normalized_search_query, merged_docs)
                return reranked_docs[:top_k]
            return merge_retrieved_docs(boosted_docs, vector_docs, top_k=top_k)
