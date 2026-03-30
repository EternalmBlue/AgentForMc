from __future__ import annotations

import re

from agent_for_mc.domain.models import RetrievedDoc
from agent_for_mc.infrastructure.clients import JinaEmbeddingClient
from agent_for_mc.infrastructure.vector_store import LancePluginVectorStore


WHITESPACE_RE = re.compile(r"\s+")


def normalize_search_query(search_query: str) -> str:
    return WHITESPACE_RE.sub(" ", search_query).strip()


class Retriever:
    def __init__(
        self,
        vector_store: LancePluginVectorStore,
        embedding_client: JinaEmbeddingClient,
    ):
        self._vector_store = vector_store
        self._embedding_client = embedding_client

    def retrieve(self, search_query: str, *, top_k: int = 8) -> list[RetrievedDoc]:
        normalized_search_query = normalize_search_query(search_query)
        boosted_docs = self._vector_store.find_name_matches(normalized_search_query)
        query_embedding = self._embedding_client.embed_query(normalized_search_query)
        vector_docs = self._vector_store.search_by_embedding(
            query_embedding,
            top_k=top_k,
        )
        return merge_retrieved_docs(boosted_docs, vector_docs, top_k=top_k)


def merge_retrieved_docs(
    *doc_groups: list[RetrievedDoc],
    top_k: int | None = None,
) -> list[RetrievedDoc]:
    merged: list[RetrievedDoc] = []
    seen_ids: set[int] = set()

    for doc_group in doc_groups:
        for doc in doc_group:
            if doc.id in seen_ids:
                continue
            seen_ids.add(doc.id)
            merged.append(doc)
            if top_k is not None and len(merged) >= top_k:
                return merged
    return merged
