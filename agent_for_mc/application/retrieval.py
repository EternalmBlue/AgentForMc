from __future__ import annotations

import re
from dataclasses import replace

from agent_for_mc.domain.models import RetrievedDoc
from agent_for_mc.domain.errors import ServiceError
from agent_for_mc.infrastructure.clients import EmbeddingClient
from agent_for_mc.infrastructure.ranker import Ranker
from agent_for_mc.infrastructure.vector_store import LancePluginVectorStore
from agent_for_mc.infrastructure.observability import record_counter, trace_operation


WHITESPACE_RE = re.compile(r"\s+")
RRF_K = 60


def normalize_search_query(search_query: str) -> str:
    return WHITESPACE_RE.sub(" ", search_query).strip()


class Retriever:
    def __init__(
        self,
        vector_store: LancePluginVectorStore,
        embedding_client: EmbeddingClient,
        ranker: Ranker | None = None,
        *,
        bm25_enabled: bool = True,
        bm25_top_k: int | None = None,
        bm25_auto_create_index: bool = True,
    ):
        self._vector_store = vector_store
        self._embedding_client = embedding_client
        self._ranker = ranker
        self._bm25_enabled = bm25_enabled
        self._bm25_top_k = bm25_top_k
        self._bm25_auto_create_index = bm25_auto_create_index

    def retrieve(self, search_query: str, *, top_k: int = 8) -> list[RetrievedDoc]:
        with trace_operation(
            "retrieval.retrieve",
            attributes={"component": "retrieval", "query.length": len(search_query.strip())},
            metric_name="rag_retrieval_seconds",
        ):
            record_counter("rag_retrieval_requests_total")
            normalized_search_query = normalize_search_query(search_query)
            boosted_docs = self._vector_store.find_name_matches(normalized_search_query)
            query_embedding = self._embedding_client.embed_query(normalized_search_query)
            vector_docs = self._vector_store.search_by_embedding(
                query_embedding,
                top_k=top_k,
            )
            bm25_docs = self._search_by_bm25(normalized_search_query, top_k=top_k)
            merged_docs = merge_retrieved_docs(boosted_docs, vector_docs, bm25_docs)
            if self._ranker is not None:
                try:
                    reranked_docs = self._ranker.rank_docs(
                        normalized_search_query,
                        merged_docs,
                    )
                    return reranked_docs[:top_k]
                except ServiceError:
                    record_counter("rag_ranker_fallbacks_total")
            return self._fallback_results(boosted_docs, vector_docs, bm25_docs, top_k=top_k)

    def _fallback_results(
        self,
        boosted_docs: list[RetrievedDoc],
        vector_docs: list[RetrievedDoc],
        bm25_docs: list[RetrievedDoc],
        *,
        top_k: int,
    ) -> list[RetrievedDoc]:
        fused_docs = fuse_ranked_docs(vector_docs, bm25_docs)
        return merge_retrieved_docs(boosted_docs, fused_docs, top_k=top_k)

    def _search_by_bm25(self, search_query: str, *, top_k: int) -> list[RetrievedDoc]:
        if not self._bm25_enabled:
            return []
        search_by_bm25 = getattr(self._vector_store, "search_by_bm25", None)
        if search_by_bm25 is None:
            return []

        bm25_top_k = max(top_k, self._bm25_top_k or top_k)
        try:
            return search_by_bm25(
                search_query,
                top_k=bm25_top_k,
                auto_create_index=self._bm25_auto_create_index,
            )
        except Exception:
            record_counter("rag_retrieval_bm25_failures_total")
            return []


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


def fuse_ranked_docs(
    *doc_groups: list[RetrievedDoc],
    top_k: int | None = None,
) -> list[RetrievedDoc]:
    docs_by_id: dict[int, RetrievedDoc] = {}
    scores_by_id: dict[int, float] = {}
    first_seen_by_id: dict[int, int] = {}
    reasons_by_id: dict[int, set[str]] = {}
    sequence = 0

    for doc_group in doc_groups:
        for rank, doc in enumerate(doc_group, start=1):
            if doc.id not in docs_by_id:
                sequence += 1
                docs_by_id[doc.id] = doc
                first_seen_by_id[doc.id] = sequence
            scores_by_id[doc.id] = scores_by_id.get(doc.id, 0.0) + 1.0 / (RRF_K + rank)
            reasons_by_id.setdefault(doc.id, set()).add(doc.match_reason)

    ordered_ids = sorted(
        docs_by_id,
        key=lambda doc_id: (scores_by_id[doc_id], -first_seen_by_id[doc_id]),
        reverse=True,
    )
    fused_docs = [
        _with_fused_match_reason(docs_by_id[doc_id], reasons_by_id[doc_id])
        for doc_id in ordered_ids
    ]
    if top_k is not None:
        return fused_docs[:top_k]
    return fused_docs


def _with_fused_match_reason(doc: RetrievedDoc, reasons: set[str]) -> RetrievedDoc:
    ordered_reasons = [
        reason
        for reason in ("vector", "bm25")
        if reason in reasons
    ]
    if len(ordered_reasons) < 2:
        return doc
    return replace(doc, match_reason="+".join(ordered_reasons))
