from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import Future, ThreadPoolExecutor, as_completed

from agent_for_mc.application.deepagent_state import record_retrieved_docs
from agent_for_mc.application.retrieval import merge_retrieved_docs
from agent_for_mc.application.retrieval import Retriever
from agent_for_mc.domain.models import RetrievedDoc
from agent_for_mc.infrastructure.observability import record_counter, trace_operation
from agent_for_mc.infrastructure.shared_context import SharedContextSlot


@dataclass(slots=True)
class RetrieveDocsToolContext:
    retriever: Retriever
    top_k: int
    citation_preview_chars: int


_TOOL_CONTEXT = SharedContextSlot[RetrieveDocsToolContext]("retrieve_docs_tool_context")


def configure_retrieve_docs_tool(context: RetrieveDocsToolContext) -> None:
    _TOOL_CONTEXT.set(context)


def get_retrieve_docs_tool_context() -> RetrieveDocsToolContext:
    return _TOOL_CONTEXT.get(
        error_message="retrieve_docs tool context has not been configured"
    )


def build_retrieve_docs_payload(
    search_query: str,
    *,
    context: RetrieveDocsToolContext,
) -> tuple[list[RetrievedDoc], str]:
    with trace_operation(
        "retrieve_docs",
        attributes={"component": "retrieval", "query.length": len(search_query.strip())},
        metric_name="rag_retrieve_docs_seconds",
    ):
        record_counter("rag_retrieve_docs_requests_total")
        docs = context.retriever.retrieve(search_query, top_k=context.top_k)
        record_retrieved_docs(docs)
        return docs, format_docs_for_tool(
            docs,
            preview_chars=context.citation_preview_chars,
        )


def build_multi_query_retrieve_docs_payload(
    search_queries: list[str],
    *,
    context: RetrieveDocsToolContext,
) -> tuple[list[RetrievedDoc], str]:
    with trace_operation(
        "multi_query_retrieve_docs",
        attributes={"component": "retrieval", "query.count": len(search_queries)},
        metric_name="rag_multi_query_retrieve_docs_seconds",
    ):
        normalized_queries = [
            query.strip() for query in search_queries if query and query.strip()
        ]
        record_counter("rag_multi_query_retrieve_docs_requests_total")
        if not normalized_queries:
            return [], "No matching documents were found."

        merged_docs: list[RetrievedDoc] = []
        summary_parts: list[str] = []
        max_workers = min(len(normalized_queries), 4)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_query: dict[Future[tuple[list[RetrievedDoc], str]], tuple[int, str]] = {}
            for index, query in enumerate(normalized_queries, start=1):
                future = executor.submit(
                    build_retrieve_docs_payload,
                    query,
                    context=context,
                )
                future_to_query[future] = (index, query)

            results: list[tuple[int, str, list[RetrievedDoc], str]] = []
            for future in as_completed(future_to_query):
                index, query = future_to_query[future]
                docs, summary = future.result()
                results.append((index, query, docs, summary))

        results.sort(key=lambda item: item[0])
        for index, query, docs, summary in results:
            merged_docs = merge_retrieved_docs(merged_docs, docs)
            summary_parts.append(f"[Query {index}] {query}\n{summary}")

        if not summary_parts:
            return [], "No matching documents were found."

        record_retrieved_docs(merged_docs)
        return merged_docs, "\n\n".join(summary_parts)


def format_docs_for_tool(
    docs: list[RetrievedDoc],
    *,
    preview_chars: int,
) -> str:
    if not docs:
        return "No matching documents were found."

    parts: list[str] = []
    for index, doc in enumerate(docs, start=1):
        preview = " ".join(doc.content.split())
        if preview_chars > 0:
            preview = preview[:preview_chars]
        parts.append(
            f"[{index}] id={doc.id} "
            f"name={doc.plugin_chinese_name} / {doc.plugin_english_name}\n"
            f"reason={doc.match_reason}\n"
            f"distance={doc.distance:.6f}\n"
            f"preview={preview}"
        )
    return "\n\n".join(parts)
