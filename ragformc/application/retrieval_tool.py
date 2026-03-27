from __future__ import annotations

from dataclasses import dataclass
from contextvars import ContextVar
from typing import Final

from ragformc.application.retrieval import merge_retrieved_docs
from ragformc.application.retrieval import Retriever
from ragformc.domain.models import RetrievedDoc


@dataclass(slots=True)
class RetrieveDocsToolContext:
    retriever: Retriever
    top_k: int
    citation_preview_chars: int


_TOOL_CONTEXT: Final[ContextVar[RetrieveDocsToolContext | None]] = ContextVar(
    "retrieve_docs_tool_context",
    default=None,
)


def configure_retrieve_docs_tool(context: RetrieveDocsToolContext) -> None:
    _TOOL_CONTEXT.set(context)


def get_retrieve_docs_tool_context() -> RetrieveDocsToolContext:
    context = _TOOL_CONTEXT.get()
    if context is None:
        raise RuntimeError("retrieve_docs tool context has not been configured")
    return context


def build_retrieve_docs_payload(
    query: str,
    *,
    context: RetrieveDocsToolContext,
) -> tuple[list[RetrievedDoc], str]:
    docs = context.retriever.retrieve(query, top_k=context.top_k)
    return docs, _format_docs_for_tool(docs, preview_chars=context.citation_preview_chars)


def build_multi_query_retrieve_docs_payload(
    queries: list[str],
    *,
    context: RetrieveDocsToolContext,
) -> tuple[list[RetrievedDoc], str]:
    merged_docs: list[RetrievedDoc] = []
    summary_parts: list[str] = []

    for index, query in enumerate(queries, start=1):
        docs, summary = build_retrieve_docs_payload(query, context=context)
        merged_docs = merge_retrieved_docs(merged_docs, docs)
        summary_parts.append(f"[Query {index}] {query}\n{summary}")

    if not summary_parts:
        return [], "No matching documents were found."

    return merged_docs, "\n\n".join(summary_parts)


def _format_docs_for_tool(
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
