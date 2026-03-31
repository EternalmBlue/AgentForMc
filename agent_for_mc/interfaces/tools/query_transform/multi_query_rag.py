from __future__ import annotations

import json
from contextvars import ContextVar
from dataclasses import dataclass

from langchain_core.tools import tool

from agent_for_mc.application.deepagent_state import record_standalone_query
from agent_for_mc.application.retrieval_tool import (
    build_multi_query_retrieve_docs_payload,
    format_docs_for_tool,
    get_retrieve_docs_tool_context,
)
from agent_for_mc.interfaces.tools.routing.planning import get_planning_tool_context


MULTI_QUERY_RAG_SYSTEM_PROMPT = """
You are a multi-query RAG planning tool for a Minecraft plugin assistant.

Return JSON only with this schema:
{"standalone_query": string, "queries": array}

Rules:
- Rewrite the question into a standalone search query.
- Generate 3 to 4 diverse search queries.
- Keep the queries semantically different but still relevant.
- Preserve plugin names, aliases, versions, and class names.
- Do not include any explanation outside JSON.
""".strip()


@dataclass(slots=True)
class MultiQueryRagToolContext:
    pass


_TOOL_CONTEXT: ContextVar[MultiQueryRagToolContext | None] = ContextVar(
    "multi_query_rag_tool_context",
    default=None,
)


def configure_multi_query_rag_tool(context: MultiQueryRagToolContext) -> None:
    _TOOL_CONTEXT.set(context)


def get_multi_query_rag_tool_context() -> MultiQueryRagToolContext:
    context = _TOOL_CONTEXT.get()
    if context is None:
        raise RuntimeError("multi_query_rag tool context has not been configured")
    return context


@tool("multi_query_rag")
def multi_query_rag(question: str, history: str = "") -> str:
    """Generate multiple search queries, retrieve docs in parallel, and merge results."""
    _ = get_multi_query_rag_tool_context()
    planning_client = get_planning_tool_context().client
    messages = [
        {"role": "system", "content": MULTI_QUERY_RAG_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Recent conversation:\n{history.strip() or 'No history.'}\n\n"
                f"User question:\n{question.strip()}\n"
            ),
        },
    ]
    content = planning_client.chat(messages, temperature=0.0)
    parsed = _parse_multi_query_rag_plan(content, fallback_question=question)
    record_standalone_query(parsed["standalone_query"])

    context = get_retrieve_docs_tool_context()
    docs, summary = build_multi_query_retrieve_docs_payload(
        parsed["queries"],
        context=context,
    )
    formatted_docs = format_docs_for_tool(
        docs,
        preview_chars=context.citation_preview_chars,
    )
    return (
        f"standalone_query: {parsed['standalone_query']}\n"
        f"queries: {json.dumps(parsed['queries'], ensure_ascii=False)}\n\n"
        f"{summary}\n\n"
        f"{formatted_docs}"
    )


def _parse_multi_query_rag_plan(content: str, *, fallback_question: str) -> dict[str, object]:
    normalized = content.strip()
    try:
        data = json.loads(normalized)
    except json.JSONDecodeError:
        return {
            "standalone_query": fallback_question.strip(),
            "queries": _fallback_queries(fallback_question),
        }

    standalone_query = (
        _normalize_text(data.get("standalone_query"))
        or _normalize_text(data.get("rewritten_question"))
        or fallback_question.strip()
    )
    raw_queries = data.get("queries") or data.get("multi_queries") or []
    if isinstance(raw_queries, str):
        raw_queries = [raw_queries]

    queries: list[str] = []
    if isinstance(raw_queries, list):
        queries = _normalize_queries([str(query) for query in raw_queries])

    if len(queries) < 3:
        queries = _fallback_queries(standalone_query, fallback_question)

    return {
        "standalone_query": standalone_query,
        "queries": queries[:4],
    }


def _fallback_queries(standalone_query: str, fallback_question: str | None = None) -> list[str]:
    base = _normalize_text(standalone_query) or _normalize_text(fallback_question or "")
    candidates = [
        base,
        f"{base} related plugins and config",
        f"{base} implementation details",
        f"{base} version differences",
    ]
    return _normalize_queries(candidates)


def _normalize_text(value: object) -> str:
    return " ".join(str(value).split()).strip()


def _normalize_queries(queries: list[str]) -> list[str]:
    unique_queries: list[str] = []
    seen: set[str] = set()
    for query in queries:
        normalized = _normalize_text(query)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique_queries.append(normalized)
    return unique_queries
