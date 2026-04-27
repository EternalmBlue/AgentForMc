from __future__ import annotations

import json
from dataclasses import dataclass

from langchain_core.tools import tool

from agent_for_mc.application.deepagent_state import record_standalone_query
from agent_for_mc.infrastructure.clients import DeepSeekChatClient
from agent_for_mc.infrastructure.observability import record_counter, trace_operation
from agent_for_mc.infrastructure.shared_context import SharedContextSlot


QUERY_EXPANSION_SYSTEM_PROMPT = """
You are a query expansion tool for a Minecraft plugin assistant.

Return JSON only with this schema:
{"expanded_question": string, "search_query": string}

Rules:
- Expand a short or underspecified question into a fuller, more explicit request.
- Restore missing context, related terminology, and likely user intent when it is clearly implied.
- Keep plugin names, aliases, versions, and class names unchanged when present.
- Do not add unsupported facts.
- Keep the search_query concise enough for retrieval while staying more complete than the original question.
- Do not include any explanation outside JSON.
""".strip()


@dataclass(slots=True)
class QueryExpansionToolContext:
    client: DeepSeekChatClient


_TOOL_CONTEXT = SharedContextSlot[QueryExpansionToolContext](
    "query_expansion_tool_context"
)


def configure_query_expansion_tool(context: QueryExpansionToolContext) -> None:
    _TOOL_CONTEXT.set(context)


def get_query_expansion_tool_context() -> QueryExpansionToolContext:
    return _TOOL_CONTEXT.get(
        error_message="query_expansion tool context has not been configured"
    )


@tool("query_expansion")
def query_expansion(question: str, history: str = "") -> str:
    """Expand a short question into a fuller search intent and retrieval query."""
    with trace_operation(
        "tool.query_expansion",
        attributes={"component": "tool", "question.length": len(question.strip())},
        metric_name="rag_tool_query_expansion_seconds",
    ):
        record_counter("rag_tool_query_expansion_requests_total")
        context = get_query_expansion_tool_context()
        messages = [
            {"role": "system", "content": QUERY_EXPANSION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Recent conversation:\n{history.strip() or 'No history.'}\n\n"
                    f"User question:\n{question.strip()}\n"
                ),
            },
        ]
        content = context.client.chat(messages, temperature=0.0)
        parsed = _parse_query_expansion(content, fallback_question=question)
        expanded_question = parsed["expanded_question"]
        search_query = parsed["search_query"]
        record_standalone_query(search_query)
        return (
            f"expanded_question: {expanded_question}\n"
            f"search_query: {search_query}"
        )


def _parse_query_expansion(content: str, *, fallback_question: str) -> dict[str, str]:
    normalized = content.strip()
    try:
        data = json.loads(normalized)
    except json.JSONDecodeError:
        fallback = _normalize_text(fallback_question)
        return {
            "expanded_question": fallback,
            "search_query": fallback,
        }

    expanded_question = (
        _normalize_text(data.get("expanded_question"))
        or _normalize_text(data.get("expanded_query"))
        or _normalize_text(data.get("rewritten_question"))
        or _normalize_text(fallback_question)
    )
    search_query = (
        _normalize_text(data.get("search_query"))
        or _normalize_text(data.get("query"))
        or expanded_question
    )
    return {
        "expanded_question": expanded_question,
        "search_query": search_query,
    }


def _normalize_text(value: object) -> str:
    return " ".join(str(value).split()).strip()
