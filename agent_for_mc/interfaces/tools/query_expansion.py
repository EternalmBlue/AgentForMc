from __future__ import annotations

import json
from contextvars import ContextVar
from dataclasses import dataclass

from langchain_core.tools import tool

from agent_for_mc.application.deepagent_state import record_standalone_query
from agent_for_mc.infrastructure.clients import DeepSeekChatClient


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


_TOOL_CONTEXT: ContextVar[QueryExpansionToolContext | None] = ContextVar(
    "query_expansion_tool_context",
    default=None,
)


def configure_query_expansion_tool(context: QueryExpansionToolContext) -> None:
    _TOOL_CONTEXT.set(context)


def get_query_expansion_tool_context() -> QueryExpansionToolContext:
    context = _TOOL_CONTEXT.get()
    if context is None:
        raise RuntimeError("query_expansion tool context has not been configured")
    return context


@tool("query_expansion")
def query_expansion(question: str, history: str = "") -> str:
    """Expand a short question into a fuller search intent and retrieval query."""
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
