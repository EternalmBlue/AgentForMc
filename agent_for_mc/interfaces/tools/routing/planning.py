from __future__ import annotations

import json
from contextvars import ContextVar
from dataclasses import dataclass

from langchain_core.tools import tool

from agent_for_mc.application.deepagent_state import record_standalone_query
from agent_for_mc.infrastructure.clients import DeepSeekChatClient


ANALYZE_QUERY_SYSTEM_PROMPT = """
You are a planning tool for a Minecraft plugin assistant.

Return JSON only with this schema:
{"standalone_query": string, "need_multi_query": boolean, "queries": array, "need_plugins": boolean}

Rules:
- Rewrite the question into a standalone search query.
- Set need_multi_query to true only when the question is ambiguous, broad, or missing key context.
- When need_multi_query is true, generate 3 to 4 diverse search queries.
- Set need_plugins to true when the answer depends on installed server plugins or plugin availability.
- Keep standalone_query concise and preserve plugin names, aliases, versions, and class names.
- Do not include any explanation outside JSON.
""".strip()


@dataclass(slots=True)
class PlanningToolContext:
    client: DeepSeekChatClient


_TOOL_CONTEXT: ContextVar[PlanningToolContext | None] = ContextVar(
    "planning_tool_context",
    default=None,
)


def configure_planning_tool(context: PlanningToolContext) -> None:
    _TOOL_CONTEXT.set(context)


def get_planning_tool_context() -> PlanningToolContext:
    context = _TOOL_CONTEXT.get()
    if context is None:
        raise RuntimeError("planning tool context has not been configured")
    return context


@tool("analyze_question")
def analyze_question(
    question: str,
    history: str = "",
    retrieval_summary: str = "",
) -> str:
    """Plan rewriting, multi-query need, and plugin checks for a question."""
    context = get_planning_tool_context()
    messages = [
        {"role": "system", "content": ANALYZE_QUERY_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Recent conversation:\n{history.strip() or 'No history.'}\n\n"
                f"User question:\n{question.strip()}\n\n"
                f"Retrieval summary:\n{retrieval_summary.strip() or 'No retrieval summary.'}\n"
            ),
        },
    ]
    content = context.client.chat(messages, temperature=0.0)
    parsed = _parse_analysis_result(content, fallback_question=question)
    record_standalone_query(str(parsed["standalone_query"]))
    return json.dumps(parsed, ensure_ascii=False)


def _parse_analysis_result(content: str, *, fallback_question: str) -> dict[str, object]:
    normalized = content.strip()
    try:
        data = json.loads(normalized)
    except json.JSONDecodeError:
        return {
            "standalone_query": fallback_question.strip(),
            "need_multi_query": False,
            "queries": [],
            "need_plugins": False,
        }

    standalone_query = (
        _normalize_text(data.get("standalone_query"))
        or _normalize_text(data.get("rewritten_question"))
        or fallback_question
    )
    need_multi_query = _parse_bool(data.get("need_multi_query"))
    need_plugins = _parse_bool(data.get("need_plugins"))

    raw_queries = data.get("queries") or data.get("multi_queries") or []
    if isinstance(raw_queries, str):
        raw_queries = [raw_queries]

    queries: list[str] = []
    if isinstance(raw_queries, list):
        queries = _normalize_queries([str(query) for query in raw_queries])

    if need_multi_query and len(queries) < 3:
        queries = _normalize_queries(
            [
                standalone_query,
                fallback_question,
                f"{fallback_question} related plugins, features, and config",
                f"{fallback_question} implementation and version differences",
            ]
        )

    return {
        "standalone_query": standalone_query,
        "need_multi_query": need_multi_query,
        "queries": queries[:4] if need_multi_query else [],
        "need_plugins": need_plugins,
    }


def _parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1"}
    return bool(value)


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
