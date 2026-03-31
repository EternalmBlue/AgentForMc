from __future__ import annotations

import json
from contextvars import ContextVar
from dataclasses import dataclass, field

from langchain_core.tools import tool

from agent_for_mc.application.deepagent_state import record_standalone_query
from agent_for_mc.interfaces.tools.routing.planning import get_planning_tool_context


SELECT_RETRIEVAL_TOOL_SYSTEM_PROMPT = """
You are a retrieval tool selector for a Minecraft plugin assistant.

Return JSON only with this schema:
{"retrieval_backend": string, "search_query": string, "reason": string, "confidence": number, "fallback_used": boolean}

Rules:
- Choose the best backend for the user's question from the available backends.
- Available backends may include:
  - chunk: semantic retrieval over document chunks.
  - fulltext: exact or keyword-style retrieval over full documents.
  - sql: structured lookup in tables or metadata stores.
  - api: live lookup against an application or service API.
  - web: up-to-date external information.
- Prefer chunk for ordinary documentation questions.
- Prefer fulltext for exact error strings, config keys, class names, and long phrases.
- Prefer sql for structured facts, records, and tabular metadata.
- Prefer api for live runtime state or service-specific lookups.
- Prefer web for current or external information that is unlikely to exist in local docs.
- If the ideal backend is unavailable, choose the closest available backend and set fallback_used true.
- Keep search_query concise but complete enough for retrieval.
- Do not include any explanation outside JSON.
""".strip()


@dataclass(slots=True)
class SelectRetrievalToolContext:
    available_backends: tuple[str, ...] = ("chunk",)
    default_backend: str = "chunk"


_TOOL_CONTEXT: ContextVar[SelectRetrievalToolContext | None] = ContextVar(
    "select_retrieval_tool_context",
    default=None,
)


def configure_select_retrieval_tool(context: SelectRetrievalToolContext) -> None:
    _TOOL_CONTEXT.set(context)


def get_select_retrieval_tool_context() -> SelectRetrievalToolContext:
    context = _TOOL_CONTEXT.get()
    if context is None:
        raise RuntimeError("select_retrieval_tool context has not been configured")
    return context


@tool("select_retrieval_tool")
def select_retrieval_tool(question: str, history: str = "", retrieval_summary: str = "") -> str:
    """Select the best retrieval backend for a question and produce a search query."""
    context = get_select_retrieval_tool_context()
    planning_client = get_planning_tool_context().client
    messages = [
        {"role": "system", "content": _build_system_prompt(context.available_backends)},
        {
            "role": "user",
            "content": (
                f"Recent conversation:\n{history.strip() or 'No history.'}\n\n"
                f"User question:\n{question.strip()}\n\n"
                f"Retrieval summary:\n{retrieval_summary.strip() or 'No retrieval summary.'}\n"
            ),
        },
    ]
    content = planning_client.chat(messages, temperature=0.0)
    parsed = _parse_selection_result(
        content,
        fallback_question=question,
        available_backends=context.available_backends,
        default_backend=context.default_backend,
    )
    record_standalone_query(parsed["search_query"])
    return json.dumps(parsed, ensure_ascii=False)


def _build_system_prompt(available_backends: tuple[str, ...]) -> str:
    allowed = ", ".join(available_backends) if available_backends else "chunk"
    return (
        f"{SELECT_RETRIEVAL_TOOL_SYSTEM_PROMPT}\n\n"
        f"Available backends for this request: {allowed}\n"
        "If a backend is not listed here, do not select it."
    )


def _parse_selection_result(
    content: str,
    *,
    fallback_question: str,
    available_backends: tuple[str, ...],
    default_backend: str,
) -> dict[str, object]:
    normalized = content.strip()
    try:
        data = json.loads(normalized)
    except json.JSONDecodeError:
        search_query = _normalize_text(fallback_question)
        backend = _resolve_backend(default_backend, available_backends)
        return {
            "retrieval_backend": backend,
            "search_query": search_query,
            "reason": "Selection output was not valid JSON.",
            "confidence": 0.0,
            "fallback_used": backend != default_backend,
        }

    retrieval_backend = _normalize_backend(
        data.get("retrieval_backend")
        or data.get("backend")
        or data.get("tool")
        or default_backend
    )
    search_query = (
        _normalize_text(data.get("search_query"))
        or _normalize_text(data.get("query"))
        or _normalize_text(data.get("standalone_query"))
        or _normalize_text(fallback_question)
    )
    reason = _normalize_text(data.get("reason")) or "No reason provided."
    confidence = _parse_confidence(data.get("confidence"))
    fallback_used = _parse_bool(data.get("fallback_used"))

    resolved_backend = _resolve_backend(retrieval_backend, available_backends)
    if resolved_backend != retrieval_backend:
        fallback_used = True
        if reason == "No reason provided.":
            reason = (
                f"Selected backend '{retrieval_backend}' is unavailable; "
                f"using '{resolved_backend}' instead."
            )
        else:
            reason = (
                f"{reason} Selected backend '{retrieval_backend}' is unavailable; "
                f"using '{resolved_backend}' instead."
            )

    return {
        "retrieval_backend": resolved_backend,
        "search_query": search_query,
        "reason": reason,
        "confidence": confidence,
        "fallback_used": fallback_used or resolved_backend != retrieval_backend,
    }


def _resolve_backend(selected_backend: str, available_backends: tuple[str, ...]) -> str:
    normalized_backend = _normalize_backend(selected_backend)
    if normalized_backend in available_backends:
        return normalized_backend
    if available_backends:
        return available_backends[0]
    return "chunk"


def _normalize_backend(value: object) -> str:
    return _normalize_text(value).lower()


def _parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1"}
    return bool(value)


def _parse_confidence(value: object) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    if confidence < 0.0:
        return 0.0
    if confidence > 1.0:
        return 1.0
    return confidence


def _normalize_text(value: object) -> str:
    return " ".join(str(value).split()).strip()
