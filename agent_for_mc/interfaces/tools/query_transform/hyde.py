from __future__ import annotations

import json
from dataclasses import dataclass

from langchain_core.tools import tool

from agent_for_mc.application.deepagent_state import record_standalone_query
from agent_for_mc.application.retrieval_tool import (
    build_retrieve_docs_payload,
    format_docs_for_tool,
    get_retrieve_docs_tool_context,
)
from agent_for_mc.infrastructure.shared_context import SharedContextSlot
from agent_for_mc.interfaces.tools.routing.planning import get_planning_tool_context


HYDE_SYSTEM_PROMPT = """
You are a HyDE planner for a Minecraft plugin assistant.

Return JSON only with this schema:
{"hypothetical_answer": string}

Rules:
- Write a short hypothetical answer that would likely appear in relevant documentation.
- Make it self-contained and focused on the user's request.
- Preserve plugin names, aliases, versions, and class names when present.
- Do not mention that the answer is hypothetical.
- Do not include any explanation outside JSON.
""".strip()


@dataclass(slots=True)
class HydeToolContext:
    pass


_TOOL_CONTEXT = SharedContextSlot[HydeToolContext]("hyde_tool_context")


def configure_hyde_tool(context: HydeToolContext) -> None:
    _TOOL_CONTEXT.set(context)


def get_hyde_tool_context() -> HydeToolContext:
    return _TOOL_CONTEXT.get(error_message="hyde tool context has not been configured")


@tool("hyde_retrieve_docs")
def hyde_retrieve_docs(question: str, history: str = "") -> str:
    """Generate a hypothetical answer and retrieve docs using it as the search query."""
    _ = get_hyde_tool_context()
    planning_client = get_planning_tool_context().client
    messages = [
        {"role": "system", "content": HYDE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Recent conversation:\n{history.strip() or 'No history.'}\n\n"
                f"User question:\n{question.strip()}\n"
            ),
        },
    ]
    content = planning_client.chat(messages, temperature=0.0)
    parsed = _parse_hyde_plan(content, fallback_question=question)
    hypothetical_answer = parsed["hypothetical_answer"]
    record_standalone_query(hypothetical_answer)

    context = get_retrieve_docs_tool_context()
    docs, summary = build_retrieve_docs_payload(
        hypothetical_answer,
        context=context,
    )
    formatted_docs = format_docs_for_tool(
        docs,
        preview_chars=context.citation_preview_chars,
    )
    return (
        f"hypothetical_answer: {hypothetical_answer}\n"
        f"search_query: {hypothetical_answer}\n\n"
        f"{summary}\n\n"
        f"{formatted_docs}"
    )


def _parse_hyde_plan(content: str, *, fallback_question: str) -> dict[str, str]:
    normalized = content.strip()
    try:
        data = json.loads(normalized)
    except json.JSONDecodeError:
        return {"hypothetical_answer": _normalize_text(fallback_question)}

    hypothetical_answer = (
        _normalize_text(data.get("hypothetical_answer"))
        or _normalize_text(data.get("answer"))
        or _normalize_text(data.get("response"))
        or _normalize_text(fallback_question)
    )
    return {"hypothetical_answer": hypothetical_answer}


def _normalize_text(value: object) -> str:
    return " ".join(str(value).split()).strip()
