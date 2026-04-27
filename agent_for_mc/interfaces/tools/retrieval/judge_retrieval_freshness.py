from __future__ import annotations

import json
from dataclasses import dataclass

from langchain_core.tools import tool

from agent_for_mc.application.deepagent_state import get_turn_context
from agent_for_mc.application.retrieval_tool import format_docs_for_tool
from agent_for_mc.infrastructure.shared_context import SharedContextSlot
from agent_for_mc.interfaces.tools.routing.planning import get_planning_tool_context


JUDGE_RETRIEVAL_FRESHNESS_SYSTEM_PROMPT = """
You are a retrieval freshness judge for a Minecraft plugin assistant.

Return JSON only with this schema:
{"is_fresh_enough": boolean, "is_covered_enough": boolean, "needs_model_knowledge_fallback": boolean, "reason": string, "confidence": number}

Rules:
- Judge whether the retrieved docs are fresh enough and cover the user's question.
- Use the retrieved docs as the primary evidence.
- Mark is_fresh_enough false when the docs are clearly stale, version-bound, or likely outdated for the user's question.
- Mark is_covered_enough false when the docs do not answer the question or miss a key subtopic.
- Set needs_model_knowledge_fallback true when the docs are stale or incomplete but the question can still be answered cautiously from model knowledge.
- If no useful docs are available, set both booleans to false and needs_model_knowledge_fallback to true.
- Keep the reason concise and concrete.
- Do not include any explanation outside JSON.
""".strip()


@dataclass(slots=True)
class JudgeRetrievalFreshnessToolContext:
    pass


_TOOL_CONTEXT = SharedContextSlot[JudgeRetrievalFreshnessToolContext](
    "judge_retrieval_freshness_tool_context"
)


def configure_judge_retrieval_freshness_tool(
    context: JudgeRetrievalFreshnessToolContext,
) -> None:
    _TOOL_CONTEXT.set(context)


def get_judge_retrieval_freshness_tool_context() -> JudgeRetrievalFreshnessToolContext:
    return _TOOL_CONTEXT.get(
        error_message="judge_retrieval_freshness tool context has not been configured"
    )


@tool("judge_retrieval_freshness")
def judge_retrieval_freshness(question: str, retrieval_summary: str = "") -> str:
    """Judge whether retrieved docs are fresh and sufficient, and whether model knowledge should supplement them."""
    _ = get_judge_retrieval_freshness_tool_context()
    turn = get_turn_context()
    docs = list(turn.retrieved_docs if turn else [])
    standalone_query = turn.standalone_query if turn and turn.standalone_query else ""
    docs_text = format_docs_for_tool(docs, preview_chars=120)
    if not docs:
        return json.dumps(
            {
                "is_fresh_enough": False,
                "is_covered_enough": False,
                "needs_model_knowledge_fallback": True,
                "reason": "No retrieved docs available.",
                "confidence": 0.0,
            },
            ensure_ascii=False,
        )

    planning_client = get_planning_tool_context().client
    messages = [
        {"role": "system", "content": JUDGE_RETRIEVAL_FRESHNESS_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Standalone query:\n{standalone_query or 'No standalone query recorded.'}\n\n"
                f"User question:\n{question.strip()}\n\n"
                f"Retrieval summary:\n{retrieval_summary.strip() or 'No retrieval summary.'}\n\n"
                f"Retrieved docs:\n{docs_text}"
            ),
        },
    ]
    content = planning_client.chat(messages, temperature=0.0)
    parsed = _parse_judge_result(content)
    return json.dumps(parsed, ensure_ascii=False)


def _parse_judge_result(content: str) -> dict[str, object]:
    normalized = content.strip()
    try:
        data = json.loads(normalized)
    except json.JSONDecodeError:
        return {
            "is_fresh_enough": False,
            "is_covered_enough": False,
            "needs_model_knowledge_fallback": True,
            "reason": "Judge output was not valid JSON.",
            "confidence": 0.0,
        }

    is_fresh_enough = _parse_bool(data.get("is_fresh_enough"))
    is_covered_enough = _parse_bool(data.get("is_covered_enough"))
    needs_model_knowledge_fallback = _parse_bool(
        data.get("needs_model_knowledge_fallback")
    )
    reason = _normalize_text(data.get("reason")) or "No reason provided."
    confidence = _parse_confidence(data.get("confidence"))

    return {
        "is_fresh_enough": is_fresh_enough,
        "is_covered_enough": is_covered_enough,
        "needs_model_knowledge_fallback": needs_model_knowledge_fallback,
        "reason": reason,
        "confidence": confidence,
    }


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
