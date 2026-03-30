from __future__ import annotations

import json
from contextvars import ContextVar
from dataclasses import dataclass

from langchain_core.tools import tool

from agent_for_mc.application.deepagent_state import get_turn_context
from agent_for_mc.application.retrieval_tool import format_docs_for_tool
from agent_for_mc.interfaces.tools.planning import get_planning_tool_context


JUDGE_ANSWER_QUALITY_SYSTEM_PROMPT = """
You are an answer quality judge for a Minecraft plugin assistant.

Return JSON only with this schema:
{"overall_score": number, "is_good_enough": boolean, "needs_retry": boolean, "retry_recommendation": string, "reason": string, "confidence": number}

Rules:
- Score the draft answer as a whole from 0.0 to 1.0.
- Consider factual grounding in retrieved docs, completeness, correctness, directness, and clarity.
- Lower the score when the answer is speculative, misses key requirements, conflicts with evidence, or does not resolve the user's question.
- Use retrieved docs as primary evidence.
- If the answer relies on unsupported general knowledge, lower the score unless the question is explicitly asking for general knowledge and the answer is clearly caveated.
- Set needs_retry true when another query or retrieval pass could materially improve the answer.
- Set retry_recommendation to one of "query_rewrite", "query_expansion", "subquery_decomposition", "multi_query_rag", "hyde_retrieve_docs", "retrieve_again", or "answer_as_is".
- Prefer query_rewrite when the question is ambiguous, uses pronouns, or needs a standalone search query.
- Prefer query_expansion when the question is short, underspecified, or missing context but stays on one topic.
- Prefer subquery_decomposition when the question contains multiple independent tasks or knowledge areas.
- Prefer multi_query_rag when the topic is broad and benefits from wider recall.
- Prefer hyde_retrieve_docs when direct retrieval is semantically weak and a hypothetical answer may help.
- Use retrieve_again when the answer quality issue is mostly evidence coverage or freshness rather than query formulation.
- Use answer_as_is when the answer is already complete and well grounded.
- Treat overall_score >= 0.8 with is_good_enough true as a strong pass.
- Treat overall_score < 0.6 or needs_retry true as a clear signal to retry retrieval or rewrite the query.
- If no useful docs are available, score based on the answer itself and mark needs_retry true when the answer depends on missing evidence.
- Keep the reason concise and concrete.
- Do not include any explanation outside JSON.
""".strip()


@dataclass(slots=True)
class JudgeAnswerQualityToolContext:
    pass


_TOOL_CONTEXT: ContextVar[JudgeAnswerQualityToolContext | None] = ContextVar(
    "judge_answer_quality_tool_context",
    default=None,
)


def configure_judge_answer_quality_tool(context: JudgeAnswerQualityToolContext) -> None:
    _TOOL_CONTEXT.set(context)


def get_judge_answer_quality_tool_context() -> JudgeAnswerQualityToolContext:
    context = _TOOL_CONTEXT.get()
    if context is None:
        raise RuntimeError("judge_answer_quality tool context has not been configured")
    return context


@tool("judge_answer_quality")
def judge_answer_quality(
    question: str,
    draft_answer: str,
    history: str = "",
    retrieval_summary: str = "",
) -> str:
    """Judge the overall quality of a draft answer and decide whether another retrieval pass is needed."""
    _ = get_judge_answer_quality_tool_context()
    turn = get_turn_context()
    docs = list(turn.retrieved_docs if turn else [])
    standalone_query = turn.standalone_query if turn and turn.standalone_query else ""
    docs_text = format_docs_for_tool(docs, preview_chars=120)

    planning_client = get_planning_tool_context().client
    messages = [
        {"role": "system", "content": JUDGE_ANSWER_QUALITY_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Recent conversation:\n{history.strip() or 'No history.'}\n\n"
                f"Standalone query:\n{standalone_query or 'No standalone query recorded.'}\n\n"
                f"User question:\n{question.strip()}\n\n"
                f"Draft answer:\n{draft_answer.strip()}\n\n"
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
            "overall_score": 0.0,
            "is_good_enough": False,
            "needs_retry": True,
            "retry_recommendation": "retrieve_again",
            "reason": "Judge output was not valid JSON.",
            "confidence": 0.0,
        }

    overall_score = _parse_score(data.get("overall_score"))
    is_good_enough = _parse_bool(data.get("is_good_enough"))
    needs_retry = _parse_bool(data.get("needs_retry"))
    retry_recommendation = _normalize_text(data.get("retry_recommendation")) or "retrieve_again"
    reason = _normalize_text(data.get("reason")) or "No reason provided."
    confidence = _parse_score(data.get("confidence"))

    return {
        "overall_score": overall_score,
        "is_good_enough": is_good_enough,
        "needs_retry": needs_retry,
        "retry_recommendation": retry_recommendation,
        "reason": reason,
        "confidence": confidence,
    }


def _parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1"}
    return bool(value)


def _parse_score(value: object) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return score


def _normalize_text(value: object) -> str:
    return " ".join(str(value).split()).strip()
