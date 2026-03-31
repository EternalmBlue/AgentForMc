from __future__ import annotations

import json
import re

from langchain_core.tools import tool

from agent_for_mc.application.deepagent_state import record_standalone_query
from agent_for_mc.application.retrieval_tool import (
    build_multi_query_retrieve_docs_payload,
    format_docs_for_tool,
    get_retrieve_docs_tool_context,
)
from agent_for_mc.interfaces.tools.routing.planning import get_planning_tool_context


SUBQUERY_DECOMPOSITION_SYSTEM_PROMPT = """
You are a sub-query decomposition tool for a Minecraft plugin assistant.

Return JSON only with this schema:
{"decomposition_question": string, "subqueries": array}

Rules:
- Normalize the user request into a concise standalone decomposition question.
- Break complex requests into 2 to 4 independent subqueries.
- Use subqueries when the request mixes distinct intents, tasks, or knowledge areas.
- Each subquery should be answerable on its own.
- Each item in subqueries should look like:
  {"subquestion": string, "search_query": string}
- Keep search_query concise and preserve plugin names, aliases, versions, and class names.
- Do not include any explanation outside JSON.
""".strip()


@tool("subquery_decomposition")
def subquery_decomposition(question: str, history: str = "") -> str:
    """Split a complex question into subqueries, retrieve docs, and merge the evidence."""
    planning_client = get_planning_tool_context().client
    messages = [
        {"role": "system", "content": SUBQUERY_DECOMPOSITION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Recent conversation:\n{history.strip() or 'No history.'}\n\n"
                f"User question:\n{question.strip()}\n"
            ),
        },
    ]
    content = planning_client.chat(messages, temperature=0.0)
    parsed = _parse_subquery_plan(content, fallback_question=question)
    record_standalone_query(parsed["decomposition_question"])

    context = get_retrieve_docs_tool_context()
    subqueries = parsed["subqueries"]
    docs, summary = build_multi_query_retrieve_docs_payload(
        [item["search_query"] for item in subqueries],
        context=context,
    )
    formatted_docs = format_docs_for_tool(
        docs,
        preview_chars=context.citation_preview_chars,
    )
    return (
        f"decomposition_question: {parsed['decomposition_question']}\n"
        f"subqueries:\n{_format_subqueries(subqueries)}\n\n"
        f"{summary}\n\n"
        f"{formatted_docs}"
    )


def _parse_subquery_plan(content: str, *, fallback_question: str) -> dict[str, object]:
    normalized = content.strip()
    try:
        data = json.loads(normalized)
    except json.JSONDecodeError:
        return {
            "decomposition_question": fallback_question.strip(),
            "subqueries": _fallback_subqueries(fallback_question),
        }

    decomposition_question = (
        _normalize_text(data.get("decomposition_question"))
        or _normalize_text(data.get("rewritten_question"))
        or fallback_question.strip()
    )
    raw_subqueries = data.get("subqueries") or data.get("sub_questions") or data.get("queries") or []
    if isinstance(raw_subqueries, str):
        raw_subqueries = [raw_subqueries]

    subqueries: list[dict[str, str]] = []
    if isinstance(raw_subqueries, list):
        for raw_item in raw_subqueries:
            subquery = _normalize_subquery_item(raw_item)
            if subquery is None:
                continue
            subqueries.append(subquery)

    if len(subqueries) < 2:
        subqueries = _fallback_subqueries(decomposition_question, fallback_question)

    return {
        "decomposition_question": decomposition_question,
        "subqueries": subqueries[:4],
    }


def _normalize_subquery_item(item: object) -> dict[str, str] | None:
    if isinstance(item, dict):
        subquestion = _normalize_text(
            item.get("subquestion")
            or item.get("question")
            or item.get("topic")
            or item.get("intent")
        )
        search_query = _normalize_text(
            item.get("search_query")
            or item.get("query")
            or item.get("keyword")
            or subquestion
        )
    else:
        subquestion = _normalize_text(item)
        search_query = subquestion

    if not subquestion:
        return None
    if not search_query:
        search_query = subquestion
    return {
        "subquestion": subquestion,
        "search_query": search_query,
    }


def _fallback_subqueries(base_question: str, fallback_question: str | None = None) -> list[dict[str, str]]:
    base = _normalize_text(base_question) or _normalize_text(fallback_question or "")
    if not base:
        return []

    candidates = _split_compound_question(base)
    if len(candidates) < 2:
        candidates = [base]

    subqueries: list[dict[str, str]] = []
    for candidate in candidates[:4]:
        normalized = _normalize_text(candidate)
        if not normalized:
            continue
        subqueries.append(
            {
                "subquestion": normalized,
                "search_query": normalized,
            }
        )
    return subqueries or [{"subquestion": base, "search_query": base}]


def _split_compound_question(question: str) -> list[str]:
    parts = re.split(
        r"\s*(?:\+|/|\\|&|、|，|,|;|；|\n|\b(?:and|or)\b)\s*",
        question,
        flags=re.IGNORECASE,
    )
    return [part for part in parts if _normalize_text(part)]


def _format_subqueries(subqueries: list[dict[str, str]]) -> str:
    if not subqueries:
        return "No subqueries were generated."

    parts: list[str] = []
    for index, item in enumerate(subqueries, start=1):
        parts.append(
            f"[{index}] subquestion={item['subquestion']}\n"
            f"search_query={item['search_query']}"
        )
    return "\n\n".join(parts)


def _normalize_text(value: object) -> str:
    return " ".join(str(value).split()).strip()
