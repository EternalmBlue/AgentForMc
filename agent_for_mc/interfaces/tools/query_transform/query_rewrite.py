from __future__ import annotations

from dataclasses import dataclass

from langchain_core.tools import tool

from agent_for_mc.application.deepagent_state import record_standalone_query
from agent_for_mc.infrastructure.clients import DeepSeekChatClient
from agent_for_mc.infrastructure.observability import record_counter, trace_operation
from agent_for_mc.infrastructure.shared_context import SharedContextSlot


QUERY_REWRITE_SYSTEM_PROMPT = """
You are a query rewriting tool for a Minecraft plugin assistant.

Rewrite the user question into a standalone search query.

Rules:
- Output only the rewritten query.
- Keep it concise.
- Preserve plugin names, aliases, versions, and class names.
- If the question uses pronouns like "it" or "that plugin", resolve them from the provided history.
- Do not invent plugin names or details that were not present.
""".strip()


@dataclass(slots=True)
class QueryRewriteToolContext:
    client: DeepSeekChatClient


_TOOL_CONTEXT = SharedContextSlot[QueryRewriteToolContext](
    "query_rewrite_tool_context"
)


def configure_query_rewrite_tool(context: QueryRewriteToolContext) -> None:
    _TOOL_CONTEXT.set(context)


def get_query_rewrite_tool_context() -> QueryRewriteToolContext:
    return _TOOL_CONTEXT.get(
        error_message="query_rewrite tool context has not been configured"
    )


@tool("query_rewrite")
def query_rewrite(question: str, history: str = "") -> str:
    """Rewrite a question into a standalone search query."""
    with trace_operation(
        "tool.query_rewrite",
        attributes={"component": "tool", "question.length": len(question.strip())},
        metric_name="rag_tool_query_rewrite_seconds",
    ):
        record_counter("rag_tool_query_rewrite_requests_total")
        context = get_query_rewrite_tool_context()
        messages = [
            {"role": "system", "content": QUERY_REWRITE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Recent conversation:\n{history.strip() or 'No history.'}\n\n"
                    f"User question:\n{question.strip()}\n"
                ),
            },
        ]
        content = context.client.chat(messages, temperature=0.0)
        standalone_query = _normalize_text(content) or _normalize_text(question)
        record_standalone_query(standalone_query)
        return standalone_query


def _normalize_text(value: str) -> str:
    return " ".join(str(value).split()).strip()
