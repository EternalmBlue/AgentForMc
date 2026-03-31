from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass

from langchain_core.tools import tool

from agent_for_mc.application.deepagent_state import record_standalone_query
from agent_for_mc.infrastructure.clients import DeepSeekChatClient


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


_TOOL_CONTEXT: ContextVar[QueryRewriteToolContext | None] = ContextVar(
    "query_rewrite_tool_context",
    default=None,
)


def configure_query_rewrite_tool(context: QueryRewriteToolContext) -> None:
    _TOOL_CONTEXT.set(context)


def get_query_rewrite_tool_context() -> QueryRewriteToolContext:
    context = _TOOL_CONTEXT.get()
    if context is None:
        raise RuntimeError("query_rewrite tool context has not been configured")
    return context


@tool("query_rewrite")
def query_rewrite(question: str, history: str = "") -> str:
    """Rewrite a question into a standalone search query."""
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
