from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field

from agent_for_mc.domain.models import RetrievedDoc


@dataclass(slots=True)
class DeepAgentTurnContext:
    rewritten_question: str = ""
    retrieved_docs: list[RetrievedDoc] = field(default_factory=list)
    server_plugins: list[str] = field(default_factory=list)


_TURN_CONTEXT: ContextVar[DeepAgentTurnContext | None] = ContextVar(
    "deepagent_turn_context",
    default=None,
)


def start_turn_context() -> None:
    _TURN_CONTEXT.set(DeepAgentTurnContext())


def consume_turn_context() -> DeepAgentTurnContext | None:
    context = _TURN_CONTEXT.get()
    _TURN_CONTEXT.set(None)
    return context


def clear_turn_context() -> None:
    _TURN_CONTEXT.set(None)


def record_rewritten_question(question: str) -> None:
    context = _TURN_CONTEXT.get()
    if context is None:
        return
    context.rewritten_question = question


def record_retrieved_docs(docs: list[RetrievedDoc]) -> None:
    context = _TURN_CONTEXT.get()
    if context is None or not docs:
        return

    seen_ids = {doc.id for doc in context.retrieved_docs}
    for doc in docs:
        if doc.id in seen_ids:
            continue
        seen_ids.add(doc.id)
        context.retrieved_docs.append(doc)


def record_server_plugins(plugins: list[str]) -> None:
    context = _TURN_CONTEXT.get()
    if context is None or not plugins:
        return

    seen_plugins = set(context.server_plugins)
    for plugin in plugins:
        if plugin in seen_plugins:
            continue
        seen_plugins.add(plugin)
        context.server_plugins.append(plugin)
