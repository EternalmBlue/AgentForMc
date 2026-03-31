from __future__ import annotations

from deepagents import create_deep_agent

from agent_for_mc.infrastructure.config import Settings
from agent_for_mc.infrastructure.observability import configure_observability, trace_operation
from agent_for_mc.interfaces.deepagent.factory import build_chat_model
from agent_for_mc.interfaces.deepagent.prompts import MEMORY_MAINTENANCE_SYSTEM_PROMPT


def build_memory_maintenance_agent(*, settings: Settings) -> object | None:
    configure_observability()
    if not settings.deepseek_api_key:
        return None

    with trace_operation(
        "build_memory_maintenance_agent",
        attributes={"component": "memory_maintenance_agent"},
    ):
        model = build_chat_model(
            settings=settings,
            model_name=settings.memory_maintenance_agent_model,
        )
        return create_deep_agent(
            model=model,
            system_prompt=MEMORY_MAINTENANCE_SYSTEM_PROMPT,
        )
