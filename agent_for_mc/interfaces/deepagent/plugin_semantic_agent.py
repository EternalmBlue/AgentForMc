from __future__ import annotations

from deepagents import create_deep_agent

from agent_for_mc.infrastructure.config import Settings
from agent_for_mc.infrastructure.observability import configure_observability, trace_operation
from agent_for_mc.interfaces.deepagent.factory import build_chat_model
from agent_for_mc.interfaces.deepagent.prompts import PLUGIN_SEMANTIC_AGENT_SYSTEM_PROMPT


def build_plugin_semantic_agent(*, settings: Settings) -> object | None:
    configure_observability()
    if not settings.deepseek_api_key:
        return None

    with trace_operation(
        "build_plugin_semantic_agent",
        attributes={"component": "plugin_semantic_agent"},
    ):
        model = build_chat_model(
            settings=settings,
            model_name=settings.plugin_semantic_agent_model,
        )
        return create_deep_agent(
            model=model,
            system_prompt=PLUGIN_SEMANTIC_AGENT_SYSTEM_PROMPT,
        )
