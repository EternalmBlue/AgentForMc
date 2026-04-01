from __future__ import annotations

from deepagents.middleware.subagents import SubAgent

from agent_for_mc.infrastructure.config import Settings
from agent_for_mc.interfaces.deepagent.factory import build_chat_model
from agent_for_mc.interfaces.deepagent.prompts import PLUGIN_CONFIG_AGENT_SYSTEM_PROMPT
from agent_for_mc.interfaces.tools.plugin_config import retrieve_plugin_configs


def build_plugin_config_subagent(*, settings: Settings) -> SubAgent:
    plugin_config_model = build_chat_model(
        settings=settings,
        model_name=settings.plugin_config_agent_model,
    )
    return {
        "name": "plugin_config_agent",
        "description": (
            "Use this agent for plugin config files, defaults, file paths, dependency wiring, "
            "and config-file differences."
        ),
        "system_prompt": PLUGIN_CONFIG_AGENT_SYSTEM_PROMPT,
        "tools": [retrieve_plugin_configs],
        "model": plugin_config_model,
    }
