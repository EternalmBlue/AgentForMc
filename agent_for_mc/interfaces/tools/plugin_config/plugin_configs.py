from __future__ import annotations

from langchain_core.tools import tool

from agent_for_mc.application.plugin_config import (
    PluginConfigToolContext,
    build_plugin_config_payload,
    configure_plugin_config_tool,
    get_plugin_config_tool_context,
)


@tool("retrieve_plugin_configs")
def retrieve_plugin_configs(
    search_query: str | None = None,
    query: str | None = None,
) -> str:
    """Retrieve plugin configuration files and summarize the relevant settings."""
    effective_search_query = search_query or query or ""
    _, formatted = build_plugin_config_payload(
        effective_search_query,
        context=get_plugin_config_tool_context(),
    )
    return formatted
