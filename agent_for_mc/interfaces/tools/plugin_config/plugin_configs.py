from __future__ import annotations

from langchain_core.tools import tool

from agent_for_mc.application.plugin_config import (
    PluginConfigToolContext,
    build_plugin_config_payload,
    configure_plugin_config_tool,
    get_plugin_config_tool_context,
)
from agent_for_mc.infrastructure.observability import trace_operation


@tool("retrieve_plugin_configs")
def retrieve_plugin_configs(
    search_query: str | None = None,
    query: str | None = None,
) -> str:
    """Retrieve plugin configuration files and summarize the relevant settings."""
    effective_search_query = search_query or query or ""
    with trace_operation(
        "tool.retrieve_plugin_configs",
        attributes={"component": "tool", "query.length": len(effective_search_query.strip())},
        metric_name="rag_tool_retrieve_plugin_configs_seconds",
    ):
        _, formatted = build_plugin_config_payload(
            effective_search_query,
            context=get_plugin_config_tool_context(),
        )
        return formatted
