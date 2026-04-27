from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

from langchain_core.tools import tool

from agent_for_mc.infrastructure.observability import record_counter, trace_operation
from agent_for_mc.infrastructure.shared_context import SharedContextSlot

if TYPE_CHECKING:
    from agent_for_mc.application.plugin_semantic_agent import PluginSemanticAgentService


@dataclass(slots=True)
class PluginSemanticRefreshToolContext:
    service: PluginSemanticAgentService


_TOOL_CONTEXT = SharedContextSlot[PluginSemanticRefreshToolContext](
    "plugin_semantic_refresh_tool_context"
)


def configure_plugin_semantic_refresh_tool(
    context: PluginSemanticRefreshToolContext,
) -> None:
    _TOOL_CONTEXT.set(context)


def get_plugin_semantic_refresh_tool_context() -> PluginSemanticRefreshToolContext:
    return _TOOL_CONTEXT.get(
        error_message="plugin semantic refresh tool context has not been configured"
    )


@tool("refresh_plugin_semantic_memory")
def refresh_plugin_semantic_memory() -> str:
    """Start an incremental background refresh of semantic plugin memory from mc_servers."""
    with trace_operation(
        "tool.refresh_plugin_semantic_memory",
        attributes={"component": "tool"},
        metric_name="rag_tool_refresh_plugin_semantic_memory_seconds",
    ):
        record_counter("rag_tool_refresh_plugin_semantic_memory_requests_total")
        context = get_plugin_semantic_refresh_tool_context()
        status = context.service.request_refresh_status(full=False)
        return json.dumps(
            {
                "status": status,
                "mode": "incremental",
                "mc_servers_root": context.service.mc_servers_root,
                "message": _render_status_message(status),
            },
            ensure_ascii=False,
        )


def _render_status_message(status: str) -> str:
    if status == "started":
        return "Incremental plugin semantic refresh started in background."
    if status == "already_running":
        return "Plugin semantic refresh is already running."
    if status == "closed":
        return "Plugin semantic refresh service is closed."
    return f"Plugin semantic refresh returned status: {status}"
