from __future__ import annotations

from dataclasses import dataclass

from agent_for_mc.application.deepagent_state import record_progress
from agent_for_mc.application.plugin_config.summarizer import summarize_plugin_configs
from agent_for_mc.application.semantic_memory import (
    SemanticMemoryRetriever,
    format_semantic_memory_docs,
)
from agent_for_mc.domain.models import SemanticMemoryDoc
from agent_for_mc.infrastructure.clients import DeepSeekChatClient
from agent_for_mc.infrastructure.observability import record_counter, trace_operation
from agent_for_mc.infrastructure.shared_context import SharedContextSlot


@dataclass(slots=True)
class PluginConfigToolContext:
    retriever: SemanticMemoryRetriever
    summarizer_client: DeepSeekChatClient
    top_k: int
    preview_chars: int
    summary_max_chars: int = 500


_TOOL_CONTEXT = SharedContextSlot[PluginConfigToolContext]("plugin_config_tool_context")


def configure_plugin_config_tool(context: PluginConfigToolContext) -> None:
    _TOOL_CONTEXT.set(context)


def get_plugin_config_tool_context() -> PluginConfigToolContext:
    return _TOOL_CONTEXT.get(
        error_message="plugin config tool context has not been configured"
    )


def build_plugin_config_payload(
    search_query: str,
    *,
    context: PluginConfigToolContext,
) -> tuple[list[SemanticMemoryDoc], str]:
    with trace_operation(
        "plugin_config.payload",
        attributes={"component": "plugin_config", "query.length": len(search_query.strip())},
        metric_name="rag_plugin_config_payload_seconds",
    ):
        record_progress("server_config_retrieval", "Searching uploaded server configuration memory.")
        record_counter("rag_plugin_config_payload_requests_total")
        docs = context.retriever.retrieve(search_query, top_k=context.top_k)
        semantic_context = (
            format_semantic_memory_docs(docs, preview_chars=context.preview_chars)
            if docs
            else ""
        )
        summary = summarize_plugin_configs(
            search_query,
            client=context.summarizer_client,
            summary_max_chars=context.summary_max_chars,
            semantic_context=semantic_context,
        )
        record_progress("server_config_ready", "Uploaded server configuration memory is ready.")
        evidence = semantic_context or "No matching uploaded server configuration semantic memories were found."
        return docs, (
            f"Query: {search_query.strip() or 'No query provided.'}\n"
            f"Summary:\n{summary}\n\n"
            f"Server config semantic memory:\n{evidence}"
        )
