from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass

from agent_for_mc.application.plugin_config.formatter import format_plugin_config_docs
from agent_for_mc.application.plugin_config.retriever import PluginConfigRetriever
from agent_for_mc.application.plugin_config.summarizer import summarize_plugin_configs
from agent_for_mc.application.semantic_memory import (
    SemanticMemoryRetriever,
    format_semantic_memory_docs,
)
from agent_for_mc.domain.models import PluginConfigDoc
from agent_for_mc.infrastructure.clients import DeepSeekChatClient, JinaEmbeddingClient
from agent_for_mc.infrastructure.observability import record_counter, trace_operation
from agent_for_mc.infrastructure.plugin_config_vector_store import (
    LancePluginConfigVectorStore,
)
from agent_for_mc.infrastructure.ranker import BceRanker


@dataclass(slots=True)
class PluginConfigToolContext:
    retriever: PluginConfigRetriever
    summarizer_client: DeepSeekChatClient
    top_k: int
    preview_chars: int
    summary_max_chars: int
    semantic_retriever: SemanticMemoryRetriever | None = None
    semantic_top_k: int = 0


_TOOL_CONTEXT: ContextVar[PluginConfigToolContext | None] = ContextVar(
    "plugin_config_tool_context",
    default=None,
)


def configure_plugin_config_tool(context: PluginConfigToolContext) -> None:
    _TOOL_CONTEXT.set(context)


def get_plugin_config_tool_context() -> PluginConfigToolContext:
    context = _TOOL_CONTEXT.get()
    if context is None:
        raise RuntimeError("plugin config tool context has not been configured")
    return context


def build_plugin_config_payload(
    search_query: str,
    *,
    context: PluginConfigToolContext,
) -> tuple[list[PluginConfigDoc], str]:
    with trace_operation(
        "plugin_config.payload",
        attributes={"component": "plugin_config", "query.length": len(search_query.strip())},
        metric_name="rag_plugin_config_payload_seconds",
    ):
        record_counter("rag_plugin_config_payload_requests_total")
        semantic_docs = []
        if context.semantic_retriever is not None:
            semantic_docs = context.semantic_retriever.retrieve(
                search_query,
                top_k=context.semantic_top_k,
            )
        docs = context.retriever.retrieve(search_query, top_k=context.top_k)
        semantic_context = format_semantic_memory_docs(
            semantic_docs,
            preview_chars=context.preview_chars,
        )
        summary = summarize_plugin_configs(
            search_query,
            docs,
            client=context.summarizer_client,
            summary_max_chars=context.summary_max_chars,
            semantic_context=semantic_context,
        )
        evidence = format_plugin_config_docs(docs, preview_chars=context.preview_chars)
        if semantic_context:
            evidence = f"Semantic memory:\n{semantic_context}\n\nConfig evidence:\n{evidence}"
        return docs, (
            f"Query: {search_query.strip() or 'No query provided.'}\n"
            f"Summary:\n{summary}\n\n"
            f"Evidence:\n{evidence}"
        )
