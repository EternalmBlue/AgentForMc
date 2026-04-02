from __future__ import annotations

import json

from agent_for_mc.application.plugin_config.formatter import serialize_docs
from agent_for_mc.application.plugin_config.retriever import normalize_search_query
from agent_for_mc.domain.models import PluginConfigDoc
from agent_for_mc.infrastructure.clients import DeepSeekChatClient
from agent_for_mc.infrastructure.observability import record_counter, trace_operation


def summarize_plugin_configs(
    search_query: str,
    docs: list[PluginConfigDoc],
    *,
    client: DeepSeekChatClient,
    summary_max_chars: int,
    semantic_context: str = "",
) -> str:
    with trace_operation(
        "plugin_config.summarize",
        attributes={"component": "plugin_config", "doc.count": len(docs)},
        metric_name="rag_plugin_config_summarize_seconds",
    ):
        record_counter("rag_plugin_config_summarize_requests_total")
        if not docs and not semantic_context.strip():
            return "No matching plugin configuration files were found."

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a plugin configuration assistant. Summarize stable conclusions "
                    "from the user question, semantic memory context, and matching config "
                    "file excerpts. Focus on defaults, overrides, dependencies, file paths, "
                    "and clear conflicts. Do not invent missing values."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"User question:\n{normalize_search_query(search_query)}\n\n"
                    f"Semantic memory:\n{semantic_context.strip() or 'No semantic memory context.'}\n\n"
                    f"Matched config excerpts:\n{json.dumps(serialize_docs(docs), ensure_ascii=False, indent=2)}\n\n"
                    f"Keep the summary within {max(summary_max_chars, 1)} characters."
                ),
            },
        ]
        summary = client.chat(messages, temperature=0.1).strip()
        if not summary:
            return "No useful summary could be generated from the matched config snippets."
        if summary_max_chars > 0 and len(summary) > summary_max_chars:
            summary = summary[:summary_max_chars].rstrip()
        return summary
