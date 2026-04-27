from __future__ import annotations

from agent_for_mc.infrastructure.clients import DeepSeekChatClient
from agent_for_mc.infrastructure.observability import record_counter, trace_operation


def normalize_search_query(search_query: str) -> str:
    return " ".join(str(search_query).split()).strip()


def summarize_plugin_configs(
    search_query: str,
    *,
    client: DeepSeekChatClient,
    summary_max_chars: int,
    semantic_context: str = "",
) -> str:
    with trace_operation(
        "plugin_config.summarize",
        attributes={
            "component": "plugin_config",
            "semantic_context.length": len(semantic_context.strip()),
        },
        metric_name="rag_plugin_config_summarize_seconds",
    ):
        record_counter("rag_plugin_config_summarize_requests_total")
        if not semantic_context.strip():
            return "No matching uploaded server configuration semantic memories were found."

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a Minecraft server configuration assistant. Summarize stable "
                    "conclusions from the user question and uploaded server configuration "
                    "semantic memory. Focus on defaults, overrides, dependencies, file paths, "
                    "and clear conflicts. Do not invent missing values."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"User question:\n{normalize_search_query(search_query)}\n\n"
                    f"Uploaded server configuration semantic memory:\n{semantic_context.strip()}\n\n"
                    f"Keep the summary within {max(summary_max_chars, 1)} characters."
                ),
            },
        ]
        summary = client.chat(messages, temperature=0.1).strip()
        if not summary:
            return "No useful summary could be generated from the uploaded server configuration semantic memory."
        if summary_max_chars > 0 and len(summary) > summary_max_chars:
            summary = summary[:summary_max_chars].rstrip()
        return summary
