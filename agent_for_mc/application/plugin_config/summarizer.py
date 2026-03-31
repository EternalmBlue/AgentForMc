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
) -> str:
    with trace_operation(
        "plugin_config.summarize",
        attributes={"component": "plugin_config", "doc.count": len(docs)},
        metric_name="rag_plugin_config_summarize_seconds",
    ):
        record_counter("rag_plugin_config_summarize_requests_total")
        if not docs:
            return "未命中相关插件配置文件。"

        messages = [
            {
                "role": "system",
                "content": (
                    "你是插件配置整理助手。请根据用户问题和命中的配置片段，"
                    "用简洁中文总结与配置相关的结论。"
                    "只讨论配置项、默认值、开关、依赖关系、路径关系和明显冲突。"
                    "不要编造未出现在证据中的内容；如果证据不足，请直接说明。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"用户问题:\n{normalize_search_query(search_query)}\n\n"
                    f"命中配置片段:\n{json.dumps(serialize_docs(docs), ensure_ascii=False, indent=2)}\n\n"
                    f"请将摘要控制在 {max(summary_max_chars, 1)} 字以内。"
                ),
            },
        ]
        summary = client.chat(messages, temperature=0.1).strip()
        if not summary:
            return "未能从配置片段中生成有效摘要。"
        if summary_max_chars > 0 and len(summary) > summary_max_chars:
            summary = summary[:summary_max_chars].rstrip()
        return summary
