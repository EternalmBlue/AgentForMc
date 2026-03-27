from __future__ import annotations

import json

from langchain_core.messages import BaseMessage

from agent_for_mc.application.prompts import format_history
from agent_for_mc.infrastructure.clients import DeepSeekChatClient


PLUGIN_DECISION_SYSTEM_PROMPT = """你是一个路由决策器。
你的任务是判断当前问题是否需要调用 get_server_plugins_list。
判断规则：
1. 如果用户在询问当前服务器装了哪些插件、可用插件列表、插件清单、服务器插件名称，返回 true。
2. 如果用户的问题可以仅靠已有检索证据回答，不需要额外知道服务器插件列表，返回 false。
3. 如果问题明显是在问“服务器上有哪些插件/已安装插件/插件列表”，优先返回 true。
4. 只输出严格 JSON，不要输出解释。
5. JSON 格式必须是 {"need_plugins": true/false}。"""


class PluginDecisionMaker:
    def __init__(self, client: DeepSeekChatClient):
        self._client = client

    def decide_need_plugins(
        self,
        history: list[BaseMessage],
        question: str,
        rewritten_question: str,
        retrieval_summary: str,
    ) -> bool:
        messages = [
            {"role": "system", "content": PLUGIN_DECISION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"最近对话历史：\n{format_history(history)}\n\n"
                    f"用户原始问题：\n{question}\n\n"
                    f"检索问题：\n{rewritten_question}\n\n"
                    f"检索摘要：\n{retrieval_summary}\n"
                ),
            },
        ]
        content = self._client.chat(messages, temperature=0.0)
        return _parse_need_plugins(content)


def _parse_need_plugins(content: str) -> bool:
    normalized = content.strip()
    try:
        data = json.loads(normalized)
    except json.JSONDecodeError:
        lowered = normalized.lower()
        return lowered in {"true", "yes", "1", "need_plugins=true"}

    value = data.get("need_plugins")
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1"}
    return bool(value)
