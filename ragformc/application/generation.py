from __future__ import annotations

from langchain_core.messages import BaseMessage

from ragformc.application.prompts import (
    ANSWER_SYSTEM_PROMPT,
    format_docs_for_prompt,
    format_history,
)
from ragformc.domain.models import AnswerResult, RetrievedDoc
from ragformc.infrastructure.clients import DeepSeekChatClient


class AnswerGenerator:
    def __init__(self, client: DeepSeekChatClient):
        self._client = client

    def answer(
        self,
        history: list[BaseMessage],
        question: str,
        rewritten_question: str,
        docs: list[RetrievedDoc],
        server_plugins: list[str] | None = None,
    ) -> AnswerResult:
        plugin_section = _format_server_plugins(server_plugins or [])
        messages = [
            {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"最近对话历史：\n{format_history(history)}\n\n"
                    f"用户原始问题：\n{question}\n\n"
                    f"检索问题：\n{rewritten_question}\n\n"
                    f"服务器插件列表：\n{plugin_section}\n\n"
                    f"检索证据：\n{format_docs_for_prompt(docs)}\n\n"
                    "请用简洁中文回答用户问题。"
                ),
            },
        ]
        answer_text = self._client.chat(messages, temperature=0.1)
        return AnswerResult(
            answer=answer_text,
            citations=docs,
            rewritten_question=rewritten_question,
        )


def _format_server_plugins(server_plugins: list[str]) -> str:
    if not server_plugins:
        return "无服务器插件列表。"

    return "\n".join(
        f"{index}. {plugin}" for index, plugin in enumerate(server_plugins, start=1)
    )
