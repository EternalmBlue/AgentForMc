from __future__ import annotations

from langchain_core.messages import BaseMessage

from agent_for_mc.application.prompts import REWRITE_SYSTEM_PROMPT, format_history
from agent_for_mc.application.retrieval import normalize_question
from agent_for_mc.infrastructure.clients import DeepSeekChatClient


class QuestionRewriter:
    def __init__(self, client: DeepSeekChatClient):
        self._client = client

    def rewrite_question(self, history: list[BaseMessage], question: str) -> str:
        normalized_question = normalize_question(question)
        if not history:
            return normalized_question

        messages = [
            {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"最近对话历史：\n{format_history(history)}\n\n"
                    f"当前问题：\n{normalized_question}"
                ),
            },
        ]
        rewritten_question = self._client.chat(messages, temperature=0.0)
        return normalize_question(rewritten_question) or normalized_question
