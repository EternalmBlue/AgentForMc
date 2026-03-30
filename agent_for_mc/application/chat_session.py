from __future__ import annotations

from collections import deque
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from agent_for_mc.application.deepagent_state import (
    clear_turn_context,
    consume_turn_context,
    record_rewritten_question,
    start_turn_context,
)
from agent_for_mc.application.retrieval import normalize_question
from agent_for_mc.domain.errors import ServiceError
from agent_for_mc.domain.models import AnswerResult
from agent_for_mc.infrastructure.config import Settings
from agent_for_mc.infrastructure.vector_store import LancePluginVectorStore


class RagChatSession:
    def __init__(
        self,
        settings: Settings,
        vector_store: LancePluginVectorStore,
        deep_agent: Any,
    ):
        self._settings = settings
        self._vector_store = vector_store
        self._deep_agent = deep_agent
        self._history: deque[BaseMessage] = deque(
            maxlen=settings.rewrite_history_turns * 2
        )

    def startup_validate(self):
        return self._vector_store.validate()

    def clear_history(self) -> None:
        self._history.clear()

    def ask(self, question: str) -> AnswerResult:
        history = list(self._history)
        rewritten_question = normalize_question(question)

        start_turn_context()
        record_rewritten_question(rewritten_question)
        try:
            state = self._deep_agent.invoke(
                {
                    "messages": [
                        *history,
                        HumanMessage(content=question),
                    ]
                }
            )
            answer_text = _extract_agent_answer(state)
            if not answer_text:
                raise ServiceError("DeepAgent 返回了空回答。")

            turn = consume_turn_context()
            citations = list(turn.retrieved_docs if turn else [])
            result = AnswerResult(
                answer=answer_text,
                citations=citations,
                rewritten_question=(
                    turn.rewritten_question if turn and turn.rewritten_question else rewritten_question
                ),
            )
        except Exception as exc:
            raise ServiceError(f"DeepAgent 调用失败: {exc}") from exc
        finally:
            clear_turn_context()

        self._history.append(HumanMessage(content=question))
        self._history.append(AIMessage(content=result.answer))
        return result


def _extract_agent_answer(state: object) -> str:
    if isinstance(state, str):
        return state.strip()

    if isinstance(state, dict):
        messages = state.get("messages")
        if isinstance(messages, list):
            for message in reversed(messages):
                if isinstance(message, AIMessage):
                    content = str(message.content).strip()
                    if content:
                        return content
                content = getattr(message, "content", None)
                if isinstance(content, str) and content.strip():
                    return content.strip()

        output = state.get("output")
        if isinstance(output, str):
            return output.strip()

    return ""
