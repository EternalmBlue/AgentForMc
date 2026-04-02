from __future__ import annotations

from collections import deque
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from agent_for_mc.application.deepagent_state import (
    clear_turn_context,
    consume_turn_context,
    record_standalone_query,
    start_turn_context,
)
from agent_for_mc.application.plugin_semantic_agent import PluginSemanticAgentService
from agent_for_mc.application.memory_service import MemoryService, format_memory_context
from agent_for_mc.application.prompts import format_history
from agent_for_mc.domain.errors import ServiceError
from agent_for_mc.domain.models import AnswerResult
from agent_for_mc.infrastructure.config import Settings
from agent_for_mc.infrastructure.observability import record_counter, trace_operation
from agent_for_mc.infrastructure.vector_store import LancePluginVectorStore


class RagChatSession:
    def __init__(
        self,
        settings: Settings,
        vector_store: LancePluginVectorStore,
        deep_agent: Any,
        memory_service: MemoryService | None = None,
        plugin_semantic_service: PluginSemanticAgentService | None = None,
    ):
        self._settings = settings
        self._vector_store = vector_store
        self._deep_agent = deep_agent
        self._memory_service = memory_service
        self._plugin_semantic_service = plugin_semantic_service
        self._history: deque[BaseMessage] = deque(
            maxlen=settings.rewrite_history_turns * 2
        )

    def startup_validate(self):
        return self._vector_store.validate()

    def clear_history(self) -> None:
        self._history.clear()

    def close(self) -> None:
        if self._memory_service is not None:
            self._memory_service.close()
        if self._plugin_semantic_service is not None:
            self._plugin_semantic_service.close()

    def has_plugin_semantic_service(self) -> bool:
        return self._plugin_semantic_service is not None

    def start_plugin_semantic_refresh(self) -> bool:
        if self._plugin_semantic_service is None:
            return False
        self._plugin_semantic_service.refresh()
        return True

    def ask(self, question: str) -> AnswerResult:
        with trace_operation(
            "session.ask",
            attributes={"component": "session", "question.length": len(question.strip())},
            metric_name="rag_session_ask_seconds",
        ):
            record_counter("rag_session_ask_requests_total")
            history = list(self._history)
            standalone_query = question.strip()
            memory_messages: list[BaseMessage] = []

            if self._memory_service is not None:
                try:
                    memory_records = self._memory_service.recall(
                        question,
                        history_text=format_history(history),
                    )
                    memory_context = format_memory_context(memory_records)
                    if memory_context:
                        memory_messages.append(
                            SystemMessage(
                                content=(
                                    "长期记忆（仅作为参考，不要把它当作检索证据）：\n"
                                    f"{memory_context}"
                                )
                            )
                        )
                except Exception:
                    memory_messages = []

            start_turn_context()
            record_standalone_query(standalone_query)
            try:
                state = self._deep_agent.invoke(
                    {
                        "messages": [
                            *memory_messages,
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
                    standalone_query=(
                        turn.standalone_query if turn and turn.standalone_query else standalone_query
                    ),
                )
                if self._memory_service is not None:
                    try:
                        self._memory_service.observe_turn(question, result.answer)
                    except Exception:
                        pass
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
