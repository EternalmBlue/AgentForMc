from __future__ import annotations

from collections import deque
from uuid import uuid4

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from agent_for_mc.application.generation import AnswerGenerator
from agent_for_mc.application.retrieval import Retriever
from agent_for_mc.application.rewrite import QuestionRewriter
from agent_for_mc.domain.models import AnswerResult
from agent_for_mc.infrastructure.config import Settings
from agent_for_mc.infrastructure.vector_store import LancePluginVectorStore


class RagChatSession:
    def __init__(
        self,
        settings: Settings,
        vector_store: LancePluginVectorStore,
        rewriter: QuestionRewriter,
        retriever: Retriever,
        answer_generator: AnswerGenerator,
        rag_chain=None,
        graph_app=None,
    ):
        self._settings = settings
        self._vector_store = vector_store
        self._rewriter = rewriter
        self._retriever = retriever
        self._answer_generator = answer_generator
        self._rag_chain = rag_chain
        self._graph_app = graph_app
        self._history: deque[BaseMessage] = deque(
            maxlen=settings.rewrite_history_turns * 2
        )

    def startup_validate(self):
        return self._vector_store.validate()

    def clear_history(self) -> None:
        self._history.clear()

    def rewrite_question(self, history: list[BaseMessage], question: str) -> str:
        return self._rewriter.rewrite_question(history, question)

    def retrieve(self, question: str, top_k: int = 8):
        return self._retriever.retrieve(question, top_k=top_k)

    def answer(
        self,
        history: list[BaseMessage],
        question: str,
        rewritten_question: str,
        docs,
    ) -> AnswerResult:
        return self._answer_generator.answer(history, question, rewritten_question, docs)

    def ask(self, question: str) -> AnswerResult:
        history = list(self._history)
        if self._graph_app is not None:
            state = self._graph_app.invoke(
                {
                    "uuid": uuid4().hex,
                    "messages": history,
                    "origin_question": question,
                    "retry_count": 0,
                }
            )
            result = AnswerResult(
                answer=str(state.get("answer", "")),
                citations=list(state.get("citations", [])),
                rewritten_question=str(state.get("rewritten_question", "")),
            )
        elif self._rag_chain is not None:
            state = self._rag_chain.invoke(
                {
                    "messages": history,
                    "origin_question": question,
                }
            )
            result = AnswerResult(
                answer=str(state.get("answer", "")),
                citations=list(state.get("citations", [])),
                rewritten_question=str(state.get("rewritten_question", "")),
            )
        else:
            rewritten_question = self.rewrite_question(history, question)
            docs = self.retrieve(rewritten_question, top_k=self._settings.retrieval_top_k)
            selected_docs = docs[: self._settings.answer_top_k]
            result = self.answer(history, question, rewritten_question, selected_docs)

        self._history.append(HumanMessage(content=question))
        self._history.append(AIMessage(content=result.answer))
        return result
