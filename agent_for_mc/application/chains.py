from __future__ import annotations

from typing import Any, cast

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from agent_for_mc.application.generation import AnswerGenerator
from agent_for_mc.application.retrieval import Retriever
from agent_for_mc.application.retrieval_tool import (
    RetrieveDocsToolContext,
    build_retrieve_docs_payload,
)
from agent_for_mc.application.rewrite import QuestionRewriter
from agent_for_mc.application.state import RagGraphState


def _merge_state(state: RagGraphState, **updates: Any) -> RagGraphState:
    merged = dict(state)
    merged.update(updates)
    return cast(RagGraphState, merged)


def build_rewrite_runnable(rewriter: QuestionRewriter):
    def _rewrite(state: RagGraphState) -> RagGraphState:
        history = list(state.get("messages", []))
        question = str(state["origin_question"])
        rewritten_question = rewriter.rewrite_question(history, question)
        return _merge_state(state, rewritten_question=rewritten_question)

    return RunnableLambda(_rewrite)


def build_retrieve_runnable(
    retriever: Retriever,
    *,
    top_k: int,
    citation_preview_chars: int,
):
    def _retrieve(state: RagGraphState) -> RagGraphState:
        query = str(state.get("rewritten_question") or state["origin_question"])
        docs, summary = build_retrieve_docs_payload(
            query,
            context=RetrieveDocsToolContext(
                retriever=retriever,
                top_k=top_k,
                citation_preview_chars=citation_preview_chars,
            ),
        )
        return _merge_state(
            state,
            retrieved_docs=docs,
            retrieval_summary=summary,
        )

    return RunnableLambda(_retrieve)


def build_answer_runnable(
    answer_generator: AnswerGenerator,
    *,
    answer_top_k: int,
):
    def _answer(state: RagGraphState) -> RagGraphState:
        history = list(state.get("messages", []))
        question = str(state["origin_question"])
        rewritten_question = str(state.get("rewritten_question") or question)
        docs = list(state.get("retrieved_docs", []))
        selected_docs = docs[:answer_top_k]
        result = answer_generator.answer(
            history,
            question,
            rewritten_question,
            selected_docs,
        )
        return _merge_state(
            state,
            answer=result.answer,
            citations=result.citations,
            rewritten_question=result.rewritten_question,
            messages=[AIMessage(content=result.answer)],
        )

    return RunnableLambda(_answer)


def build_rag_chain(
    *,
    rewriter: QuestionRewriter,
    retriever: Retriever,
    answer_generator: AnswerGenerator,
    top_k: int,
    answer_top_k: int,
    citation_preview_chars: int,
):
    return (
        build_rewrite_runnable(rewriter)
        | build_retrieve_runnable(
            retriever,
            top_k=top_k,
            citation_preview_chars=citation_preview_chars,
        )
        | build_answer_runnable(
            answer_generator,
            answer_top_k=answer_top_k,
        )
    )
