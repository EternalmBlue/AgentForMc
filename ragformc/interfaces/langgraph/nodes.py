from __future__ import annotations

from typing import Any, cast

from langchain_core.messages import AIMessage

from ragformc.application.generation import AnswerGenerator
from ragformc.application.multi_query import MultiQueryPlanner
from ragformc.application.plugin_decision import PluginDecisionMaker
from ragformc.application.retrieval import Retriever
from ragformc.application.retrieval_tool import (
    RetrieveDocsToolContext,
    build_multi_query_retrieve_docs_payload,
    build_retrieve_docs_payload,
)
from ragformc.application.rewrite import QuestionRewriter
from ragformc.application.state import RagGraphState
from ragformc.interfaces.tools.server_plugins import get_server_plugins_list


def _merge_state(state: RagGraphState, **updates: Any) -> RagGraphState:
    merged = dict(state)
    merged.update(updates)
    return cast(RagGraphState, merged)


def build_rewrite_node(rewriter: QuestionRewriter):
    def _rewrite(state: RagGraphState) -> RagGraphState:
        history = list(state.get("messages", []))
        question = str(state["origin_question"])
        rewritten_question = rewriter.rewrite_question(history, question)
        return _merge_state(state, rewritten_question=rewritten_question)

    return _rewrite


def build_retrieve_node(
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

    return _retrieve


def build_decide_multi_query_node(multi_query_planner: MultiQueryPlanner):
    def _decide_multi_query(state: RagGraphState) -> RagGraphState:
        history = list(state.get("messages", []))
        question = str(state["origin_question"])
        rewritten_question = str(state.get("rewritten_question") or question)
        retrieval_summary = str(state.get("retrieval_summary", ""))
        need_multi_query, queries = multi_query_planner.decide_and_generate_queries(
            history=history,
            question=question,
            rewritten_question=rewritten_question,
            retrieval_summary=retrieval_summary,
        )
        return _merge_state(
            state,
            need_multi_query=need_multi_query,
            multi_query_variants=queries,
        )

    return _decide_multi_query


def build_multi_query_retrieve_node(
    retriever: Retriever,
    *,
    top_k: int,
    citation_preview_chars: int,
):
    def _multi_query_retrieve(state: RagGraphState) -> RagGraphState:
        existing_docs = list(state.get("retrieved_docs", []))
        queries = list(state.get("multi_query_variants") or [])
        if not queries:
            query = str(state.get("rewritten_question") or state["origin_question"])
            queries = [query]

        multi_docs, multi_summary = build_multi_query_retrieve_docs_payload(
            queries,
            context=RetrieveDocsToolContext(
                retriever=retriever,
                top_k=top_k,
                citation_preview_chars=citation_preview_chars,
            ),
        )
        merged_docs = list(existing_docs)
        seen_ids = {doc.id for doc in merged_docs}
        for doc in multi_docs:
            if doc.id in seen_ids:
                continue
            seen_ids.add(doc.id)
            merged_docs.append(doc)

        retrieval_summary = str(state.get("retrieval_summary", ""))
        if retrieval_summary:
            retrieval_summary = f"{retrieval_summary}\n\n{multi_summary}"
        else:
            retrieval_summary = multi_summary

        return _merge_state(
            state,
            retrieved_docs=merged_docs,
            retrieval_summary=retrieval_summary,
        )

    return _multi_query_retrieve


def build_decide_plugins_node(plugin_decider: PluginDecisionMaker):
    def _decide_plugins(state: RagGraphState) -> RagGraphState:
        history = list(state.get("messages", []))
        question = str(state["origin_question"])
        rewritten_question = str(state.get("rewritten_question") or question)
        retrieval_summary = str(state.get("retrieval_summary", ""))
        need_plugins = plugin_decider.decide_need_plugins(
            history=history,
            question=question,
            rewritten_question=rewritten_question,
            retrieval_summary=retrieval_summary,
        )
        return _merge_state(state, need_plugins=need_plugins)

    return _decide_plugins


def build_get_server_plugins_list_node():
    def _get_server_plugins_list(state: RagGraphState) -> RagGraphState:
        plugins = list(get_server_plugins_list.invoke({}))
        return _merge_state(state, server_plugins=plugins)

    return _get_server_plugins_list


def build_answer_node(
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
        server_plugins = list(state.get("server_plugins") or [])
        result = answer_generator.answer(
            history,
            question,
            rewritten_question,
            selected_docs,
            server_plugins=server_plugins or None,
        )
        return _merge_state(
            state,
            answer=result.answer,
            citations=result.citations,
            rewritten_question=result.rewritten_question,
            messages=[AIMessage(content=result.answer)],
        )

    return _answer


def route_after_plugin_decision(state: RagGraphState) -> str:
    return "get_server_plugins_list" if state.get("need_plugins") else "answer"


def route_after_multi_query_decision(state: RagGraphState) -> str:
    return "multi_query_retrieve" if state.get("need_multi_query") else "decide_plugins"
