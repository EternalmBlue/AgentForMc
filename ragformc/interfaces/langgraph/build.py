from __future__ import annotations

from typing import cast

from langgraph.graph import END, START, StateGraph

from ragformc.application.generation import AnswerGenerator
from ragformc.application.multi_query import MultiQueryPlanner
from ragformc.application.plugin_decision import PluginDecisionMaker
from ragformc.application.retrieval import Retriever
from ragformc.application.rewrite import QuestionRewriter
from ragformc.application.state import RagGraphState
from ragformc.interfaces.langgraph.nodes import (
    build_answer_node,
    build_decide_multi_query_node,
    build_decide_plugins_node,
    build_get_server_plugins_list_node,
    build_multi_query_retrieve_node,
    build_retrieve_node,
    build_rewrite_node,
    route_after_multi_query_decision,
    route_after_plugin_decision,
)
from ragformc.interfaces.tools.retrieval import (
    RetrieveDocsToolContext,
    configure_retrieve_docs_tool,
)


def build_app(
    *,
    rewriter: QuestionRewriter,
    retriever: Retriever,
    answer_generator: AnswerGenerator,
    multi_query_planner: MultiQueryPlanner,
    plugin_decider: PluginDecisionMaker,
    top_k: int,
    answer_top_k: int,
    citation_preview_chars: int,
):
    configure_retrieve_docs_tool(
        RetrieveDocsToolContext(
            retriever=retriever,
            top_k=top_k,
            citation_preview_chars=citation_preview_chars,
        )
    )

    builder: StateGraph[RagGraphState] = StateGraph(
        cast(type[RagGraphState], RagGraphState)
    )
    builder.add_node("rewrite", build_rewrite_node(rewriter))  # type: ignore[arg-type]
    builder.add_node(  # type: ignore[arg-type]
        "retrieve",
        build_retrieve_node(
            retriever,
            top_k=top_k,
            citation_preview_chars=citation_preview_chars,
        ),
    )
    builder.add_node(  # type: ignore[arg-type]
        "decide_multi_query",
        build_decide_multi_query_node(multi_query_planner),
    )
    builder.add_node(  # type: ignore[arg-type]
        "multi_query_retrieve",
        build_multi_query_retrieve_node(
            retriever,
            top_k=top_k,
            citation_preview_chars=citation_preview_chars,
        ),
    )
    builder.add_node(  # type: ignore[arg-type]
        "decide_plugins",
        build_decide_plugins_node(plugin_decider),
    )
    builder.add_node(  # type: ignore[arg-type]
        "get_server_plugins_list",
        build_get_server_plugins_list_node(),
    )
    builder.add_node(  # type: ignore[arg-type]
        "answer",
        build_answer_node(
            answer_generator,
            answer_top_k=answer_top_k,
        ),
    )
    builder.add_edge(START, "rewrite")
    builder.add_edge("rewrite", "retrieve")
    builder.add_edge("retrieve", "decide_multi_query")
    builder.add_conditional_edges(  # type: ignore[arg-type]
        "decide_multi_query",
        route_after_multi_query_decision,
        {
            "multi_query_retrieve": "multi_query_retrieve",
            "decide_plugins": "decide_plugins",
        },
    )
    builder.add_edge("multi_query_retrieve", "decide_plugins")
    builder.add_conditional_edges(  # type: ignore[arg-type]
        "decide_plugins",
        route_after_plugin_decision,
        {
            "get_server_plugins_list": "get_server_plugins_list",
            "answer": "answer",
        },
    )
    builder.add_edge("get_server_plugins_list", "answer")
    builder.add_edge("answer", END)
    return builder.compile()
