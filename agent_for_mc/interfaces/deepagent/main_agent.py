from __future__ import annotations

from deepagents import create_deep_agent

from agent_for_mc.application.retrieval import Retriever
from agent_for_mc.infrastructure.config import Settings
from agent_for_mc.infrastructure.observability import trace_operation
from agent_for_mc.infrastructure.ranker import BceRanker
from agent_for_mc.interfaces.deepagent.factory import (
    build_chat_model,
    configure_deepagent_dependencies,
)
from agent_for_mc.interfaces.deepagent.prompts import DEEPAGENT_SYSTEM_PROMPT
from agent_for_mc.interfaces.deepagent.subagents import build_plugin_config_subagent
from agent_for_mc.interfaces.tools.query_transform import (
    hyde_retrieve_docs,
    multi_query_rag,
    multi_query_retrieve_docs,
    query_expansion,
    query_rewrite,
    subquery_decomposition,
)
from agent_for_mc.interfaces.tools.retrieval import (
    get_server_plugins_list,
    judge_answer_quality,
    judge_retrieval_freshness,
    retrieve_docs,
    select_retrieval_tool,
)
from agent_for_mc.interfaces.tools.routing import (
    analyze_question,
    route_plugin_config_request,
)


def build_deep_agent(
    *,
    settings: Settings,
    retriever: Retriever,
    ranker: BceRanker | None = None,
) -> object | None:
    if not settings.deepseek_api_key:
        return None

    with trace_operation("build_deep_agent", attributes={"component": "deepagent"}):
        configure_deepagent_dependencies(
            settings=settings,
            retriever=retriever,
            ranker=ranker,
        )
        model = build_chat_model(
            settings=settings,
            model_name=settings.deepseek_model,
        )
        plugin_config_agent = build_plugin_config_subagent(settings=settings)
        return create_deep_agent(
            model=model,
            tools=[
                select_retrieval_tool,
                route_plugin_config_request,
                query_expansion,
                query_rewrite,
                subquery_decomposition,
                multi_query_rag,
                hyde_retrieve_docs,
                analyze_question,
                judge_retrieval_freshness,
                judge_answer_quality,
                retrieve_docs,
                multi_query_retrieve_docs,
                get_server_plugins_list,
            ],
            subagents=[plugin_config_agent],
            system_prompt=DEEPAGENT_SYSTEM_PROMPT,
        )
