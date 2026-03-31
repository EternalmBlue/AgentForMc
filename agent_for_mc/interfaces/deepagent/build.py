from __future__ import annotations

from deepagents import create_deep_agent
from deepagents.middleware.subagents import SubAgent
from langchain_deepseek import ChatDeepSeek

from agent_for_mc.infrastructure.clients import DeepSeekChatClient, JinaEmbeddingClient
from agent_for_mc.application.retrieval import Retriever
from agent_for_mc.infrastructure.config import Settings
from agent_for_mc.infrastructure.plugin_config_vector_store import (
    LancePluginConfigVectorStore,
)
from agent_for_mc.infrastructure.ranker import BceRanker
from agent_for_mc.application.plugin_config import PluginConfigRetriever
from agent_for_mc.interfaces.tools.query_transform import (
    HydeToolContext,
    MultiQueryRagToolContext,
    QueryExpansionToolContext,
    QueryRewriteToolContext,
    configure_hyde_tool,
    configure_multi_query_rag_tool,
    configure_query_expansion_tool,
    configure_query_rewrite_tool,
    hyde_retrieve_docs,
    multi_query_rag,
    multi_query_retrieve_docs,
    query_expansion,
    query_rewrite,
    subquery_decomposition,
)
from agent_for_mc.interfaces.tools.routing import (
    PlanningToolContext,
    PluginConfigRoutingToolContext,
    analyze_question,
    configure_planning_tool,
    configure_plugin_config_routing_tool,
    route_plugin_config_request,
)
from agent_for_mc.interfaces.tools.retrieval import (
    JudgeAnswerQualityToolContext,
    JudgeRetrievalFreshnessToolContext,
    RetrieveDocsToolContext,
    SelectRetrievalToolContext,
    configure_judge_answer_quality_tool,
    configure_judge_retrieval_freshness_tool,
    configure_retrieve_docs_tool,
    configure_select_retrieval_tool,
    get_server_plugins_list,
    judge_answer_quality,
    judge_retrieval_freshness,
    retrieve_docs,
    select_retrieval_tool,
)
from agent_for_mc.interfaces.tools.plugin_config import (
    PluginConfigToolContext,
    configure_plugin_config_tool,
    retrieve_plugin_configs,
)
from agent_for_mc.interfaces.deepagent.prompts import (
    DEEPAGENT_SYSTEM_PROMPT,
    MEMORY_MAINTENANCE_SYSTEM_PROMPT,
    PLUGIN_CONFIG_AGENT_SYSTEM_PROMPT,
)


def build_deep_agent(
    *,
    settings: Settings,
    retriever: Retriever,
    ranker: BceRanker | None = None,
) -> object | None:
    if not settings.deepseek_api_key:
        return None

    planning_client = DeepSeekChatClient(settings)
    embedding_client = JinaEmbeddingClient(settings)
    configure_select_retrieval_tool(
        SelectRetrievalToolContext(
            available_backends=("chunk",),
            default_backend="chunk",
        )
    )
    configure_plugin_config_routing_tool(
        PluginConfigRoutingToolContext(client=planning_client)
    )
    configure_query_expansion_tool(QueryExpansionToolContext(client=planning_client))
    configure_query_rewrite_tool(QueryRewriteToolContext(client=planning_client))
    configure_multi_query_rag_tool(MultiQueryRagToolContext())
    configure_hyde_tool(HydeToolContext())
    configure_planning_tool(PlanningToolContext(client=planning_client))
    configure_judge_retrieval_freshness_tool(JudgeRetrievalFreshnessToolContext())
    configure_judge_answer_quality_tool(JudgeAnswerQualityToolContext())
    plugin_config_vector_store = LancePluginConfigVectorStore(
        settings.plugin_config_db_dir,
        settings.plugin_config_table_name,
        expected_embedding_dimension=settings.expected_embedding_dimension,
    )
    configure_plugin_config_tool(
        PluginConfigToolContext(
            retriever=PluginConfigRetriever(
                plugin_config_vector_store,
                embedding_client,
                ranker=ranker,
            ),
            summarizer_client=planning_client,
            top_k=settings.plugin_config_top_k,
            preview_chars=settings.plugin_config_preview_chars,
            summary_max_chars=settings.plugin_config_summary_chars,
        )
    )
    configure_retrieve_docs_tool(
        RetrieveDocsToolContext(
            retriever=retriever,
            top_k=settings.retrieval_top_k,
            citation_preview_chars=settings.citation_preview_chars,
        )
    )

    model = ChatDeepSeek(
        model=settings.deepseek_model,
        api_key=settings.deepseek_api_key,
        api_base=settings.deepseek_api_base,
        temperature=0.1,
        timeout=settings.request_timeout_seconds,
    )
    plugin_config_model = ChatDeepSeek(
        model=settings.plugin_config_agent_model,
        api_key=settings.deepseek_api_key,
        api_base=settings.deepseek_api_base,
        temperature=0.1,
        timeout=settings.request_timeout_seconds,
    )
    plugin_config_agent: SubAgent = {
        "name": "plugin_config_agent",
        "description": (
            "Use this agent for plugin config files, defaults, file paths, dependency wiring, "
            "and config-file differences."
        ),
        "system_prompt": PLUGIN_CONFIG_AGENT_SYSTEM_PROMPT,
        "tools": [retrieve_plugin_configs],
        "model": plugin_config_model,
    }

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


def build_memory_maintenance_agent(*, settings: Settings) -> object | None:
    if not settings.deepseek_api_key:
        return None

    model = ChatDeepSeek(
        model=settings.memory_maintenance_agent_model,
        api_key=settings.deepseek_api_key,
        api_base=settings.deepseek_api_base,
        temperature=0.1,
        timeout=settings.request_timeout_seconds,
    )
    return create_deep_agent(
        model=model,
        system_prompt=MEMORY_MAINTENANCE_SYSTEM_PROMPT,
    )
