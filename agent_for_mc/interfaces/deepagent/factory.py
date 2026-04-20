from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_deepseek import ChatDeepSeek

from agent_for_mc.application.plugin_config import PluginConfigRetriever
from agent_for_mc.application.retrieval import Retriever
from agent_for_mc.infrastructure.clients import DeepSeekChatClient, JinaEmbeddingClient
from agent_for_mc.infrastructure.config import Settings
from agent_for_mc.infrastructure.observability import configure_observability
from agent_for_mc.infrastructure.plugin_config_vector_store import (
    LancePluginConfigVectorStore,
)
from agent_for_mc.infrastructure.ranker import BceRanker
from agent_for_mc.infrastructure.semantic_memory_vector_store import (
    LanceSemanticMemoryVectorStore,
)
from agent_for_mc.interfaces.tools.memory import (
    PluginSemanticRefreshToolContext,
    configure_plugin_semantic_refresh_tool,
)
from agent_for_mc.interfaces.tools.plugin_config import (
    PluginConfigToolContext,
    configure_plugin_config_tool,
)
from agent_for_mc.interfaces.tools.query_transform import (
    HydeToolContext,
    MultiQueryRagToolContext,
    QueryExpansionToolContext,
    QueryRewriteToolContext,
    configure_hyde_tool,
    configure_multi_query_rag_tool,
    configure_query_expansion_tool,
    configure_query_rewrite_tool,
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
)
from agent_for_mc.application.semantic_memory import SemanticMemoryRetriever
from agent_for_mc.interfaces.tools.routing import (
    PlanningToolContext,
    PluginConfigRoutingToolContext,
    configure_planning_tool,
    configure_plugin_config_routing_tool,
)

if TYPE_CHECKING:
    from agent_for_mc.application.plugin_semantic_agent import PluginSemanticAgentService


def build_chat_model(*, settings: Settings, model_name: str) -> ChatDeepSeek:
    return ChatDeepSeek(
        model=model_name,
        api_key=settings.deepseek_api_key,
        api_base=settings.deepseek_api_base,
        temperature=0.1,
        timeout=settings.request_timeout_seconds,
    )


def configure_deepagent_dependencies(
    *,
    settings: Settings,
    retriever: Retriever,
    ranker: BceRanker | None = None,
    plugin_semantic_service: PluginSemanticAgentService | None = None,
) -> None:
    configure_observability()

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
    if plugin_semantic_service is not None:
        configure_plugin_semantic_refresh_tool(
            PluginSemanticRefreshToolContext(service=plugin_semantic_service)
        )

    plugin_config_vector_store = LancePluginConfigVectorStore(
        settings.plugin_config_db_dir,
        settings.plugin_config_table_name,
        expected_embedding_dimension=settings.expected_embedding_dimension,
    )
    semantic_memory_vector_store = LanceSemanticMemoryVectorStore(
        settings.semantic_memory_db_dir,
        settings.semantic_memory_table_name,
        expected_embedding_dimension=settings.expected_embedding_dimension,
    )
    configure_plugin_config_tool(
        PluginConfigToolContext(
            retriever=PluginConfigRetriever(
                plugin_config_vector_store,
                embedding_client,
                ranker=ranker,
            ),
            semantic_retriever=SemanticMemoryRetriever(
                semantic_memory_vector_store,
                embedding_client,
            ),
            summarizer_client=planning_client,
            top_k=settings.plugin_config_top_k,
            semantic_top_k=settings.semantic_memory_top_k,
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
