from __future__ import annotations

from agent_for_mc.application.chat_session import RagChatSession
from agent_for_mc.application.memory_service import build_memory_service
from agent_for_mc.application.plugin_semantic_agent import build_plugin_semantic_service
from agent_for_mc.application.retrieval import Retriever
from agent_for_mc.infrastructure.clients import build_embedding_client
from agent_for_mc.infrastructure.config import Settings
from agent_for_mc.infrastructure.observability import configure_observability
from agent_for_mc.infrastructure.ranker import build_reranker_client
from agent_for_mc.infrastructure.vector_store import LancePluginVectorStore
from agent_for_mc.interfaces.deepagent import (
    build_deep_agent,
    build_memory_maintenance_agent,
)


def build_session(
    settings: Settings,
    *,
    memory_scope_id: str,
    ranker: object | None = None,
    plugin_semantic_service: object | None = None,
    attach_plugin_semantic_service_to_session: bool = True,
    configure_runtime_observability: bool = True,
) -> RagChatSession:
    if configure_runtime_observability:
        configure_observability()

    vector_store = LancePluginVectorStore(
        settings.plugin_docs_vector_db_dir,
        settings.plugin_docs_table_name,
        expected_embedding_dimension=settings.expected_embedding_dimension,
    )
    embedding_client = build_embedding_client(settings)
    resolved_ranker = ranker
    if resolved_ranker is None and settings.reranker_enabled:
        resolved_ranker = build_reranker_client(settings)

    retriever = Retriever(
        vector_store,
        embedding_client,
        ranker=resolved_ranker,
        bm25_enabled=settings.plugin_docs_bm25_enabled,
        bm25_top_k=settings.plugin_docs_bm25_top_k,
        bm25_auto_create_index=settings.plugin_docs_bm25_auto_create_index,
    )
    memory_maintenance_agent = build_memory_maintenance_agent(settings=settings)
    memory_service = build_memory_service(
        settings,
        scope_id=memory_scope_id,
        maintenance_agent=memory_maintenance_agent,
    )

    resolved_plugin_semantic_service = plugin_semantic_service
    if resolved_plugin_semantic_service is None:
        resolved_plugin_semantic_service = build_plugin_semantic_service(settings)

    deep_agent = build_deep_agent(
        settings=settings,
        retriever=retriever,
        ranker=resolved_ranker,
        plugin_semantic_service=resolved_plugin_semantic_service,
    )
    return RagChatSession(
        settings=settings,
        vector_store=vector_store,
        deep_agent=deep_agent,
        memory_service=memory_service,
        plugin_semantic_service=(
            resolved_plugin_semantic_service
            if attach_plugin_semantic_service_to_session
            else None
        ),
    )
