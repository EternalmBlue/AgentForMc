from __future__ import annotations

from agent_for_mc.domain.errors import ConfigurationError
from agent_for_mc.infrastructure.clients import validate_embedding_settings
from agent_for_mc.infrastructure.config import Settings


def validate_runtime_settings(
    settings: Settings,
    *,
    require_grpc: bool = False,
) -> None:
    validate_embedding_settings(settings)

    if not settings.llm_api_key:
        raise ConfigurationError(
            "Missing LLM API key. "
            "Set the env var RAG_LLM_API_KEY (or RAG_DEEPSEEK_API_KEY for backward compatibility)."
        )
    if settings.plugin_docs_bm25_enabled and settings.plugin_docs_bm25_top_k < 1:
        raise ConfigurationError("plugin_docs_store.bm25_top_k must be > 0.")
    if settings.reranker_enabled:
        if not settings.reranker_auth_token or not settings.reranker_auth_token.strip():
            raise ConfigurationError(
                "Missing reranker auth token. Set the environment variable "
                "RAG_RERANKER_GRPC_AUTH_TOKEN."
            )
        if settings.reranker_port < 1 or settings.reranker_port > 65535:
            raise ConfigurationError("reranker.port must be within 1..65535.")
        if settings.reranker_timeout_seconds <= 0:
            raise ConfigurationError("reranker.timeout_seconds must be > 0.")
    if settings.memory_enabled and settings.memory_consolidation_turns < 1:
        raise ConfigurationError("memory.consolidation_turns must be > 0.")
    if settings.skill_max_bytes < 1024:
        raise ConfigurationError("skills.max_bytes must be >= 1024.")
    if settings.skill_selection_top_k < 1:
        raise ConfigurationError("skills.selection_top_k must be > 0.")
    if settings.skill_draft_ttl_seconds < 60:
        raise ConfigurationError("skills.draft_ttl_seconds must be >= 60.")
    if settings.web_research_enabled:
        if settings.web_research_provider != "zhipu":
            raise ConfigurationError(
                "web_research.provider must be 'zhipu' when web_research.enabled=true."
            )
        if not settings.resolved_web_research_api_key:
            raise ConfigurationError(
                "Missing Web Research API key. "
                "web_research reuses RAG_ZHIPU_API_KEY for Zhipu Web Search."
            )
        if not settings.web_research_url.strip():
            raise ConfigurationError("web_research.url must not be blank when enabled.")
        if settings.web_research_top_k < 1:
            raise ConfigurationError("web_research.top_k must be > 0 when enabled.")
    if require_grpc:
        if not settings.grpc_auth_token or not settings.grpc_auth_token.strip():
            raise ConfigurationError(
                "Missing gRPC auth token. Set the environment variable RAG_GRPC_AUTH_TOKEN."
            )
        if settings.grpc_port < 1 or settings.grpc_port > 65535:
            raise ConfigurationError("grpc.port must be within 1..65535.")
        if settings.grpc_max_workers < 1:
            raise ConfigurationError("grpc.max_workers must be > 0.")
        if settings.grpc_session_ttl_seconds < 1:
            raise ConfigurationError("grpc.session_ttl_seconds must be > 0.")
        if settings.grpc_sync_ttl_seconds < 1:
            raise ConfigurationError("grpc.sync_ttl_seconds must be > 0.")
