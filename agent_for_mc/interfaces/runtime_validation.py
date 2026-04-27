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

    if not settings.deepseek_api_key:
        raise ConfigurationError("Missing environment variable RAG_DEEPSEEK_API_KEY.")
    if settings.plugin_docs_bm25_enabled and settings.plugin_docs_bm25_top_k < 1:
        raise ConfigurationError("plugin_docs_store.bm25_top_k must be > 0.")
    if settings.memory_enabled and settings.memory_consolidation_turns < 1:
        raise ConfigurationError("memory.consolidation_turns must be > 0.")
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
