from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_for_mc.infrastructure.dotenv import load_dotenv
from agent_for_mc.infrastructure.runtime_paths import (
    default_config_path,
    ensure_external_runtime_layout,
    resolve_runtime_path,
    runtime_base_dir,
)

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib  # type: ignore[no-redef]


BASE_DIR = runtime_base_dir()
DEFAULT_CONFIG_PATH = default_config_path()


def _get_env(
    primary_name: str,
    *fallback_names: str,
    fallback_name: str | None = None,
    default: str | None = None,
) -> str | None:
    value = os.getenv(primary_name)
    if value:
        return value
    candidate_names = list(fallback_names)
    if fallback_name:
        candidate_names.append(fallback_name)
    for candidate_name in candidate_names:
        fallback_value = os.getenv(candidate_name)
        if fallback_value:
            return fallback_value
    return default


def _load_toml_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    return data if isinstance(data, dict) else {}


def _config_value(
    config: dict[str, Any],
    section: str,
    key: str,
    default: Any,
) -> Any:
    section_data = config.get(section, {})
    if not isinstance(section_data, dict):
        return default
    return section_data.get(key, default)


def _parse_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1", "on"}
    return bool(value)


def _resolve_path(path_value: str | Path, *, base_dir: Path = BASE_DIR) -> Path:
    return resolve_runtime_path(path_value, base_dir=base_dir)


def _configure_model_cache_env(model_cache_dir: Path) -> None:
    model_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(model_cache_dir)
    os.environ["HF_HUB_CACHE"] = str(model_cache_dir / "huggingface_hub")
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(model_cache_dir / "huggingface_hub")
    os.environ["TRANSFORMERS_CACHE"] = str(model_cache_dir / "transformers")
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(
        model_cache_dir / "sentence_transformers"
    )


@dataclass(frozen=True, slots=True)
class RuntimeConfigSource:
    path: Path
    data: dict[str, Any]
    base_dir: Path


def load_runtime_config_source() -> RuntimeConfigSource:
    load_dotenv()
    configured_path = _get_env("RAG_CONFIG_TOML")
    config_path = _resolve_path(
        configured_path or str(DEFAULT_CONFIG_PATH),
        base_dir=BASE_DIR,
    )
    ensure_external_runtime_layout(
        config_path=config_path,
        copy_default_config=configured_path is None,
    )
    return RuntimeConfigSource(
        path=config_path,
        data=_load_toml_config(config_path),
        base_dir=config_path.parent,
    )


@dataclass(frozen=True, slots=True)
class Settings:
    plugin_docs_vector_db_dir: Path
    plugin_docs_table_name: str
    deepseek_api_key: str | None
    deepseek_model: str
    deepseek_chat_url: str
    expected_embedding_dimension: int
    rewrite_history_turns: int
    retrieval_top_k: int
    answer_top_k: int
    citation_preview_chars: int
    request_timeout_seconds: int
    model_cache_dir: Path
    reranker_enabled: bool
    reranker_model_name_or_path: str
    plugin_config_agent_model: str
    memory_maintenance_agent_model: str
    memory_enabled: bool
    user_semantic_memory_db_path: Path
    memory_recall_limit: int
    memory_min_confidence: float
    memory_consolidation_turns: int
    plugin_semantic_mc_servers_root: Path
    plugin_semantic_agent_model: str
    plugin_semantic_agent_scan_on_startup: bool
    plugin_semantic_agent_refresh_interval_seconds: int
    plugin_semantic_agent_max_file_chars: int
    plugin_semantic_agent_max_files_per_plugin: int
    server_config_semantic_vector_db_dir: Path
    server_config_semantic_table_name: str
    server_config_semantic_top_k: int
    server_config_semantic_preview_chars: int
    plugin_docs_bm25_enabled: bool = True
    plugin_docs_bm25_top_k: int = 7
    plugin_docs_bm25_auto_create_index: bool = True
    grpc_host: str = "127.0.0.1"
    grpc_port: int = 50051
    grpc_auth_token: str | None = None
    grpc_max_workers: int = 8
    grpc_session_ttl_seconds: int = 1800
    grpc_sync_ttl_seconds: int = 3600
    grpc_upload_tmp_dir: Path = BASE_DIR / ".cache" / "grpc_uploads"
    server_instance_bindings_path: Path = BASE_DIR / "data" / "server_instance_bindings.json"
    embedding_api_key: str | None = None
    embedding_api_key_env: str = "RAG_ZHIPU_API_KEY"
    embedding_url: str = "https://open.bigmodel.cn/api/paas/v4/embeddings"
    embedding_model: str = "embedding-3"
    reranker_host: str = "127.0.0.1"
    reranker_port: int = 50052
    reranker_timeout_seconds: float = 10.0
    reranker_auth_token: str | None = None

    @property
    def deepseek_api_base(self) -> str:
        if self.deepseek_chat_url.endswith("/chat/completions"):
            return self.deepseek_chat_url[: -len("/chat/completions")] + "/v1"
        return self.deepseek_chat_url.rstrip("/")

    @property
    def resolved_embedding_api_key_env(self) -> str:
        explicit = str(self.embedding_api_key_env or "").strip()
        return explicit or "RAG_ZHIPU_API_KEY"

    @property
    def resolved_embedding_api_key(self) -> str | None:
        explicit = str(self.embedding_api_key or "").strip()
        if explicit:
            return explicit
        return None

    @property
    def resolved_embedding_url(self) -> str:
        explicit = str(self.embedding_url or "").strip()
        return explicit or "https://open.bigmodel.cn/api/paas/v4/embeddings"

    @property
    def resolved_embedding_model(self) -> str:
        explicit = str(self.embedding_model or "").strip()
        return explicit or "embedding-3"

    @property
    def resolved_embedding_dimensions(self) -> int:
        return self.expected_embedding_dimension

    @classmethod
    def from_env(cls) -> "Settings":
        config_source = load_runtime_config_source()
        config = config_source.data
        config_base_dir = config_source.base_dir
        model_cache_dir = _resolve_path(
            str(_config_value(config, "paths", "model_cache_dir", ".cache/models"))
            or ".cache/models",
            base_dir=config_base_dir,
        )
        _configure_model_cache_env(model_cache_dir)
        return cls(
            plugin_docs_vector_db_dir=_resolve_path(
                str(
                    _config_value(
                        config,
                        "paths",
                        "plugin_docs_vector_db_dir",
                        "data/plugin_docs_vector_db",
                    )
                )
                or "data/plugin_docs_vector_db",
                base_dir=config_base_dir,
            ),
            plugin_docs_table_name=str(
                _config_value(config, "plugin_docs_store", "table_name", "plugin_docs")
            )
            or "plugin_docs",
            deepseek_api_key=_get_env("RAG_DEEPSEEK_API_KEY", "DEEPSEEK_API_KEY"),
            deepseek_model=str(_config_value(config, "deepseek", "model", "deepseek-chat"))
            or "deepseek-chat",
            deepseek_chat_url=str(
                _config_value(
                    config,
                    "deepseek",
                    "chat_url",
                    "https://api.deepseek.com/chat/completions",
                )
            )
            or "https://api.deepseek.com/chat/completions",
            expected_embedding_dimension=int(
                str(_config_value(config, "embedding", "dimensions", 1024))
                or "1024"
            ),
            rewrite_history_turns=int(
                str(_config_value(config, "chat", "rewrite_history_turns", 4))
                or "4"
            ),
            retrieval_top_k=int(
                str(_config_value(config, "plugin_docs_store", "retrieval_top_k", 5)) or "5"
            ),
            answer_top_k=int(
                str(_config_value(config, "plugin_docs_store", "answer_top_k", 4))
                or "4"
            ),
            citation_preview_chars=int(
                str(_config_value(config, "plugin_docs_store", "citation_preview_chars", 200))
                or "200"
            ),
            plugin_docs_bm25_enabled=_parse_bool(
                _config_value(config, "plugin_docs_store", "bm25_enabled", True),
                default=True,
            ),
            plugin_docs_bm25_top_k=int(
                str(_config_value(config, "plugin_docs_store", "bm25_top_k", 7)) or "7"
            ),
            plugin_docs_bm25_auto_create_index=_parse_bool(
                _config_value(
                    config,
                    "plugin_docs_store",
                    "bm25_auto_create_index",
                    True,
                ),
                default=True,
            ),
            request_timeout_seconds=int(
                str(_config_value(config, "runtime", "request_timeout_seconds", 60))
                or "60"
            ),
            model_cache_dir=model_cache_dir,
            reranker_enabled=_parse_bool(
                _config_value(config, "reranker", "enabled", False),
                default=False,
            ),
            reranker_model_name_or_path=str(
                _config_value(
                    config,
                    "reranker",
                    "model_name_or_path",
                    "maidalun1020/bce-reranker-base_v1",
                )
                or "maidalun1020/bce-reranker-base_v1"
            ),
            reranker_host=str(
                _config_value(config, "reranker", "host", "127.0.0.1")
            )
            or "127.0.0.1",
            reranker_port=int(
                str(_config_value(config, "reranker", "port", 50052)) or "50052"
            ),
            reranker_timeout_seconds=float(
                str(_config_value(config, "reranker", "timeout_seconds", 10.0))
                or "10.0"
            ),
            reranker_auth_token=_get_env("RAG_RERANKER_GRPC_AUTH_TOKEN"),
            plugin_config_agent_model=str(
                _config_value(config, "plugin_config_agent", "model", "deepseek-chat")
                or "deepseek-chat"
            ),
            memory_maintenance_agent_model=str(
                _config_value(config, "memory_maintenance_agent", "model", "deepseek-chat")
                or "deepseek-chat"
            ),
            memory_enabled=_parse_bool(
                _config_value(config, "memory", "enabled", False),
                default=False,
            ),
            user_semantic_memory_db_path=_resolve_path(
                str(
                    _config_value(
                        config,
                        "memory",
                        "db_path",
                        "data/user_semantic_memory.sqlite3",
                    )
                )
                or "data/user_semantic_memory.sqlite3",
                base_dir=config_base_dir,
            ),
            memory_recall_limit=int(
                str(_config_value(config, "memory", "recall_limit", 5))
                or "5"
            ),
            memory_min_confidence=float(
                str(_config_value(config, "memory", "min_confidence", 0.75))
                or "0.75"
            ),
            memory_consolidation_turns=int(
                str(_config_value(config, "memory", "consolidation_turns", 4))
                or "4"
            ),
            plugin_semantic_mc_servers_root=_resolve_path(
                str(
                    _config_value(
                        config,
                        "plugin_semantic_agent",
                        "mc_servers_root",
                        "mc_servers",
                    )
                )
                or "mc_servers",
                base_dir=config_base_dir,
            ),
            plugin_semantic_agent_model=str(
                _config_value(config, "plugin_semantic_agent", "model", "deepseek-chat")
                or "deepseek-chat"
            ),
            plugin_semantic_agent_scan_on_startup=_parse_bool(
                _config_value(config, "plugin_semantic_agent", "scan_on_startup", True),
                default=True,
            ),
            plugin_semantic_agent_refresh_interval_seconds=int(
                str(
                    _config_value(
                        config,
                        "plugin_semantic_agent",
                        "refresh_interval_seconds",
                        1800,
                    )
                )
                or "1800"
            ),
            plugin_semantic_agent_max_file_chars=int(
                str(
                    _config_value(config, "plugin_semantic_agent", "max_file_chars", 12000)
                )
                or "12000"
            ),
            plugin_semantic_agent_max_files_per_plugin=int(
                str(
                    _config_value(config, "plugin_semantic_agent", "max_files_per_plugin", 20)
                )
                or "20"
            ),
            server_config_semantic_vector_db_dir=_resolve_path(
                str(
                    _config_value(
                        config,
                        "server_config_semantic_store",
                        "db_dir",
                        "data/server_config_semantic_vector_db",
                    )
                )
                or "data/server_config_semantic_vector_db",
                base_dir=config_base_dir,
            ),
            server_config_semantic_table_name=str(
                _config_value(
                    config,
                    "server_config_semantic_store",
                    "table_name",
                    "server_config_semantic_memories",
                )
                or "server_config_semantic_memories"
            ),
            server_config_semantic_top_k=int(
                str(_config_value(config, "server_config_semantic_store", "top_k", 8))
                or "8"
            ),
            server_config_semantic_preview_chars=int(
                str(
                    _config_value(
                        config,
                        "server_config_semantic_store",
                        "preview_chars",
                        220,
                    )
                )
                or "220"
            ),
            grpc_host=str(_config_value(config, "grpc", "host", "127.0.0.1"))
            or "127.0.0.1",
            grpc_port=int(str(_config_value(config, "grpc", "port", 50051)) or "50051"),
            grpc_auth_token=_get_env("RAG_GRPC_AUTH_TOKEN"),
            grpc_max_workers=int(
                str(_config_value(config, "grpc", "max_workers", 8))
                or "8"
            ),
            grpc_session_ttl_seconds=int(
                str(_config_value(config, "grpc", "session_ttl_seconds", 1800))
                or "1800"
            ),
            grpc_sync_ttl_seconds=int(
                str(_config_value(config, "grpc", "sync_ttl_seconds", 3600))
                or "3600"
            ),
            grpc_upload_tmp_dir=_resolve_path(
                str(_config_value(config, "grpc", "upload_tmp_dir", ".cache/grpc_uploads"))
                or ".cache/grpc_uploads",
                base_dir=config_base_dir,
            ),
            server_instance_bindings_path=_resolve_path(
                str(
                    _config_value(
                        config,
                        "server_identity",
                        "bindings_path",
                        "data/server_instance_bindings.json",
                    )
                )
                or "data/server_instance_bindings.json",
                base_dir=config_base_dir,
            ),
            embedding_api_key=_get_env("RAG_ZHIPU_API_KEY"),
            embedding_api_key_env="RAG_ZHIPU_API_KEY",
            embedding_url=str(
                _config_value(
                    config,
                    "embedding",
                    "url",
                    "https://open.bigmodel.cn/api/paas/v4/embeddings",
                )
            ).strip()
            or "https://open.bigmodel.cn/api/paas/v4/embeddings",
            embedding_model=str(
                _config_value(
                    config,
                    "embedding",
                    "model",
                    "embedding-3",
                )
            ).strip()
            or "embedding-3",
        )
