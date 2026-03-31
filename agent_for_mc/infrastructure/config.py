from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_for_mc.infrastructure.dotenv import load_dotenv

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib  # type: ignore[no-redef]


BASE_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG_PATH = BASE_DIR / "config.toml"


def _get_env(
    primary_name: str,
    *,
    fallback_name: str | None = None,
    default: str | None = None,
) -> str | None:
    value = os.getenv(primary_name)
    if value:
        return value
    if fallback_name:
        fallback_value = os.getenv(fallback_name)
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
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


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
class Settings:
    lance_db_dir: Path
    lance_table_name: str
    jina_api_key: str | None
    jina_embeddings_url: str
    jina_embeddings_model: str
    jina_embeddings_task: str
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
    memory_db_path: Path
    memory_recall_limit: int
    memory_min_confidence: float
    memory_consolidation_turns: int
    plugin_config_db_dir: Path
    plugin_config_table_name: str
    plugin_config_top_k: int
    plugin_config_preview_chars: int
    plugin_config_summary_chars: int

    @property
    def deepseek_api_base(self) -> str:
        if self.deepseek_chat_url.endswith("/chat/completions"):
            return self.deepseek_chat_url[: -len("/chat/completions")] + "/v1"
        return self.deepseek_chat_url.rstrip("/")

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        config_path = Path(
            _get_env("RAG_CONFIG_TOML", default=str(DEFAULT_CONFIG_PATH))
            or str(DEFAULT_CONFIG_PATH)
        )
        config = _load_toml_config(config_path)
        config_base_dir = config_path.parent
        model_cache_dir = _resolve_path(
            _get_env(
                "RAG_MODEL_CACHE_DIR",
                default=str(
                    _config_value(config, "paths", "model_cache_dir", ".cache/model_cache")
                ),
            )
            or ".cache/model_cache",
            base_dir=config_base_dir,
        )
        _configure_model_cache_env(model_cache_dir)
        return cls(
            lance_db_dir=_resolve_path(
                _get_env(
                    "RAG_LANCE_DB_DIR",
                    default=str(
                        _config_value(
                            config,
                            "paths",
                            "lance_db_dir",
                            "data/plugins_vector_db",
                        )
                    ),
                )
                or "data/plugins_vector_db",
                base_dir=config_base_dir,
            ),
            lance_table_name=_get_env(
                "RAG_LANCE_TABLE_NAME",
                fallback_name="VECTOR_TABLE_NAME",
                default=str(
                    _config_value(
                        config, "vector_store", "lance_table_name", "plugins_docs"
                    )
                ),
            )
            or "plugins_docs",
            jina_api_key=_get_env("RAG_JINA_API_KEY", fallback_name="JINA_API_KEY"),
            jina_embeddings_url=_get_env(
                "RAG_JINA_EMBEDDINGS_URL",
                fallback_name="JINA_EMBEDDINGS_URL",
                default=str(
                    _config_value(
                        config,
                        "jina",
                        "embeddings_url",
                        "https://api.jina.ai/v1/embeddings",
                    )
                ),
            )
            or "https://api.jina.ai/v1/embeddings",
            jina_embeddings_model=_get_env(
                "RAG_JINA_EMBEDDINGS_MODEL",
                fallback_name="JINA_EMBEDDINGS_MODEL",
                default=str(
                    _config_value(
                        config,
                        "jina",
                        "embeddings_model",
                        "jina-embeddings-v5-text-small",
                    )
                ),
            )
            or "jina-embeddings-v5-text-small",
            jina_embeddings_task=str(
                _config_value(config, "jina", "embeddings_task", "retrieval.query")
            ),
            deepseek_api_key=_get_env(
                "RAG_DEEPSEEK_API_KEY",
                fallback_name="DEEPSEEK_API_KEY",
            ),
            deepseek_model=_get_env(
                "RAG_DEEPSEEK_MODEL",
                fallback_name="DEEPSEEK_MODEL",
                default=str(_config_value(config, "deepseek", "model", "deepseek-chat")),
            )
            or "deepseek-chat",
            deepseek_chat_url=_get_env(
                "RAG_DEEPSEEK_CHAT_URL",
                fallback_name="DEEPSEEK_CHAT_URL",
                default=str(
                    _config_value(
                        config,
                        "deepseek",
                        "chat_url",
                        "https://api.deepseek.com/chat/completions",
                    )
                ),
            )
            or "https://api.deepseek.com/chat/completions",
            expected_embedding_dimension=int(
                _get_env(
                    "RAG_EXPECTED_EMBEDDING_DIMENSION",
                    default=str(
                        _config_value(
                            config, "vector_store", "expected_embedding_dimension", 1024
                        )
                    ),
                )
                or "1024"
            ),
            rewrite_history_turns=int(
                _get_env(
                    "RAG_REWRITE_HISTORY_TURNS",
                    default=str(
                        _config_value(config, "vector_store", "rewrite_history_turns", 4)
                    ),
                )
                or "4"
            ),
            retrieval_top_k=int(
                _get_env(
                    "RAG_RETRIEVAL_TOP_K",
                    default=str(
                        _config_value(config, "vector_store", "retrieval_top_k", 8)
                    ),
                )
                or "8"
            ),
            answer_top_k=int(
                _get_env(
                    "RAG_ANSWER_TOP_K",
                    default=str(_config_value(config, "vector_store", "answer_top_k", 4)),
                )
                or "4"
            ),
            citation_preview_chars=int(
                _get_env(
                    "RAG_CITATION_PREVIEW_CHARS",
                    default=str(
                        _config_value(
                            config, "vector_store", "citation_preview_chars", 200
                        )
                    ),
                )
                or "200"
            ),
            request_timeout_seconds=int(
                _get_env(
                    "RAG_REQUEST_TIMEOUT_SECONDS",
                    default=str(
                        _config_value(config, "runtime", "request_timeout_seconds", 60)
                    ),
                )
                or "60"
            ),
            model_cache_dir=model_cache_dir,
            reranker_enabled=_parse_bool(
                _get_env(
                    "RAG_RERANKER_ENABLED",
                    default=str(_config_value(config, "reranker", "enabled", False)),
                ),
                default=False,
            ),
            reranker_model_name_or_path=str(
                _get_env(
                    "RAG_RERANKER_MODEL_NAME_OR_PATH",
                    default=str(
                        _config_value(
                            config,
                            "reranker",
                            "model_name_or_path",
                            "maidalun1020/bce-reranker-base_v1",
                        )
                    ),
                )
                or "maidalun1020/bce-reranker-base_v1"
            ),
            plugin_config_agent_model=str(
                _get_env(
                    "RAG_PLUGIN_CONFIG_AGENT_MODEL",
                    default=str(
                        _config_value(
                            config,
                            "plugin_config_agent",
                            "model",
                            "deepseek-chat",
                        )
                    ),
                )
                or "deepseek-chat"
            ),
            memory_maintenance_agent_model=str(
                _get_env(
                    "RAG_MEMORY_MAINTENANCE_AGENT_MODEL",
                    default=str(
                        _config_value(
                            config,
                            "memory_maintenance_agent",
                            "model",
                            "deepseek-chat",
                        )
                    ),
                )
                or "deepseek-chat"
            ),
            memory_enabled=_parse_bool(
                _get_env(
                    "RAG_MEMORY_ENABLED",
                    default=str(_config_value(config, "memory", "enabled", False)),
                ),
                default=False,
            ),
            memory_db_path=_resolve_path(
                _get_env(
                    "RAG_MEMORY_DB_PATH",
                    default=str(
                        _config_value(
                            config,
                            "memory",
                            "db_path",
                            ".cache/memory/memory.sqlite3",
                        )
                    ),
                )
                or ".cache/memory/memory.sqlite3",
                base_dir=config_base_dir,
            ),
            memory_recall_limit=int(
                _get_env(
                    "RAG_MEMORY_RECALL_LIMIT",
                    default=str(_config_value(config, "memory", "recall_limit", 5)),
                )
                or "5"
            ),
            memory_min_confidence=float(
                _get_env(
                    "RAG_MEMORY_MIN_CONFIDENCE",
                    default=str(_config_value(config, "memory", "min_confidence", 0.75)),
                )
                or "0.75"
            ),
            memory_consolidation_turns=int(
                _get_env(
                    "RAG_MEMORY_CONSOLIDATION_TURNS",
                    default=str(
                        _config_value(config, "memory", "consolidation_turns", 4)
                    ),
                )
                or "4"
            ),
            plugin_config_db_dir=_resolve_path(
                _get_env(
                    "RAG_PLUGIN_CONFIG_DB_DIR",
                    default=str(
                        _config_value(
                            config,
                            "plugin_config_store",
                            "db_dir",
                            "data/plugin_config_vector_db",
                        )
                    ),
                )
                or "data/plugin_config_vector_db",
                base_dir=config_base_dir,
            ),
            plugin_config_table_name=str(
                _get_env(
                    "RAG_PLUGIN_CONFIG_TABLE_NAME",
                    default=str(
                        _config_value(
                            config,
                            "plugin_config_store",
                            "table_name",
                            "plugin_config_docs",
                        )
                    ),
                )
                or "plugin_config_docs"
            ),
            plugin_config_top_k=int(
                _get_env(
                    "RAG_PLUGIN_CONFIG_TOP_K",
                    default=str(
                        _config_value(config, "plugin_config_store", "top_k", 6)
                    ),
                )
                or "6"
            ),
            plugin_config_preview_chars=int(
                _get_env(
                    "RAG_PLUGIN_CONFIG_PREVIEW_CHARS",
                    default=str(
                        _config_value(
                            config,
                            "plugin_config_store",
                            "preview_chars",
                            220,
                        )
                    ),
                )
                or "220"
            ),
            plugin_config_summary_chars=int(
                _get_env(
                    "RAG_PLUGIN_CONFIG_SUMMARY_CHARS",
                    default=str(
                        _config_value(
                            config,
                            "plugin_config_store",
                            "summary_chars",
                            500,
                        )
                    ),
                )
                or "500"
            ),
        )
