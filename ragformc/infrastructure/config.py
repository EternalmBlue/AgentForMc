from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent.parent


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

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            lance_db_dir=Path(
                _get_env(
                    "RAG_LANCE_DB_DIR",
                    default=str(BASE_DIR / "data" / "plugins_vector_db"),
                )
            ),
            lance_table_name=_get_env(
                "RAG_LANCE_TABLE_NAME",
                fallback_name="VECTOR_TABLE_NAME",
                default="plugins_docs",
            )
            or "plugins_docs",
            jina_api_key=_get_env("RAG_JINA_API_KEY", fallback_name="JINA_API_KEY"),
            jina_embeddings_url=_get_env(
                "RAG_JINA_EMBEDDINGS_URL",
                fallback_name="JINA_EMBEDDINGS_URL",
                default="https://api.jina.ai/v1/embeddings",
            )
            or "https://api.jina.ai/v1/embeddings",
            jina_embeddings_model=_get_env(
                "RAG_JINA_EMBEDDINGS_MODEL",
                fallback_name="JINA_EMBEDDINGS_MODEL",
                default="jina-embeddings-v5-text-small",
            )
            or "jina-embeddings-v5-text-small",
            jina_embeddings_task="retrieval.query",
            deepseek_api_key=_get_env(
                "RAG_DEEPSEEK_API_KEY",
                fallback_name="DEEPSEEK_API_KEY",
            ),
            deepseek_model=_get_env(
                "RAG_DEEPSEEK_MODEL",
                fallback_name="DEEPSEEK_MODEL",
                default="deepseek-chat",
            )
            or "deepseek-chat",
            deepseek_chat_url="https://api.deepseek.com/chat/completions",
            expected_embedding_dimension=1024,
            rewrite_history_turns=4,
            retrieval_top_k=8,
            answer_top_k=4,
            citation_preview_chars=200,
            request_timeout_seconds=60,
        )
