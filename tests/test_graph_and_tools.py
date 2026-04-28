from __future__ import annotations

import os
import sqlite3
from concurrent import futures
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
import json

import grpc
import lancedb
import pytest

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent_for_mc.application.chat_session import RagChatSession
from agent_for_mc.application.plugin_semantic_agent import (
    PluginSemanticAgentService,
    PluginSemanticExtractionResult,
    scan_plugin_semantic_bundles,
)
from agent_for_mc.application.memory_service import (
    MemoryService,
    build_memory_service,
    extract_memory_candidates,
    format_memory_context,
    validate_memory_actions,
)
from agent_for_mc.application.deepagent_state import (
    clear_turn_context,
    record_server_plugins,
    record_retrieved_docs,
    start_turn_context,
)
from agent_for_mc.application.prompts import format_history
from agent_for_mc.application.retrieval import Retriever
from agent_for_mc.domain.models import (
    AnswerResult,
    RetrievedDoc,
    SemanticMemoryDoc,
    SemanticMemoryEntry,
)
from agent_for_mc.infrastructure.config import Settings
from agent_for_mc.infrastructure.clients import (
    OpenAICompatibleEmbeddingClient,
    build_embedding_client,
    validate_embedding_settings,
)
from agent_for_mc.infrastructure.memory_store import MemoryAction, MemoryCandidate, SQLiteMemoryStore
from agent_for_mc.infrastructure.vector_store import LancePluginVectorStore
from agent_for_mc.domain.errors import ConfigurationError, ServiceError
from agent_for_mc.infrastructure.ranker import GrpcRerankerClient
from agent_for_mc.interfaces.grpc import reranker_pb2, reranker_pb2_grpc
from agent_for_mc.interfaces.runtime_validation import validate_runtime_settings
from agent_for_mc.interfaces.deepagent import build_deep_agent
from agent_for_mc.interfaces.deepagent import main_agent as deepagent_main_agent_module
from agent_for_mc.interfaces.deepagent import memory_agent as deepagent_memory_agent_module
from agent_for_mc.interfaces import session_factory as session_factory_module
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
    build_retrieve_docs_payload,
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
from agent_for_mc.interfaces.tools.memory import (
    PluginSemanticRefreshToolContext,
    configure_plugin_semantic_refresh_tool,
    refresh_plugin_semantic_memory,
)


class FakeVectorStore:
    def find_name_matches(self, search_query: str):
        return [
            RetrievedDoc(
                id=1,
                plugin_chinese_name="插件A",
                plugin_english_name="PluginA",
                content="name match content",
                distance=0.0,
                match_reason="name-boost",
            )
        ]

    def search_by_embedding(self, embedding, *, top_k: int):
        return [
            RetrievedDoc(
                id=2,
                plugin_chinese_name="插件B",
                plugin_english_name="PluginB",
                content="vector content",
                distance=0.12,
                match_reason="vector",
            )
        ]


class FakeBM25VectorStore(FakeVectorStore):
    def __init__(self):
        self.bm25_calls: list[tuple[str, int, bool]] = []

    def search_by_bm25(
        self,
        search_query: str,
        *,
        top_k: int,
        auto_create_index: bool = True,
    ):
        self.bm25_calls.append((search_query, top_k, auto_create_index))
        return [
            RetrievedDoc(
                id=3,
                plugin_chinese_name="插件C",
                plugin_english_name="PluginC",
                content="bm25 content",
                distance=1.25,
                match_reason="bm25",
            )
        ]


class FakeEmbeddingClient:
    def embed_query(self, search_query: str):
        return [0.0, 1.0]


class FakeSemanticMemoryRetriever:
    def retrieve(self, search_query: str, *, top_k: int = 8, server_id=None, plugin_name=None):
        return [
            SemanticMemoryDoc(
                server_id="lobby-1",
                plugin_name="MMORPG",
                memory_type="plugin_config",
                relation_type="contains",
                memory_text="mmorpg/config.yml contains mob-spawn-rate: 1.0",
                distance=0.0,
                match_reason="name-boost",
            ),
            SemanticMemoryDoc(
                server_id="lobby-1",
                plugin_name="MMORPG",
                memory_type="plugin_config",
                relation_type="contains",
                memory_text="mmorpg/mob.yml contains mob-level: 5",
                distance=0.13,
                match_reason="vector",
            ),
        ]


class FakePluginConfigSummaryClient:
    def chat(self, messages, *, temperature: float = 0.0):
        system_prompt = str(messages[0]["content"])
        lowered = system_prompt.lower()
        if (
            "插件配置整理助手" in system_prompt
            or "plugin configuration" in lowered
            or "server configuration" in lowered
        ):
            return "MMORPG 的配置分散在 config.yml 和 mob.yml 中，当前问题涉及怪物生成和等级设置。"
        return "unexpected prompt"


class FakeDeepAgent:
    def invoke(self, payload):
        record_retrieved_docs(
            [
                RetrievedDoc(
                    id=777,
                    plugin_chinese_name="深度代理插件",
                    plugin_english_name="DeepAgentPlugin",
                    content="deep agent content",
                    distance=0.01,
                    match_reason="deep-agent",
                )
            ]
        )
        return {"messages": [AIMessage(content="deep answer")]}


class MemoryAwareDeepAgent:
    def __init__(self):
        self.invocations: list[list[object]] = []

    def invoke(self, payload):
        messages = list(payload["messages"])
        self.invocations.append(messages)

        memory_message = None
        for message in messages:
            if isinstance(message, SystemMessage) and "长期记忆" in str(message.content):
                memory_message = str(message.content)
                break

        if memory_message:
            return {
                "messages": [
                    AIMessage(content=f"收到记忆：{memory_message}"),
                ]
            }

        return {"messages": [AIMessage(content="第一次回复，已记录。")]} 


class FakePlanningClient:
    def chat(self, messages, *, temperature: float = 0.0):
        return json.dumps(
            {
                "standalone_query": "plugin config issue",
                "need_multi_query": True,
                "queries": [
                    "plugin config issue",
                    "plugin config issue version differences",
                    "plugin config issue related plugins",
                ],
                "need_plugins": True,
            },
            ensure_ascii=False,
        )


class FakeRoutingClient:
    def __init__(self, route: str):
        self.route = route

    def chat(self, messages, *, temperature: float = 0.0):
        if self.route == "plugin_config_agent":
            return json.dumps(
                {
                    "route": "plugin_config_agent",
                    "use_subagent": True,
                    "normalized_query": "How should I set the plugin config defaults?",
                    "reason": "The question is about defaults and file-level configuration.",
                    "confidence": 0.96,
                },
                ensure_ascii=False,
            )

        return json.dumps(
            {
                "route": "main_agent",
                "use_subagent": False,
                "normalized_query": "How do I compare plugin behavior?",
                "reason": "The question is not about plugin config files.",
                "confidence": 0.91,
            },
            ensure_ascii=False,
        )


class FakeRewriteClient:
    def chat(self, messages, *, temperature: float = 0.0):
        return "plugin config issue for authme"


class FakeExpansionClient:
    def chat(self, messages, *, temperature: float = 0.0):
        return json.dumps(
            {
                "expanded_question": "How should the AuthMe plugin configuration be updated for the latest release?",
                "search_query": "AuthMe plugin configuration latest release update",
            },
            ensure_ascii=False,
        )


class FakeSelectorClient:
    def __init__(self, backend: str):
        self._backend = backend

    def chat(self, messages, *, temperature: float = 0.0):
        return json.dumps(
            {
                "retrieval_backend": self._backend,
                "search_query": "AuthMe plugin configuration latest release",
                "reason": f"Selected {self._backend} for the request.",
                "confidence": 0.91,
                "fallback_used": False,
            },
            ensure_ascii=False,
        )


class FakeAdaptiveClient:
    def __init__(self):
        self.selection_calls = 0
        self.judge_calls = 0

    def chat(self, messages, *, temperature: float = 0.0):
        system = str(messages[0]["content"])
        user = str(messages[1]["content"])

        if "retrieval tool selector" in system:
            self.selection_calls += 1
            if self.selection_calls == 1:
                return json.dumps(
                    {
                        "retrieval_backend": "chunk",
                        "search_query": "AuthMe config",
                        "reason": "Initial short query.",
                        "confidence": 0.74,
                        "fallback_used": False,
                    },
                    ensure_ascii=False,
                )
            return json.dumps(
                {
                    "retrieval_backend": "chunk",
                    "search_query": "AuthMe plugin configuration latest release",
                    "reason": "Expanded to a version-specific query after reflection.",
                    "confidence": 0.91,
                    "fallback_used": False,
                },
                ensure_ascii=False,
            )

        if "query expansion tool" in system:
            return json.dumps(
                {
                    "expanded_question": "How should the AuthMe plugin configuration be updated for the latest release?",
                    "search_query": "AuthMe plugin configuration latest release",
                },
                ensure_ascii=False,
            )

        if "retrieval freshness judge" in system:
            self.judge_calls += 1
            if "latest release" in user or self.judge_calls > 1:
                return json.dumps(
                    {
                        "is_fresh_enough": True,
                        "is_covered_enough": True,
                        "needs_model_knowledge_fallback": False,
                        "reason": "Version-specific docs found.",
                        "confidence": 0.93,
                    },
                    ensure_ascii=False,
                )
            return json.dumps(
                {
                    "is_fresh_enough": False,
                    "is_covered_enough": False,
                    "needs_model_knowledge_fallback": True,
                    "reason": "Docs are generic and miss version-specific guidance.",
                    "confidence": 0.81,
                },
                ensure_ascii=False,
            )

        return "{}"


class FakeAdaptiveVectorStore:
    def __init__(self):
        self.last_query = ""

    def find_name_matches(self, search_query: str):
        self.last_query = search_query
        return [
            RetrievedDoc(
                id=21,
                plugin_chinese_name="鎻掍欢A",
                plugin_english_name="PluginA",
                content=f"name match content for {search_query}",
                distance=0.0,
                match_reason="name-boost",
            )
        ]

    def search_by_embedding(self, embedding, *, top_k: int):
        if "latest release" in self.last_query:
            return [
                RetrievedDoc(
                    id=31,
                    plugin_chinese_name="鎻掍欢A",
                    plugin_english_name="PluginA",
                    content="version-specific guidance for the latest release",
                    distance=0.05,
                    match_reason="vector",
                ),
                RetrievedDoc(
                    id=32,
                    plugin_chinese_name="鎻掍欢B",
                    plugin_english_name="PluginB",
                    content="updated configuration reference",
                    distance=0.09,
                    match_reason="vector",
                ),
            ]

        return [
            RetrievedDoc(
                id=11,
                plugin_chinese_name="鎻掍欢A",
                plugin_english_name="PluginA",
                content="generic configuration guidance",
                distance=0.12,
                match_reason="vector",
            )
        ]


class FakeAdaptiveEmbeddingClient:
    def __init__(self, vector_store: FakeAdaptiveVectorStore):
        self._vector_store = vector_store

    def embed_query(self, search_query: str):
        self._vector_store.last_query = search_query
        return [0.0, 1.0]


class FakeMultiQueryRagClient:
    def chat(self, messages, *, temperature: float = 0.0):
        return json.dumps(
            {
                "standalone_query": "plugin config issue",
                "queries": [
                    "plugin config issue",
                    "plugin config issue with authme",
                    "plugin config issue version differences",
                ],
            },
            ensure_ascii=False,
        )


class FakeJudgeClient:
    def chat(self, messages, *, temperature: float = 0.0):
        return json.dumps(
            {
                "is_fresh_enough": False,
                "is_covered_enough": True,
                "needs_model_knowledge_fallback": True,
                "reason": "Retrieved docs mention the right plugin but look version-agnostic.",
                "confidence": 0.82,
            },
            ensure_ascii=False,
        )


class FakeAnswerJudgeClient:
    def chat(self, messages, *, temperature: float = 0.0):
        return json.dumps(
            {
                "overall_score": 0.42,
                "is_good_enough": False,
                "needs_retry": True,
                "retry_recommendation": "query_expansion",
                "reason": "The draft answer is too generic and misses version-specific details.",
                "confidence": 0.88,
            },
            ensure_ascii=False,
        )


class FakeHydeClient:
    def chat(self, messages, *, temperature: float = 0.0):
        return json.dumps(
            {
                "hypothetical_answer": "The plugin config should be updated in the latest release.",
            },
            ensure_ascii=False,
        )


class FakeSubQueryClient:
    def chat(self, messages, *, temperature: float = 0.0):
        return json.dumps(
            {
                "standalone_query": "frontend spec and first week onboarding",
                "subqueries": [
                    {
                        "subquestion": "前端规范",
                        "search_query": "前端 规范",
                    },
                    {
                        "subquestion": "新人第一周安排",
                        "search_query": "新人 第一周 安排",
                    },
                ],
            },
            ensure_ascii=False,
        )


class FakeMemoryChatClient:
    def __init__(self):
        self.calls: list[str] = []

    def chat(self, messages, *, temperature: float = 0.0):
        system_prompt = str(messages[0]["content"])
        self.calls.append(system_prompt)

        if "memory consolidation" in system_prompt:
            return json.dumps(
                {
                    "actions": [
                        {
                            "action": "add",
                            "type": "preference",
                            "key": "answer_style",
                            "value": "简洁中文回答",
                            "confidence": 0.96,
                        },
                        {
                            "action": "add",
                            "type": "fact",
                            "key": "project_stack",
                            "value": "Paper",
                            "confidence": 0.9,
                        },
                    ]
                },
                ensure_ascii=False,
            )

        if "session summary" in system_prompt:
            return json.dumps(
                {
                    "session_summary": "用户偏好简洁中文回答，并且项目使用 Paper。",
                    "candidate_memories": [
                        {
                            "type": "preference",
                            "key": "answer_style",
                            "value": "简洁中文回答",
                            "confidence": 0.96,
                        },
                        {
                            "type": "fact",
                            "key": "project_stack",
                            "value": "Paper",
                            "confidence": 0.9,
                        },
                    ],
                },
                ensure_ascii=False,
            )

        raise AssertionError(f"unexpected memory prompt: {system_prompt}")


def make_settings() -> Settings:
    return Settings(
        plugin_docs_vector_db_dir=Path("data/plugin_docs_vector_db"),
        plugin_docs_table_name="plugin_docs",
        deepseek_api_key="test",
        deepseek_model="test-model",
        deepseek_chat_url="https://api.deepseek.com/chat/completions",
        expected_embedding_dimension=2,
        rewrite_history_turns=4,
        retrieval_top_k=2,
        answer_top_k=2,
        citation_preview_chars=40,
        request_timeout_seconds=5,
        model_cache_dir=Path(".cache/models"),
        reranker_enabled=False,
        reranker_model_name_or_path="maidalun1020/bce-reranker-base_v1",
        plugin_config_agent_model="deepseek-chat",
        memory_maintenance_agent_model="deepseek-chat",
        memory_enabled=False,
        user_semantic_memory_db_path=Path("data/user_semantic_memory.sqlite3"),
        memory_recall_limit=5,
        memory_min_confidence=0.75,
        memory_consolidation_turns=4,
        plugin_semantic_mc_servers_root=Path("mc_servers"),
        plugin_semantic_agent_model="deepseek-chat",
        plugin_semantic_agent_scan_on_startup=False,
        plugin_semantic_agent_refresh_interval_seconds=1800,
        plugin_semantic_agent_max_file_chars=12000,
        plugin_semantic_agent_max_files_per_plugin=20,
        server_config_semantic_vector_db_dir=Path("data/server_config_semantic_vector_db"),
        server_config_semantic_table_name="server_config_semantic_memories",
        server_config_semantic_top_k=8,
        server_config_semantic_preview_chars=220,
        embedding_api_key="test-zhipu-key",
        embedding_api_key_env="RAG_ZHIPU_API_KEY",
        embedding_url="https://open.bigmodel.cn/api/paas/v4/embeddings",
        embedding_model="embedding-3",
    )


def test_format_history_uses_base_messages():
    history = [
        HumanMessage(content="how to use plugin?"),
        AIMessage(content="use the config file"),
    ]

    text = format_history(history)

    assert "第1轮用户问题：how to use plugin?" in text
    assert "第1轮助手回答：use the config file" in text


def test_settings_from_env_loads_dotenv_and_config(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    config_path = tmp_path / "config.toml"
    env_file.write_text(
        r"""
RAG_ZHIPU_API_KEY=test-zhipu-key
RAG_DEEPSEEK_API_KEY=test-deepseek-key
RAG_GRPC_AUTH_TOKEN=test-grpc-token
RAG_RERANKER_GRPC_AUTH_TOKEN=test-reranker-token
""".strip(),
        encoding="utf-8",
    )
    config_path.write_text(
        """
[paths]
plugin_docs_vector_db_dir = "data/plugin_docs_vector_db"
model_cache_dir = ".cache/models"

[plugin_docs_store]
table_name = "plugin_docs"
retrieval_top_k = 8
answer_top_k = 4
citation_preview_chars = 200
bm25_enabled = true
bm25_top_k = 6
bm25_auto_create_index = false

[embedding]
dimensions = 1024

[chat]
rewrite_history_turns = 4

[reranker]
enabled = true
model_name_or_path = "maidalun1020/bce-reranker-base_v1"
host = "127.0.0.2"
port = 50053
timeout_seconds = 12.5

[plugin_config_agent]
model = "deepseek-lite"

[memory_maintenance_agent]
model = "deepseek-mini"

[plugin_semantic_agent]
mc_servers_root = "mc_servers"
model = "deepseek-config"
scan_on_startup = true
refresh_interval_seconds = 900
max_file_chars = 8000
max_files_per_plugin = 12

[server_config_semantic_store]
db_dir = "data/server_config_semantic_vector_db"
table_name = "server_config_semantic_memories"
top_k = 7
preview_chars = 180

[memory]
enabled = true
db_path = "data/user_semantic_memory.sqlite3"
recall_limit = 3
min_confidence = 0.8
consolidation_turns = 6

[grpc]
host = "127.0.0.1"
port = 50051
max_workers = 8
session_ttl_seconds = 1800
sync_ttl_seconds = 3600
upload_tmp_dir = ".cache/grpc_uploads"
""".strip(),
        encoding="utf-8",
    )

    monkeypatch.setenv("RAG_ENV_FILE", str(env_file))
    monkeypatch.setenv("RAG_CONFIG_TOML", str(config_path))
    monkeypatch.delenv("RAG_ZHIPU_API_KEY", raising=False)
    monkeypatch.delenv("RAG_DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("RAG_GRPC_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("RAG_RERANKER_GRPC_AUTH_TOKEN", raising=False)

    settings = Settings.from_env()

    assert settings.plugin_docs_vector_db_dir == (tmp_path / "data/plugin_docs_vector_db").resolve()
    assert settings.model_cache_dir == (tmp_path / ".cache/models").resolve()
    assert settings.embedding_api_key == "test-zhipu-key"
    assert settings.embedding_url == "https://open.bigmodel.cn/api/paas/v4/embeddings"
    assert settings.embedding_model == "embedding-3"
    assert settings.resolved_embedding_dimensions == 1024
    assert settings.plugin_docs_bm25_enabled is True
    assert settings.plugin_docs_bm25_top_k == 6
    assert settings.plugin_docs_bm25_auto_create_index is False
    assert settings.deepseek_api_key == "test-deepseek-key"
    assert settings.reranker_enabled is True
    assert settings.reranker_model_name_or_path == "maidalun1020/bce-reranker-base_v1"
    assert settings.reranker_host == "127.0.0.2"
    assert settings.reranker_port == 50053
    assert settings.reranker_timeout_seconds == 12.5
    assert settings.reranker_auth_token == "test-reranker-token"
    assert settings.plugin_config_agent_model == "deepseek-lite"
    assert settings.memory_maintenance_agent_model == "deepseek-mini"
    assert settings.memory_enabled is True
    assert settings.user_semantic_memory_db_path == (
        tmp_path / "data/user_semantic_memory.sqlite3"
    ).resolve()
    assert settings.memory_recall_limit == 3
    assert settings.memory_min_confidence == 0.8
    assert settings.memory_consolidation_turns == 6
    assert settings.plugin_semantic_mc_servers_root == (tmp_path / "mc_servers").resolve()
    assert settings.plugin_semantic_agent_model == "deepseek-config"
    assert settings.plugin_semantic_agent_scan_on_startup is True
    assert settings.plugin_semantic_agent_refresh_interval_seconds == 900
    assert settings.plugin_semantic_agent_max_file_chars == 8000
    assert settings.plugin_semantic_agent_max_files_per_plugin == 12
    assert settings.server_config_semantic_vector_db_dir == (
        tmp_path / "data/server_config_semantic_vector_db"
    ).resolve()
    assert settings.server_config_semantic_table_name == "server_config_semantic_memories"
    assert settings.server_config_semantic_top_k == 7
    assert settings.server_config_semantic_preview_chars == 180
    assert settings.server_instance_bindings_path == (
        tmp_path / "data/server_instance_bindings.json"
    ).resolve()
    assert settings.grpc_auth_token == "test-grpc-token"
    assert os.environ["HF_HOME"] == str(settings.model_cache_dir)
    assert os.environ["TRANSFORMERS_CACHE"] == str(
        settings.model_cache_dir / "transformers"
    )


def test_settings_from_env_uses_convention_defaults_for_optional_config(
    tmp_path, monkeypatch
):
    env_file = tmp_path / ".env"
    config_path = tmp_path / "config.toml"
    env_file.write_text(
        r"""
RAG_ZHIPU_API_KEY=test-zhipu-key
RAG_DEEPSEEK_API_KEY=test-deepseek-key
RAG_GRPC_AUTH_TOKEN=test-grpc-token
""".strip(),
        encoding="utf-8",
    )
    config_path.write_text(
        """
# empty on purpose: rely on built-in defaults
""".strip(),
        encoding="utf-8",
    )

    monkeypatch.setenv("RAG_ENV_FILE", str(env_file))
    monkeypatch.setenv("RAG_CONFIG_TOML", str(config_path))
    monkeypatch.delenv("RAG_ZHIPU_API_KEY", raising=False)
    monkeypatch.delenv("RAG_DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("RAG_GRPC_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("RAG_RERANKER_GRPC_AUTH_TOKEN", raising=False)

    settings = Settings.from_env()

    assert settings.model_cache_dir == (tmp_path / ".cache/models").resolve()
    assert settings.plugin_docs_vector_db_dir == (
        tmp_path / "data/plugin_docs_vector_db"
    ).resolve()
    assert settings.server_config_semantic_vector_db_dir == (
        tmp_path / "data/server_config_semantic_vector_db"
    ).resolve()
    assert settings.user_semantic_memory_db_path == (
        tmp_path / "data/user_semantic_memory.sqlite3"
    ).resolve()
    assert settings.server_instance_bindings_path == (
        tmp_path / "data/server_instance_bindings.json"
    ).resolve()
    assert settings.grpc_host == "127.0.0.1"
    assert settings.grpc_port == 50051
    assert settings.reranker_enabled is False
    assert settings.reranker_host == "127.0.0.1"
    assert settings.reranker_port == 50052
    assert settings.reranker_timeout_seconds == 10.0
    assert settings.reranker_auth_token is None
    assert settings.plugin_semantic_mc_servers_root == (tmp_path / "mc_servers").resolve()
    assert settings.plugin_semantic_agent_scan_on_startup is True
    assert settings.embedding_url == "https://open.bigmodel.cn/api/paas/v4/embeddings"
    assert settings.embedding_model == "embedding-3"
    assert settings.resolved_embedding_dimensions == 1024
    assert settings.retrieval_top_k == 5
    assert settings.answer_top_k == 4
    assert settings.plugin_docs_bm25_enabled is True
    assert settings.plugin_docs_bm25_top_k == 7
    assert settings.plugin_docs_bm25_auto_create_index is True


def test_settings_from_env_loads_zhipu_embedding_overrides(
    tmp_path, monkeypatch
):
    env_file = tmp_path / ".env"
    config_path = tmp_path / "config.toml"
    env_file.write_text(
        r"""
RAG_ZHIPU_API_KEY=test-zhipu-key
RAG_DEEPSEEK_API_KEY=test-deepseek-key
RAG_GRPC_AUTH_TOKEN=test-grpc-token
""".strip(),
        encoding="utf-8",
    )
    config_path.write_text(
        """
[embedding]
url = "https://embeddings.example.com/v1/embeddings"
model = "bge-m3"
""".strip(),
        encoding="utf-8",
    )

    monkeypatch.setenv("RAG_ENV_FILE", str(env_file))
    monkeypatch.setenv("RAG_CONFIG_TOML", str(config_path))
    monkeypatch.delenv("RAG_ZHIPU_API_KEY", raising=False)
    monkeypatch.delenv("RAG_DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("RAG_GRPC_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("RAG_RERANKER_GRPC_AUTH_TOKEN", raising=False)

    settings = Settings.from_env()

    assert settings.resolved_embedding_api_key == "test-zhipu-key"
    assert settings.resolved_embedding_api_key_env == "RAG_ZHIPU_API_KEY"
    assert settings.resolved_embedding_url == "https://embeddings.example.com/v1/embeddings"
    assert settings.resolved_embedding_model == "bge-m3"
    assert settings.resolved_embedding_dimensions == 1024


def test_runtime_validation_requires_reranker_token_when_enabled():
    settings = replace(
        make_settings(),
        expected_embedding_dimension=1024,
        reranker_enabled=True,
        reranker_auth_token=None,
    )

    with pytest.raises(ConfigurationError, match="RAG_RERANKER_GRPC_AUTH_TOKEN"):
        validate_runtime_settings(settings)


def test_build_embedding_client_uses_zhipu_openai_payload(monkeypatch):
    captured: dict[str, object] = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"data": [{"embedding": [0.5, 1.5]}]}

    def fake_post(url, *, headers, json, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(
        "agent_for_mc.infrastructure.clients.requests.post",
        fake_post,
    )

    settings = replace(
        make_settings(),
        embedding_api_key="embedding-secret",
        embedding_api_key_env="RAG_ZHIPU_API_KEY",
        embedding_url="https://embeddings.example.com/v1/embeddings",
        embedding_model="embedding-3",
        expected_embedding_dimension=1024,
    )

    client = build_embedding_client(settings)
    embedding = client.embed_query("hello world")

    assert isinstance(client, OpenAICompatibleEmbeddingClient)
    assert embedding == [0.5, 1.5]
    assert captured["url"] == "https://embeddings.example.com/v1/embeddings"
    assert captured["headers"] == {
        "Authorization": "Bearer embedding-secret",
        "Content-Type": "application/json",
    }
    assert captured["json"] == {
        "model": "embedding-3",
        "input": "hello world",
        "dimensions": 1024,
    }
    assert captured["timeout"] == 5


def test_validate_embedding_settings_rejects_unsupported_dimension():
    with pytest.raises(ConfigurationError, match="embedding\\.dimensions"):
        validate_embedding_settings(
            replace(
                make_settings(),
                expected_embedding_dimension=1536,
            )
        )


def test_extract_memory_candidates_filters_stable_preferences():
    candidates = extract_memory_candidates(
        "Please remember that I prefer concise Chinese answers and my project uses Paper.",
        "Sure.",
    )

    assert [candidate.kind for candidate in candidates] == ["preference", "fact"]
    assert candidates[0].content == "偏好：concise Chinese answers"
    assert candidates[0].confidence >= 0.95
    assert candidates[1].content == "事实：Paper"


def test_sqlite_memory_store_persists_and_recalls_by_keyword(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "user_semantic_memory.sqlite3", scope_id="test-user")
    store.save_candidates(
        [
            MemoryCandidate(
                kind="preference",
                content="偏好：回答时使用简洁中文",
                source_question="请记住我偏好简洁中文回答",
                source_answer="ok",
                confidence=0.97,
            ),
            MemoryCandidate(
                kind="fact",
                content="事实：当前项目使用 Paper",
                source_question="我们的项目用的是 Paper",
                source_answer="ok",
                confidence=0.9,
            ),
        ]
    )

    recalled = store.recall("请用简洁中文回答", limit=5)

    assert [record.kind for record in recalled][0] == "preference"
    assert recalled[0].content == "偏好：回答时使用简洁中文"


def test_memory_service_recall_context_formats_memories(tmp_path):
    settings = replace(
        make_settings(),
        memory_enabled=True,
        user_semantic_memory_db_path=tmp_path / "user_semantic_memory.sqlite3",
        memory_recall_limit=3,
        memory_min_confidence=0.75,
    )
    memory_service = build_memory_service(settings, scope_id="test-user")
    assert memory_service is not None
    memory_service.store.save_candidates(
        [
            MemoryCandidate(
                kind="preference",
                content="偏好：回答时使用简洁中文",
                source_question="请记住我偏好简洁中文回答",
                source_answer="ok",
                confidence=0.97,
            )
        ]
    )

    recalled = memory_service.recall("请继续用简洁中文回答", history_text="之前说过偏好")
    formatted = format_memory_context(recalled)

    assert "长期记忆" not in formatted
    assert "偏好：回答时使用简洁中文" in formatted


def test_build_memory_service_uses_supplied_maintenance_agent(tmp_path):
    class FakeMaintenanceAgent:
        pass

    settings = replace(
        make_settings(),
        memory_enabled=True,
        user_semantic_memory_db_path=tmp_path / "user_semantic_memory.sqlite3",
    )

    maintenance_agent = FakeMaintenanceAgent()
    memory_service = build_memory_service(
        settings,
        scope_id="test-user",
        maintenance_agent=maintenance_agent,
    )

    assert memory_service is not None
    assert memory_service.maintenance_runner.agent is maintenance_agent


def test_scan_plugin_semantic_bundles_reads_plugin_directory(tmp_path):
    mc_servers_root = tmp_path / "mc_servers"
    source_root = mc_servers_root / "[1]大厅服" / "plugins"
    plugin_dir = source_root / "MMORPG"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "config.yml").write_text("spawn-rate: 1\n", encoding="utf-8")
    (plugin_dir / "mob.yml").write_text("mob-level: 5\n", encoding="utf-8")

    bundles = scan_plugin_semantic_bundles(
        mc_servers_root,
        max_file_chars=1000,
        max_files_per_plugin=10,
    )

    assert len(bundles) == 1
    bundle = bundles[0]
    assert bundle.server_id == "[1]大厅服"
    assert bundle.plugin_name == "MMORPG"
    assert {file.relative_path for file in bundle.files} == {
        "config.yml",
        "mob.yml",
    }


def test_plugin_semantic_agent_service_refresh_writes_semantic_memory(tmp_path):
    mc_servers_root = tmp_path / "mc_servers"
    source_root = mc_servers_root / "[1]大厅服" / "plugins"
    plugin_dir = source_root / "MMORPG"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "config.yml").write_text("spawn-rate: 1\n", encoding="utf-8")
    (plugin_dir / "mob.yml").write_text("mob-level: 5\n", encoding="utf-8")

    captured: dict[str, object] = {}

    class FakeStore:
        def __init__(self):
            self.upserts: list[tuple[str, str, list[SemanticMemoryEntry], list[list[float]]]] = []
            self.deletes: list[tuple[str, str]] = []

        def upsert_bundle_entries(self, *, server_id, plugin_name, entries, embeddings):
            self.upserts.append((server_id, plugin_name, list(entries), list(embeddings)))
            captured["entries"] = list(entries)
            captured["embeddings"] = list(embeddings)

        def delete_bundle(self, *, server_id, plugin_name):
            self.deletes.append((server_id, plugin_name))

    class FakeEmbeddingClient:
        def embed_query(self, search_query: str):
            return [float(len(search_query) % 3), 1.0]

    class FakeMaintenanceRunner:
        def run(self, bundle):
            assert bundle.plugin_name == "MMORPG"
            return PluginSemanticExtractionResult(
                entries=[
                    SemanticMemoryEntry(
                        server_id=bundle.server_id,
                        plugin_name=bundle.plugin_name,
                        memory_type="plugin_config",
                        relation_type="located_in",
                        memory_text=(
                            "[1]大厅服 的 global 作用域下，MMORPG 的主配置位于 "
                            "plugins/MMORPG/config.yml。"
                        ),
                    )
                ]
            )

    service = PluginSemanticAgentService(
        store=FakeStore(),
        embedding_client=FakeEmbeddingClient(),
        maintenance_runner=FakeMaintenanceRunner(),
        mc_servers_root=str(mc_servers_root),
        manifest_path=tmp_path / "manifest.json",
        refresh_interval_seconds=0,
        max_file_chars=1000,
        max_files_per_plugin=10,
    )

    service.refresh()
    service.wait_for_idle(timeout=5)

    assert "entries" in captured
    entries = captured["entries"]
    embeddings = captured["embeddings"]
    assert len(entries) == 1
    assert len(embeddings) == 1
    assert entries[0].plugin_name == "MMORPG"
    assert entries[0].memory_text.startswith("[1]大厅服")


def test_plugin_semantic_agent_service_refresh_is_incremental(tmp_path):
    mc_servers_root = tmp_path / "mc_servers"
    first_plugin_dir = mc_servers_root / "[1]大厅服" / "plugins" / "MMORPG"
    first_plugin_dir.mkdir(parents=True)
    first_config = first_plugin_dir / "config.yml"
    first_config.write_text("spawn-rate: 1\n", encoding="utf-8")

    captured: dict[str, list[tuple[str, str]]] = {"upserts": [], "deletes": []}

    class FakeStore:
        def upsert_bundle_entries(self, *, server_id, plugin_name, entries, embeddings):
            captured["upserts"].append((server_id, plugin_name))

        def delete_bundle(self, *, server_id, plugin_name):
            captured["deletes"].append((server_id, plugin_name))

    class FakeEmbeddingClient:
        def embed_query(self, search_query: str):
            return [float(len(search_query) % 3), 1.0]

    class FakeMaintenanceRunner:
        def run(self, bundle):
            return PluginSemanticExtractionResult(
                entries=[
                    SemanticMemoryEntry(
                        server_id=bundle.server_id,
                        plugin_name=bundle.plugin_name,
                        memory_type="plugin_config",
                        relation_type="located_in",
                        memory_text=f"{bundle.server_id} / {bundle.plugin_name}",
                    )
                ]
            )

    service = PluginSemanticAgentService(
        store=FakeStore(),
        embedding_client=FakeEmbeddingClient(),
        maintenance_runner=FakeMaintenanceRunner(),
        mc_servers_root=str(mc_servers_root),
        manifest_path=tmp_path / "manifest.json",
        refresh_interval_seconds=0,
        max_file_chars=1000,
        max_files_per_plugin=10,
    )

    service.refresh()
    service.wait_for_idle(timeout=5)
    assert captured["upserts"] == [("[1]大厅服", "MMORPG")]
    assert captured["deletes"] == []

    service.refresh()
    service.wait_for_idle(timeout=5)
    assert captured["upserts"] == [("[1]大厅服", "MMORPG")]

    second_plugin_dir = mc_servers_root / "[2]生存服" / "plugins" / "AuthMe"
    second_plugin_dir.mkdir(parents=True)
    (second_plugin_dir / "config.yml").write_text("timeout: 30\n", encoding="utf-8")

    service.refresh()
    service.wait_for_idle(timeout=5)
    assert ("[2]生存服", "AuthMe") in captured["upserts"]


def test_plugin_semantic_agent_service_refresh_full_forces_rebuild(tmp_path):
    mc_servers_root = tmp_path / "mc_servers"
    plugin_dir = mc_servers_root / "[1]大厅服" / "plugins" / "MMORPG"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "config.yml").write_text("spawn-rate: 1\n", encoding="utf-8")

    captured: dict[str, list[tuple[str, str]]] = {"upserts": [], "deletes": []}

    class FakeStore:
        def upsert_bundle_entries(self, *, server_id, plugin_name, entries, embeddings):
            captured["upserts"].append((server_id, plugin_name))

        def delete_bundle(self, *, server_id, plugin_name):
            captured["deletes"].append((server_id, plugin_name))

    class FakeEmbeddingClient:
        def embed_query(self, search_query: str):
            return [float(len(search_query) % 3), 1.0]

    class FakeMaintenanceRunner:
        def run(self, bundle):
            return PluginSemanticExtractionResult(
                entries=[
                    SemanticMemoryEntry(
                        server_id=bundle.server_id,
                        plugin_name=bundle.plugin_name,
                        memory_type="plugin_config",
                        relation_type="located_in",
                        memory_text=f"{bundle.server_id} / {bundle.plugin_name}",
                    )
                ]
            )

    service = PluginSemanticAgentService(
        store=FakeStore(),
        embedding_client=FakeEmbeddingClient(),
        maintenance_runner=FakeMaintenanceRunner(),
        mc_servers_root=str(mc_servers_root),
        manifest_path=tmp_path / "manifest.json",
        refresh_interval_seconds=0,
        max_file_chars=1000,
        max_files_per_plugin=10,
    )

    service.refresh()
    service.wait_for_idle(timeout=5)
    service.refresh_full()
    service.wait_for_idle(timeout=5)

    assert captured["upserts"] == [("[1]大厅服", "MMORPG"), ("[1]大厅服", "MMORPG")]


def test_build_memory_maintenance_agent_uses_configured_model(monkeypatch):
    captured: dict[str, object] = {}

    class FakeChatDeepSeek:
        def __init__(self, *, model: str, api_key: str, api_base: str, temperature: float, timeout: int):
            captured["model"] = model
            captured["api_key"] = api_key
            captured["api_base"] = api_base
            captured["temperature"] = temperature
            captured["timeout"] = timeout

    def fake_create_deep_agent(**kwargs):
        captured["kwargs"] = kwargs
        return {"agent": "memory-maintenance"}

    monkeypatch.setattr(deepagent_memory_agent_module, "create_deep_agent", fake_create_deep_agent)
    monkeypatch.setattr(deepagent_memory_agent_module, "build_chat_model", lambda *, settings, model_name: FakeChatDeepSeek(
        model=model_name,
        api_key=settings.deepseek_api_key,
        api_base=settings.deepseek_api_base,
        temperature=0.1,
        timeout=settings.request_timeout_seconds,
    ))

    settings = replace(
        make_settings(),
        memory_maintenance_agent_model="deepseek-mini",
    )

    agent = deepagent_memory_agent_module.build_memory_maintenance_agent(settings=settings)

    assert agent == {"agent": "memory-maintenance"}
    assert captured["model"] == "deepseek-mini"
    assert (
        captured["kwargs"]["system_prompt"]
        == deepagent_memory_agent_module.MEMORY_MAINTENANCE_SYSTEM_PROMPT
    )


def test_retrieve_docs_tool_formats_output():
    vector_store = FakeVectorStore()
    retriever = Retriever(vector_store, FakeEmbeddingClient())
    context = RetrieveDocsToolContext(
        retriever=retriever,
        top_k=2,
        citation_preview_chars=20,
    )
    configure_retrieve_docs_tool(context)

    output = retrieve_docs.invoke({"search_query": "plugin"})

    assert "PluginA" in output
    assert "PluginB" in output
    assert "distance=" in output


class FakeRanker:
    def __init__(self):
        self.last_query = ""
        self.last_docs: list[RetrievedDoc] = []

    def rank_docs(self, query: str, docs: list[RetrievedDoc]) -> list[RetrievedDoc]:
        self.last_query = query
        self.last_docs = list(docs)
        return list(reversed(docs))


class FailingRanker:
    def rank_docs(self, query: str, docs: list[RetrievedDoc]) -> list[RetrievedDoc]:
        raise ServiceError("reranker unavailable")


class RecordingRerankerService(reranker_pb2_grpc.RerankerServiceServicer):
    def __init__(self):
        self.authorization = ""
        self.seen_query = ""
        self.seen_documents: list[tuple[int, str, str]] = []

    def Health(self, request, context):
        return reranker_pb2.HealthResponse(ready=True, model_name="fake")

    def Rerank(self, request, context):
        self.authorization = dict(context.invocation_metadata()).get("authorization", "")
        self.seen_query = request.query
        self.seen_documents = [
            (document.index, document.document_id, document.text)
            for document in request.documents
        ]
        return reranker_pb2.RerankResponse(
            request_id=request.request_id,
            results=[
                reranker_pb2.RankedDocument(index=1, document_id="2", score=0.9),
                reranker_pb2.RankedDocument(index=0, document_id="1", score=0.2),
            ],
        )


@contextmanager
def running_recording_reranker(service: RecordingRerankerService):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    reranker_pb2_grpc.add_RerankerServiceServicer_to_server(service, server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    try:
        yield port
    finally:
        server.stop(None)


def test_retriever_uses_ranker_when_configured():
    ranker = FakeRanker()
    retriever = Retriever(
        FakeVectorStore(),
        FakeEmbeddingClient(),
        ranker=ranker,
    )

    docs = retriever.retrieve("plugin", top_k=2)

    assert [doc.id for doc in docs] == [2, 1]
    assert ranker.last_query == "plugin"
    assert [doc.id for doc in ranker.last_docs] == [1, 2]


def test_grpc_reranker_client_sends_auth_and_orders_docs():
    service = RecordingRerankerService()
    docs = [
        RetrievedDoc(
            id=1,
            plugin_chinese_name="插件A",
            plugin_english_name="PluginA",
            content="first",
            distance=0.1,
        ),
        RetrievedDoc(
            id=2,
            plugin_chinese_name="插件B",
            plugin_english_name="PluginB",
            content="second",
            distance=0.2,
        ),
    ]

    with running_recording_reranker(service) as port:
        client = GrpcRerankerClient(
            host="127.0.0.1",
            port=port,
            auth_token="secret-token",
            timeout_seconds=5,
        )
        try:
            ranked_docs = client.rank_docs("plugin", docs)
        finally:
            client.close()

    assert [doc.id for doc in ranked_docs] == [2, 1]
    assert service.authorization == "Bearer secret-token"
    assert service.seen_query == "plugin"
    assert service.seen_documents[0][0] == 0
    assert service.seen_documents[0][1] == "1"
    assert "PluginA" in service.seen_documents[0][2]


def test_retriever_falls_back_when_ranker_fails():
    retriever = Retriever(
        FakeBM25VectorStore(),
        FakeEmbeddingClient(),
        ranker=FailingRanker(),
        bm25_enabled=True,
    )

    docs = retriever.retrieve("plugin", top_k=3)

    assert [doc.id for doc in docs] == [1, 2, 3]
    assert docs[2].match_reason == "bm25"


def test_retriever_adds_bm25_recall_path():
    vector_store = FakeBM25VectorStore()
    retriever = Retriever(
        vector_store,
        FakeEmbeddingClient(),
        bm25_enabled=True,
        bm25_top_k=5,
        bm25_auto_create_index=False,
    )

    docs = retriever.retrieve(" plugin config ", top_k=3)

    assert [doc.id for doc in docs] == [1, 2, 3]
    assert docs[2].match_reason == "bm25"
    assert vector_store.bm25_calls == [("plugin config", 5, False)]


def test_lance_plugin_vector_store_searches_bm25(tmp_path):
    db_dir = tmp_path / "plugin_docs_vector_db"
    db = lancedb.connect(str(db_dir))
    db.create_table(
        "plugin_docs",
        data=[
            {
                "id": 1,
                "content": "AuthMe login password session configuration",
                "plugin_chinese_name": "认证",
                "plugin_english_name": "AuthMe",
                "embedding": [0.0, 1.0],
            },
            {
                "id": 2,
                "content": "WorldGuard region flag protection",
                "plugin_chinese_name": "领地",
                "plugin_english_name": "WorldGuard",
                "embedding": [1.0, 0.0],
            },
        ],
    )
    store = LancePluginVectorStore(
        db_dir,
        "plugin_docs",
        expected_embedding_dimension=2,
    )

    docs = store.search_by_bm25("AuthMe password", top_k=2)

    assert docs
    assert docs[0].id == 1
    assert docs[0].match_reason == "bm25"
    assert docs[0].distance > 0


def test_build_session_uses_remote_reranker_client(monkeypatch):
    built_clients: list[Settings] = []
    fake_ranker = object()

    def fake_build_reranker_client(settings):
        built_clients.append(settings)
        return fake_ranker

    monkeypatch.setattr(
        session_factory_module,
        "build_reranker_client",
        fake_build_reranker_client,
    )

    settings = replace(
        make_settings(),
        reranker_enabled=True,
        reranker_auth_token="reranker-token",
    )

    session = session_factory_module.build_session(settings, memory_scope_id="test-user")

    assert isinstance(session, RagChatSession)
    assert built_clients == [settings]


def test_build_session_does_not_auto_refresh_plugin_semantic_service(monkeypatch):
    class FakePluginSemanticService:
        def __init__(self):
            self.refresh_calls = 0

        def refresh(self) -> bool:
            self.refresh_calls += 1
            return True

    class FakeDeepAgent:
        def invoke(self, payload):
            return {"messages": [AIMessage(content="ok")]}

    fake_service = FakePluginSemanticService()

    monkeypatch.setattr(
        session_factory_module,
        "build_memory_maintenance_agent",
        lambda *, settings: object(),
    )
    monkeypatch.setattr(session_factory_module, "build_memory_service", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        session_factory_module,
        "build_plugin_semantic_service",
        lambda settings: fake_service,
    )
    monkeypatch.setattr(session_factory_module, "build_deep_agent", lambda **kwargs: FakeDeepAgent())

    session = session_factory_module.build_session(
        make_settings(),
        memory_scope_id="test-user",
    )

    assert session.has_plugin_semantic_service() is True
    assert fake_service.refresh_calls == 0
    assert session.start_plugin_semantic_refresh() is True
    assert fake_service.refresh_calls == 1


def test_payload_helper_returns_docs_and_summary():
    vector_store = FakeVectorStore()
    retriever = Retriever(vector_store, FakeEmbeddingClient())
    docs, summary = build_retrieve_docs_payload(
        "plugin",
        context=RetrieveDocsToolContext(
            retriever=retriever,
            top_k=2,
            citation_preview_chars=20,
        ),
    )

    assert len(docs) == 2
    assert "PluginA" in summary


def test_plugin_config_retrieve_tool_returns_summary_and_paths():
    configure_plugin_config_tool(
        PluginConfigToolContext(
            retriever=FakeSemanticMemoryRetriever(),
            summarizer_client=FakePluginConfigSummaryClient(),
            top_k=2,
            preview_chars=18,
            summary_max_chars=120,
        )
    )

    output = retrieve_plugin_configs.invoke({"query": "How do I configure mob spawn?"})

    assert "MMORPG 的配置分散在 config.yml 和 mob.yml 中" in output
    assert "mmorpg/config.yml" in output
    assert "mmorpg/mob.yml" in output
    assert "Server config semantic memory:" in output


def test_build_deep_agent_registers_plugin_config_tool(monkeypatch):
    captured: dict[str, object] = {}

    def fake_create_deep_agent(**kwargs):
        captured["tools"] = kwargs["tools"]
        captured["subagents"] = kwargs.get("subagents", [])
        captured["system_prompt"] = kwargs["system_prompt"]

        class _Agent:
            def invoke(self, payload):
                return {"messages": [AIMessage(content="ok")]}

        return _Agent()

    monkeypatch.setattr(deepagent_main_agent_module, "create_deep_agent", fake_create_deep_agent)

    settings = make_settings()
    agent = build_deep_agent(
        settings=settings,
        retriever=Retriever(FakeVectorStore(), FakeEmbeddingClient()),
    )

    tool_names = [getattr(tool, "name", "") for tool in captured["tools"]]
    assert "route_plugin_config_request" in tool_names
    assert "retrieve_plugin_configs" not in tool_names
    assert "route_plugin_config_request" in str(captured["system_prompt"])
    assert "plugin_config_agent" in str(captured["system_prompt"])
    assert len(captured["subagents"]) == 1
    plugin_subagent = captured["subagents"][0]
    assert plugin_subagent["name"] == "plugin_config_agent"
    assert getattr(plugin_subagent["model"], "model_name", None) == settings.plugin_config_agent_model
    plugin_subagent_tool_names = [getattr(tool, "name", "") for tool in plugin_subagent["tools"]]
    assert plugin_subagent_tool_names == ["retrieve_plugin_configs"]
    assert agent is not None


def test_build_deep_agent_registers_plugin_semantic_refresh_tool(monkeypatch):
    captured: dict[str, object] = {}

    class FakePluginSemanticService:
        def request_refresh_status(self, *, full: bool = False) -> str:
            return "started"

    def fake_create_deep_agent(**kwargs):
        captured["tools"] = kwargs["tools"]
        captured["system_prompt"] = kwargs["system_prompt"]

        class _Agent:
            def invoke(self, payload):
                return {"messages": [AIMessage(content="ok")]}

        return _Agent()

    monkeypatch.setattr(deepagent_main_agent_module, "create_deep_agent", fake_create_deep_agent)

    settings = make_settings()
    build_deep_agent(
        settings=settings,
        retriever=Retriever(FakeVectorStore(), FakeEmbeddingClient()),
        plugin_semantic_service=FakePluginSemanticService(),
    )

    tool_names = [getattr(tool, "name", "") for tool in captured["tools"]]
    assert "refresh_plugin_semantic_memory" in tool_names
    assert "refresh_plugin_semantic_memory" in str(captured["system_prompt"])


def test_refresh_plugin_semantic_memory_tool_starts_incremental_refresh():
    class FakePluginSemanticService:
        def __init__(self):
            self.calls: list[bool] = []
            self.mc_servers_root = "F:/AgentForMc/mc_servers"

        def request_refresh_status(self, *, full: bool = False) -> str:
            self.calls.append(full)
            return "started"

    service = FakePluginSemanticService()
    configure_plugin_semantic_refresh_tool(
        PluginSemanticRefreshToolContext(service=service)
    )

    output = refresh_plugin_semantic_memory.invoke({})
    data = json.loads(output)

    assert service.calls == [False]
    assert data["status"] == "started"
    assert data["mode"] == "incremental"
    assert data["mc_servers_root"] == "F:/AgentForMc/mc_servers"


def test_refresh_plugin_semantic_memory_tool_reports_inflight_refresh():
    class FakePluginSemanticService:
        def __init__(self):
            self.mc_servers_root = "F:/AgentForMc/mc_servers"

        def request_refresh_status(self, *, full: bool = False) -> str:
            return "already_running"

    configure_plugin_semantic_refresh_tool(
        PluginSemanticRefreshToolContext(service=FakePluginSemanticService())
    )

    output = refresh_plugin_semantic_memory.invoke({})
    data = json.loads(output)

    assert data["status"] == "already_running"
    assert "already running" in data["message"]


def test_analyze_question_tool_returns_plan_json():
    configure_planning_tool(PlanningToolContext(client=FakePlanningClient()))

    output = analyze_question.invoke(
        {
            "question": "How to configure plugin?",
            "history": "Previous chat",
            "retrieval_summary": "No retrieval summary.",
        }
    )

    data = json.loads(output)
    assert data["standalone_query"] == "plugin config issue"
    assert data["need_multi_query"] is True
    assert data["need_plugins"] is True
    assert len(data["queries"]) == 3


def test_analyze_question_tool_context_survives_cross_thread_invocation():
    configure_planning_tool(PlanningToolContext(client=FakePlanningClient()))

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            analyze_question.invoke,
            {
                "question": "How to configure plugin?",
                "history": "Previous chat",
                "retrieval_summary": "No retrieval summary.",
            },
        )
        output = future.result(timeout=5)

    data = json.loads(output)
    assert data["standalone_query"] == "plugin config issue"
    assert data["need_multi_query"] is True


def test_route_plugin_config_request_returns_subagent_route():
    configure_plugin_config_routing_tool(
        PluginConfigRoutingToolContext(client=FakeRoutingClient("plugin_config_agent"))
    )

    output = route_plugin_config_request.invoke(
        {
            "question": "How should I set the plugin config defaults?",
            "history": "We talked about YAML files.",
        }
    )

    data = json.loads(output)
    assert data["route"] == "plugin_config_agent"
    assert data["use_subagent"] is True
    assert data["normalized_query"] == "How should I set the plugin config defaults?"
    assert "defaults" in data["reason"]


def test_retrieve_plugin_configs_tool_context_survives_cross_thread_invocation():
    configure_plugin_config_tool(
        PluginConfigToolContext(
            retriever=FakeSemanticMemoryRetriever(),
            summarizer_client=FakePluginConfigSummaryClient(),
            top_k=2,
            preview_chars=80,
            summary_max_chars=200,
        )
    )

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            retrieve_plugin_configs.invoke,
            {"search_query": "MMORPG mob config"},
        )
        output = future.result(timeout=5)

    assert "MMORPG" in output
    assert "config.yml" in output


def test_route_plugin_config_request_defaults_to_main_agent():
    configure_plugin_config_routing_tool(
        PluginConfigRoutingToolContext(client=FakeRoutingClient("main_agent"))
    )

    output = route_plugin_config_request.invoke(
        {
            "question": "How do I compare plugin behavior?",
            "history": "",
        }
    )

    data = json.loads(output)
    assert data["route"] == "main_agent"
    assert data["use_subagent"] is False
    assert data["normalized_query"] == "How do I compare plugin behavior?"


def test_query_rewrite_tool_returns_standalone_query():
    configure_query_rewrite_tool(QueryRewriteToolContext(client=FakeRewriteClient()))

    output = query_rewrite.invoke(
        {
            "question": "How to configure it?",
            "history": "We talked about AuthMe before.",
        }
    )

    assert output == "plugin config issue for authme"


def test_query_expansion_tool_returns_expanded_question_and_search_query():
    configure_query_expansion_tool(QueryExpansionToolContext(client=FakeExpansionClient()))

    output = query_expansion.invoke(
        {
            "question": "AuthMe config?",
            "history": "We talked about updating the plugin earlier.",
        }
    )

    assert "expanded_question:" in output
    assert "latest release" in output
    assert "search_query:" in output
    assert "AuthMe plugin configuration latest release update" in output


def test_select_retrieval_tool_routes_to_available_backend():
    configure_planning_tool(PlanningToolContext(client=FakeSelectorClient("chunk")))
    configure_select_retrieval_tool(
        SelectRetrievalToolContext(
            available_backends=("chunk",),
            default_backend="chunk",
        )
    )

    output = select_retrieval_tool.invoke(
        {
            "question": "AuthMe config?",
            "history": "We talked about updating the plugin earlier.",
        }
    )

    data = json.loads(output)
    assert data["retrieval_backend"] == "chunk"
    assert data["search_query"] == "AuthMe plugin configuration latest release"
    assert data["fallback_used"] is False


def test_select_retrieval_tool_falls_back_when_backend_unavailable():
    configure_planning_tool(PlanningToolContext(client=FakeSelectorClient("web")))
    configure_select_retrieval_tool(
        SelectRetrievalToolContext(
            available_backends=("chunk",),
            default_backend="chunk",
        )
    )

    output = select_retrieval_tool.invoke(
        {
            "question": "AuthMe config?",
            "history": "We talked about updating the plugin earlier.",
        }
    )

    data = json.loads(output)
    assert data["retrieval_backend"] == "chunk"
    assert data["fallback_used"] is True
    assert "unavailable" in data["reason"]


def test_multi_query_retrieve_docs_tool_formats_output():
    vector_store = FakeVectorStore()
    retriever = Retriever(vector_store, FakeEmbeddingClient())
    configure_retrieve_docs_tool(
        RetrieveDocsToolContext(
            retriever=retriever,
            top_k=2,
            citation_preview_chars=20,
        )
    )

    output = multi_query_retrieve_docs.invoke(
        {"queries": "plugin-a\nplugin-b"}
    )

    assert "PluginA" in output
    assert "PluginB" in output
    assert "distance=" in output


def test_multi_query_rag_tool_generates_and_merges_queries():
    configure_planning_tool(PlanningToolContext(client=FakeMultiQueryRagClient()))
    configure_multi_query_rag_tool(MultiQueryRagToolContext())
    vector_store = FakeVectorStore()
    retriever = Retriever(vector_store, FakeEmbeddingClient())
    configure_retrieve_docs_tool(
        RetrieveDocsToolContext(
            retriever=retriever,
            top_k=2,
            citation_preview_chars=20,
        )
    )

    output = multi_query_rag.invoke(
        {
            "question": "How to configure plugin?",
            "history": "Previous chat",
        }
    )

    assert "standalone_query:" in output
    assert "plugin config issue" in output
    assert "PluginA" in output
    assert "PluginB" in output


def test_subquery_decomposition_tool_splits_and_merges_subqueries():
    configure_planning_tool(PlanningToolContext(client=FakeSubQueryClient()))
    vector_store = FakeVectorStore()
    retriever = Retriever(vector_store, FakeEmbeddingClient())
    configure_retrieve_docs_tool(
        RetrieveDocsToolContext(
            retriever=retriever,
            top_k=2,
            citation_preview_chars=20,
        )
    )

    output = subquery_decomposition.invoke(
        {
            "question": "前端规范 + 新人第一周安排",
            "history": "Previous chat",
        }
    )

    assert "decomposition_question:" in output
    assert "前端规范" in output
    assert "新人第一周安排" in output
    assert "PluginA" in output
    assert "PluginB" in output


def test_judge_retrieval_freshness_tool_assesses_turn_docs():
    configure_planning_tool(PlanningToolContext(client=FakeJudgeClient()))
    configure_judge_retrieval_freshness_tool(JudgeRetrievalFreshnessToolContext())
    vector_store = FakeVectorStore()
    retriever = Retriever(vector_store, FakeEmbeddingClient())
    configure_retrieve_docs_tool(
        RetrieveDocsToolContext(
            retriever=retriever,
            top_k=2,
            citation_preview_chars=20,
        )
    )

    start_turn_context()
    try:
        retrieve_docs.invoke({"search_query": "plugin"})
        output = judge_retrieval_freshness.invoke(
            {
                "question": "Is this plugin config still current?",
                "retrieval_summary": "Two docs retrieved.",
            }
        )
    finally:
        clear_turn_context()

    data = json.loads(output)
    assert data["is_fresh_enough"] is False
    assert data["is_covered_enough"] is True
    assert data["needs_model_knowledge_fallback"] is True
    assert "version-agnostic" in data["reason"]


def test_server_plugins_tool_returns_turn_context_plugins():
    plugins = [
        "EssentialsX 2.20.1 (enabled)",
        "ViaVersion 5.2.1 (enabled)",
    ]

    start_turn_context()
    try:
        record_server_plugins(plugins)
        output = get_server_plugins_list.invoke({})
    finally:
        clear_turn_context()

    assert output == plugins


def test_judge_answer_quality_tool_scores_draft_answer():
    configure_planning_tool(PlanningToolContext(client=FakeAnswerJudgeClient()))
    configure_judge_answer_quality_tool(JudgeAnswerQualityToolContext())
    vector_store = FakeVectorStore()
    retriever = Retriever(vector_store, FakeEmbeddingClient())
    configure_retrieve_docs_tool(
        RetrieveDocsToolContext(
            retriever=retriever,
            top_k=2,
            citation_preview_chars=20,
        )
    )

    start_turn_context()
    try:
        retrieve_docs.invoke({"search_query": "plugin"})
        output = judge_answer_quality.invoke(
            {
                "question": "How should this plugin be configured?",
                "draft_answer": "It should be configured somehow.",
                "history": "Previous chat",
                "retrieval_summary": "Two docs retrieved.",
            }
        )
    finally:
        clear_turn_context()

    data = json.loads(output)
    assert data["overall_score"] == 0.42
    assert data["is_good_enough"] is False
    assert data["needs_retry"] is True
    assert data["retry_recommendation"] == "query_expansion"
    assert "version-specific" in data["reason"]


def test_hyde_retrieve_docs_tool_generates_hypothesis_and_merges_docs():
    configure_planning_tool(PlanningToolContext(client=FakeHydeClient()))
    configure_hyde_tool(HydeToolContext())
    vector_store = FakeVectorStore()
    retriever = Retriever(vector_store, FakeEmbeddingClient())
    configure_retrieve_docs_tool(
        RetrieveDocsToolContext(
            retriever=retriever,
            top_k=2,
            citation_preview_chars=20,
        )
    )

    output = hyde_retrieve_docs.invoke(
        {
            "question": "How should the plugin config be updated?",
            "history": "Previous chat",
        }
    )

    assert "hypothetical_answer:" in output
    assert "latest release" in output
    assert "PluginA" in output
    assert "PluginB" in output


def test_build_deep_agent_creates_agent_graph():
    settings = make_settings()
    retriever = Retriever(FakeVectorStore(), FakeEmbeddingClient())

    agent = build_deep_agent(settings=settings, retriever=retriever)

    assert agent is not None
    assert hasattr(agent, "invoke")


def test_rag_chat_session_uses_deep_agent():
    settings = make_settings()
    vector_store = FakeVectorStore()
    session = RagChatSession(
        settings=settings,
        vector_store=vector_store,
        deep_agent=FakeDeepAgent(),
    )

    result = session.ask("plugin question")

    assert isinstance(result, AnswerResult)
    assert result.answer == "deep answer"
    assert result.standalone_query == "plugin question"
    assert [doc.id for doc in result.citations] == [777]
    assert isinstance(session._history[-1], AIMessage)


def test_rag_chat_session_reports_coarse_progress_stages():
    progress: list[tuple[str, str]] = []
    session = RagChatSession(
        settings=make_settings(),
        vector_store=FakeVectorStore(),
        deep_agent=FakeDeepAgent(),
    )

    result = session.ask(
        "plugin question",
        progress_callback=lambda stage, message: progress.append((stage, message)),
    )

    assert result.answer == "deep answer"
    stages = [stage for stage, _ in progress]
    assert stages[0] == "started"
    assert "agent_running" in stages
    assert "answer_ready" in stages
    assert stages[-1] == "completed"


def test_retrieval_tool_reports_progress_stages():
    progress: list[tuple[str, str]] = []
    start_turn_context(progress_callback=lambda stage, message: progress.append((stage, message)))
    try:
        docs, formatted = build_retrieve_docs_payload(
            "homes",
            context=RetrieveDocsToolContext(
                retriever=Retriever(FakeVectorStore(), FakeEmbeddingClient()),
                top_k=2,
                citation_preview_chars=100,
            ),
        )
    finally:
        clear_turn_context()

    assert docs
    assert "PluginA" in formatted
    stages = [stage for stage, _ in progress]
    assert "retrieval" in stages
    assert "retrieval_ready" in stages


def test_rag_chat_session_exposes_server_plugins_to_tools():
    class PluginAwareDeepAgent:
        def __init__(self):
            self.plugins: list[str] = []

        def invoke(self, payload):
            self.plugins = get_server_plugins_list.invoke({})
            return {"messages": [AIMessage(content="deep answer")]}

    agent = PluginAwareDeepAgent()
    session = RagChatSession(
        settings=make_settings(),
        vector_store=FakeVectorStore(),
        deep_agent=agent,
    )

    result = session.ask(
        "which plugins are installed?",
        server_plugins=[
            "EssentialsX 2.20.1 (enabled)",
            "ViaVersion 5.2.1 (enabled)",
        ],
    )

    assert result.answer == "deep answer"
    assert agent.plugins == [
        "EssentialsX 2.20.1 (enabled)",
        "ViaVersion 5.2.1 (enabled)",
    ]


def test_rag_chat_session_recalls_and_writes_memory(tmp_path):
    settings = replace(
        make_settings(),
        memory_enabled=True,
        user_semantic_memory_db_path=tmp_path / "user_semantic_memory.sqlite3",
        memory_recall_limit=3,
        memory_min_confidence=0.75,
        memory_consolidation_turns=2,
    )

    class FakeMemoryMaintenanceAgent:
        def __init__(self):
            self.invocations: list[list[object]] = []

        def invoke(self, payload):
            messages = list(payload["messages"])
            self.invocations.append(messages)
            return {
                "messages": [
                    {
                        "content": json.dumps(
                            {
                                "session_summary": "用户偏好：回答保持简洁中文，项目使用 Paper。",
                                "actions": [
                                    {
                                        "action": "add",
                                        "type": "preference",
                                        "key": "answer_style",
                                        "value": "简洁中文回答",
                                        "confidence": 0.96,
                                    }
                                ],
                            },
                            ensure_ascii=False,
                        )
                    }
                ]
            }

    memory_service = build_memory_service(
        settings,
        scope_id="test-user",
        maintenance_agent=FakeMemoryMaintenanceAgent(),
    )
    assert memory_service is not None

    session = RagChatSession(
        settings=settings,
        vector_store=FakeVectorStore(),
        deep_agent=MemoryAwareDeepAgent(),
        memory_service=memory_service,
    )

    first_result = session.ask("Please remember that I prefer concise Chinese answers.")
    second_result = session.ask("My project uses Paper.")
    memory_service.wait_for_idle(timeout=5)
    third_result = session.ask("Please continue in concise Chinese.")
    memory_service.wait_for_idle(timeout=5)

    assert first_result.answer == "第一次回复，已记录。"
    assert second_result.answer == "第一次回复，已记录。"
    assert "收到记忆" in third_result.answer
    assert "简洁中文回答" in third_result.answer
    assert len(session._history) == 6

    third_invocation = session._deep_agent.invocations[-1]
    assert isinstance(third_invocation[0], SystemMessage)
    assert "长期记忆" in str(third_invocation[0].content)
    assert isinstance(third_invocation[1], HumanMessage)

    stored = memory_service.recall("简洁中文回答")
    assert stored
    assert stored[0].kind == "preference"
    assert stored[0].key == "answer_style"
    assert stored[0].value == "简洁中文回答"
    assert isinstance(session._history[-1], AIMessage)
    session.close()


def test_validate_memory_actions_enforces_rules():
    validated = validate_memory_actions(
        [
            MemoryAction(
                action="add",
                type="preference",
                key="answer_style",
                value="简洁中文回答",
                confidence=0.92,
            ),
            MemoryAction(
                action="add",
                type="preference",
                key="answer_style",
                value="简洁中文回答",
                confidence=0.91,
            ),
        ],
        [],
    )
    assert len(validated) == 1

    with pytest.raises(ValueError, match="非法的 memory type"):
        validate_memory_actions(
            [
                MemoryAction(
                    action="add",
                    type="unknown",
                    key="answer_style",
                    value="简洁中文回答",
                    confidence=0.92,
                )
            ],
            [],
        )

    with pytest.raises(ValueError, match="非法的 memory key"):
        validate_memory_actions(
            [
                MemoryAction(
                    action="add",
                    type="preference",
                    key="AnswerStyle",
                    value="简洁中文回答",
                    confidence=0.92,
                )
            ],
            [],
        )

    with pytest.raises(ValueError, match="update action 指向不存在的 memory_id"):
        validate_memory_actions(
            [
                MemoryAction(
                    action="update",
                    type="preference",
                    key="answer_style",
                    value="简洁中文回答",
                    confidence=0.92,
                    memory_id=999,
                )
            ],
            [],
        )


def test_sqlite_memory_store_deletes_and_migrates_legacy_rows(tmp_path):
    db_path = tmp_path / "user_semantic_memory.sqlite3"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE memory_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            kind TEXT NOT NULL,
            content TEXT NOT NULL,
            normalized_content TEXT NOT NULL UNIQUE,
            source_question TEXT NOT NULL,
            source_answer TEXT NOT NULL,
            confidence REAL NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            hit_count INTEGER NOT NULL DEFAULT 1
        )
        """
    )
    conn.execute(
        """
        INSERT INTO memory_items (
            kind, content, normalized_content, source_question, source_answer,
            confidence, created_at, updated_at, hit_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "preference",
            "偏好：简洁中文回答",
            "偏好：简洁中文回答",
            "remember",
            "ok",
            0.9,
            "2026-03-31T00:00:00+00:00",
            "2026-03-31T00:00:00+00:00",
            1,
        ),
    )
    conn.commit()
    conn.close()

    store = SQLiteMemoryStore(db_path, scope_id="test-user")
    store.initialize()
    recalled = store.recall("简洁中文回答", limit=5)
    assert recalled
    assert recalled[0].kind == "preference"
    assert recalled[0].value == "简洁中文回答"

    store.apply_actions(
        [
            MemoryAction(
                action="delete",
                type=recalled[0].kind,
                key=recalled[0].key,
                memory_id=recalled[0].id,
            )
        ],
        source_question="summary",
        source_answer="ledger",
    )
    assert store.recall("简洁中文回答", limit=5) == []


def test_memory_store_scopes_by_user(tmp_path):
    db_path = tmp_path / "user_semantic_memory.sqlite3"
    alice_store = SQLiteMemoryStore(db_path, scope_id="alice")
    bob_store = SQLiteMemoryStore(db_path, scope_id="bob")

    alice_store.apply_actions(
        [
            MemoryAction(
                action="add",
                type="preference",
                key="answer_style",
                value="简洁中文回答",
                confidence=0.95,
            )
        ],
        source_question="alice question",
        source_answer="alice answer",
    )
    bob_store.apply_actions(
        [
            MemoryAction(
                action="add",
                type="fact",
                key="project_stack",
                value="Paper",
                confidence=0.92,
            )
        ],
        source_question="bob question",
        source_answer="bob answer",
    )

    alice_recall = alice_store.recall("简洁中文回答", limit=5)
    bob_recall = bob_store.recall("Paper", limit=5)

    assert [(record.scope_id, record.key, record.value) for record in alice_recall] == [
        ("alice", "answer_style", "简洁中文回答")
    ]
    assert [(record.scope_id, record.key, record.value) for record in bob_recall] == [
        ("bob", "project_stack", "Paper")
    ]



