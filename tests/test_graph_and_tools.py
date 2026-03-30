from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
import json

from langchain_core.messages import AIMessage, HumanMessage

from agent_for_mc.application.chat_session import RagChatSession
from agent_for_mc.application.deepagent_state import (
    clear_turn_context,
    record_retrieved_docs,
    start_turn_context,
)
from agent_for_mc.application.prompts import format_history
from agent_for_mc.application.retrieval import Retriever
from agent_for_mc.domain.models import AnswerResult, RetrievedDoc
from agent_for_mc.infrastructure.config import Settings
from agent_for_mc.interfaces import cli as cli_module
from agent_for_mc.interfaces.deepagent.build import build_deep_agent
from agent_for_mc.interfaces.tools.multi_query import multi_query_retrieve_docs
from agent_for_mc.interfaces.tools.multi_query_rag import (
    MultiQueryRagToolContext,
    configure_multi_query_rag_tool,
    multi_query_rag,
)
from agent_for_mc.interfaces.tools.planning import (
    PlanningToolContext,
    analyze_question,
    configure_planning_tool,
)
from agent_for_mc.interfaces.tools.judge_retrieval_freshness import (
    JudgeRetrievalFreshnessToolContext,
    configure_judge_retrieval_freshness_tool,
    judge_retrieval_freshness,
)
from agent_for_mc.interfaces.tools.judge_answer_quality import (
    JudgeAnswerQualityToolContext,
    configure_judge_answer_quality_tool,
    judge_answer_quality,
)
from agent_for_mc.interfaces.tools.hyde import (
    HydeToolContext,
    configure_hyde_tool,
    hyde_retrieve_docs,
)
from agent_for_mc.interfaces.tools.query_rewrite import (
    QueryRewriteToolContext,
    configure_query_rewrite_tool,
    query_rewrite,
)
from agent_for_mc.interfaces.tools.query_expansion import (
    QueryExpansionToolContext,
    configure_query_expansion_tool,
    query_expansion,
)
from agent_for_mc.interfaces.tools.select_retrieval_tool import (
    SelectRetrievalToolContext,
    configure_select_retrieval_tool,
    select_retrieval_tool,
)
from agent_for_mc.interfaces.tools.subquery_decomposition import subquery_decomposition
from agent_for_mc.interfaces.tools.retrieval import (
    RetrieveDocsToolContext,
    build_retrieve_docs_payload,
    configure_retrieve_docs_tool,
    retrieve_docs,
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


class FakeEmbeddingClient:
    def embed_query(self, search_query: str):
        return [0.0, 1.0]


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


def make_settings() -> Settings:
    return Settings(
        lance_db_dir=Path("data/plugins_vector_db"),
        lance_table_name="plugins_docs",
        jina_api_key="test",
        jina_embeddings_url="https://example.com/jina",
        jina_embeddings_model="test-model",
        jina_embeddings_task="retrieval.query",
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
RAG_CONFIG_TOML=.\config.toml
RAG_MODEL_CACHE_DIR=.\.cache\models
RAG_LANCE_DB_DIR=.\data\plugins_vector_db
RAG_JINA_API_KEY=test-jina-key
RAG_DEEPSEEK_API_KEY=test-deepseek-key
""".strip(),
        encoding="utf-8",
    )
    config_path.write_text(
        """
[paths]
lance_db_dir = "data/plugins_vector_db"
model_cache_dir = ".cache/models"

[vector_store]
lance_table_name = "plugins_docs"
expected_embedding_dimension = 1024
rewrite_history_turns = 4
retrieval_top_k = 8
answer_top_k = 4
citation_preview_chars = 200

[reranker]
enabled = true
model_name_or_path = "maidalun1020/bce-reranker-base_v1"
""".strip(),
        encoding="utf-8",
    )

    monkeypatch.setenv("RAG_ENV_FILE", str(env_file))
    monkeypatch.delenv("RAG_CONFIG_TOML", raising=False)
    monkeypatch.delenv("RAG_MODEL_CACHE_DIR", raising=False)
    monkeypatch.delenv("RAG_LANCE_DB_DIR", raising=False)
    monkeypatch.delenv("RAG_JINA_API_KEY", raising=False)
    monkeypatch.delenv("RAG_DEEPSEEK_API_KEY", raising=False)

    settings = Settings.from_env()

    assert settings.lance_db_dir == (tmp_path / "data/plugins_vector_db").resolve()
    assert settings.model_cache_dir == (tmp_path / ".cache/models").resolve()
    assert settings.jina_api_key == "test-jina-key"
    assert settings.deepseek_api_key == "test-deepseek-key"
    assert settings.reranker_enabled is True
    assert settings.reranker_model_name_or_path == "maidalun1020/bce-reranker-base_v1"
    assert os.environ["HF_HOME"] == str(settings.model_cache_dir)
    assert os.environ["TRANSFORMERS_CACHE"] == str(
        settings.model_cache_dir / "transformers"
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


class FakeReranker:
    def __init__(self):
        self.last_query = ""
        self.last_docs: list[RetrievedDoc] = []

    def rerank_docs(self, query: str, docs: list[RetrievedDoc]) -> list[RetrievedDoc]:
        self.last_query = query
        self.last_docs = list(docs)
        return list(reversed(docs))


def test_retriever_uses_reranker_when_configured():
    reranker = FakeReranker()
    retriever = Retriever(
        FakeVectorStore(),
        FakeEmbeddingClient(),
        reranker=reranker,
    )

    docs = retriever.retrieve("plugin", top_k=2)

    assert [doc.id for doc in docs] == [2, 1]
    assert reranker.last_query == "plugin"
    assert [doc.id for doc in reranker.last_docs] == [1, 2]


def test_build_session_warmups_reranker(monkeypatch):
    warmed: list[str] = []

    class FakeBceReranker:
        def __init__(self, model_name_or_path: str):
            self.model_name_or_path = model_name_or_path

        def warmup(self) -> None:
            warmed.append(self.model_name_or_path)

    monkeypatch.setattr(cli_module, "BceReranker", FakeBceReranker)

    settings = replace(make_settings(), reranker_enabled=True)

    session = cli_module.build_session(settings)

    assert isinstance(session, RagChatSession)
    assert warmed == ["maidalun1020/bce-reranker-base_v1"]


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
