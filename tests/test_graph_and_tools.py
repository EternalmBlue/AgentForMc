from __future__ import annotations

from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage

from agent_for_mc.application.chat_session import RagChatSession
from agent_for_mc.application.deepagent_state import record_retrieved_docs
from agent_for_mc.application.prompts import format_history
from agent_for_mc.application.retrieval import Retriever
from agent_for_mc.domain.models import AnswerResult, RetrievedDoc
from agent_for_mc.infrastructure.config import Settings
from agent_for_mc.interfaces.deepagent.build import build_deep_agent
from agent_for_mc.interfaces.tools.retrieval import (
    RetrieveDocsToolContext,
    build_retrieve_docs_payload,
    configure_retrieve_docs_tool,
    retrieve_docs,
)


class FakeVectorStore:
    def find_name_matches(self, question: str):
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
    def embed_query(self, text: str):
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
    )


def test_format_history_uses_base_messages():
    history = [
        HumanMessage(content="how to use plugin?"),
        AIMessage(content="use the config file"),
    ]

    text = format_history(history)

    assert "第1轮用户问题：how to use plugin?" in text
    assert "第1轮助手回答：use the config file" in text


def test_retrieve_docs_tool_formats_output():
    vector_store = FakeVectorStore()
    retriever = Retriever(vector_store, FakeEmbeddingClient())
    context = RetrieveDocsToolContext(
        retriever=retriever,
        top_k=2,
        citation_preview_chars=20,
    )
    configure_retrieve_docs_tool(context)

    output = retrieve_docs.invoke({"query": "plugin"})

    assert "PluginA" in output
    assert "PluginB" in output
    assert "distance=" in output


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
    assert result.rewritten_question == "plugin question"
    assert [doc.id for doc in result.citations] == [777]
    assert isinstance(session._history[-1], AIMessage)
