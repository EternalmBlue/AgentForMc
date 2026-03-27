from __future__ import annotations

from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage

from agent_for_mc.application.chains import build_rag_chain
from agent_for_mc.application.retrieval import Retriever
from agent_for_mc.domain.models import AnswerResult, RetrievedDoc
from agent_for_mc.infrastructure.config import Settings
from agent_for_mc.interfaces.langgraph.build import build_app
from agent_for_mc.interfaces.tools.retrieval import (
    RetrieveDocsToolContext,
    build_retrieve_docs_payload,
    configure_retrieve_docs_tool,
    retrieve_docs,
)
from agent_for_mc.application.prompts import format_history


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


class FakeRewriter:
    def rewrite_question(self, history, question: str):
        return f"rewritten: {question}"


class FakeMultiQueryPlanner:
    def __init__(self, need_multi_query: bool, queries=None):
        self._need_multi_query = need_multi_query
        self._queries = list(queries or [])

    def decide_and_generate_queries(
        self,
        history,
        question: str,
        rewritten_question: str,
        retrieval_summary: str,
    ):
        return self._need_multi_query, list(self._queries)


class FakePluginDecisionMaker:
    def __init__(self, need_plugins: bool):
        self._need_plugins = need_plugins

    def decide_need_plugins(
        self,
        history,
        question: str,
        rewritten_question: str,
        retrieval_summary: str,
    ) -> bool:
        return self._need_plugins


class FakeAnswerGenerator:
    def answer(
        self,
        history,
        question,
        rewritten_question: str,
        docs,
        server_plugins=None,
    ):
        return AnswerResult(
            answer=f"answer:{rewritten_question}",
            citations=list(docs),
            rewritten_question=rewritten_question,
        )


class FakeMultiQueryRetriever:
    def retrieve(self, question: str, *, top_k: int = 8):
        question_lower = question.lower()
        shared = RetrievedDoc(
            id=900,
            plugin_chinese_name="共享插件",
            plugin_english_name="SharedPlugin",
            content=f"shared content for {question}",
            distance=0.3,
            match_reason="shared",
        )
        if "variant-a" in question_lower:
            return [
                RetrievedDoc(
                    id=901,
                    plugin_chinese_name="变体A",
                    plugin_english_name="VariantA",
                    content="variant A content",
                    distance=0.1,
                    match_reason="variant-a",
                ),
                shared,
            ]
        if "variant-b" in question_lower:
            return [
                RetrievedDoc(
                    id=902,
                    plugin_chinese_name="变体B",
                    plugin_english_name="VariantB",
                    content="variant B content",
                    distance=0.2,
                    match_reason="variant-b",
                ),
                shared,
            ]
        if "variant-c" in question_lower:
            return [
                RetrievedDoc(
                    id=903,
                    plugin_chinese_name="变体C",
                    plugin_english_name="VariantC",
                    content="variant C content",
                    distance=0.25,
                    match_reason="variant-c",
                ),
                shared,
            ]
        return [
            RetrievedDoc(
                id=1,
                plugin_chinese_name="基础插件",
                plugin_english_name="BasePlugin",
                content="base content",
                distance=0.05,
                match_reason="base",
            ),
        ]


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
        deepseek_chat_url="https://example.com/deepseek",
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


def test_graph_build_invokes_plugin_fetch_when_needed():
    settings = make_settings()
    vector_store = FakeVectorStore()
    retriever = Retriever(vector_store, FakeEmbeddingClient())
    graph_app = build_app(
        rewriter=FakeRewriter(),
        retriever=retriever,
        answer_generator=FakeAnswerGenerator(),
        multi_query_planner=FakeMultiQueryPlanner(False),
        plugin_decider=FakePluginDecisionMaker(True),
        top_k=settings.retrieval_top_k,
        answer_top_k=settings.answer_top_k,
        citation_preview_chars=settings.citation_preview_chars,
    )

    state = graph_app.invoke(
        {
            "messages": [HumanMessage(content="previous question")],
            "origin_question": "plugin question",
        }
    )

    assert state["rewritten_question"] == "rewritten: plugin question"
    assert state["answer"] == "answer:rewritten: plugin question"
    assert len(state["citations"]) == 2
    assert state["server_plugins"] == ["authme", "viaversion"]
    assert state["need_plugins"] is True
    assert state["retrieval_summary"]
    assert isinstance(state["messages"][-1], AIMessage)


def test_graph_build_skips_plugin_fetch_when_not_needed():
    settings = make_settings()
    vector_store = FakeVectorStore()
    retriever = Retriever(vector_store, FakeEmbeddingClient())
    graph_app = build_app(
        rewriter=FakeRewriter(),
        retriever=retriever,
        answer_generator=FakeAnswerGenerator(),
        multi_query_planner=FakeMultiQueryPlanner(False),
        plugin_decider=FakePluginDecisionMaker(False),
        top_k=settings.retrieval_top_k,
        answer_top_k=settings.answer_top_k,
        citation_preview_chars=settings.citation_preview_chars,
    )

    state = graph_app.invoke(
        {
            "messages": [HumanMessage(content="previous question")],
            "origin_question": "plugin question",
        }
    )

    assert state["rewritten_question"] == "rewritten: plugin question"
    assert state["answer"] == "answer:rewritten: plugin question"
    assert len(state["citations"]) == 2
    assert state["need_plugins"] is False
    assert "server_plugins" not in state
    assert isinstance(state["messages"][-1], AIMessage)


def test_graph_build_runs_multi_query_rag_and_dedupes_docs():
    settings = make_settings()
    retriever = FakeMultiQueryRetriever()
    graph_app = build_app(
        rewriter=FakeRewriter(),
        retriever=retriever,
        answer_generator=FakeAnswerGenerator(),
        multi_query_planner=FakeMultiQueryPlanner(
            True,
            queries=[
                "variant-a question",
                "variant-b question",
                "variant-c question",
            ],
        ),
        plugin_decider=FakePluginDecisionMaker(False),
        top_k=settings.retrieval_top_k,
        answer_top_k=settings.answer_top_k,
        citation_preview_chars=settings.citation_preview_chars,
    )

    state = graph_app.invoke(
        {
            "messages": [HumanMessage(content="previous question")],
            "origin_question": "ambiguous plugin question",
        }
    )

    assert state["need_multi_query"] is True
    assert state["multi_query_variants"] == [
        "variant-a question",
        "variant-b question",
        "variant-c question",
    ]
    assert "variant-a question" in state["retrieval_summary"]
    assert "variant-b question" in state["retrieval_summary"]
    assert "variant-c question" in state["retrieval_summary"]
    assert len(state["retrieved_docs"]) == 5
    assert {doc.id for doc in state["retrieved_docs"]} == {1, 900, 901, 902, 903}
    assert state["answer"] == "answer:rewritten: ambiguous plugin question"
    assert isinstance(state["messages"][-1], AIMessage)


def test_rag_chain_runs_end_to_end():
    settings = make_settings()
    vector_store = FakeVectorStore()
    retriever = Retriever(vector_store, FakeEmbeddingClient())
    chain = build_rag_chain(
        rewriter=FakeRewriter(),
        retriever=retriever,
        answer_generator=FakeAnswerGenerator(),
        top_k=settings.retrieval_top_k,
        answer_top_k=settings.answer_top_k,
        citation_preview_chars=settings.citation_preview_chars,
    )

    state = chain.invoke(
        {
            "messages": [HumanMessage(content="previous question")],
            "origin_question": "plugin question",
        }
    )

    assert state["rewritten_question"] == "rewritten: plugin question"
    assert state["answer"] == "answer:rewritten: plugin question"
    assert len(state["citations"]) == 2
    assert isinstance(state["messages"][-1], AIMessage)
