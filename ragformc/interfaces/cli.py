from __future__ import annotations

from ragformc.application.chat_session import RagChatSession
from ragformc.application.chains import build_rag_chain
from ragformc.application.generation import AnswerGenerator
from ragformc.application.multi_query import MultiQueryPlanner
from ragformc.application.plugin_decision import PluginDecisionMaker
from ragformc.application.retrieval import Retriever
from ragformc.application.rewrite import QuestionRewriter
from ragformc.domain.errors import ConfigurationError, RagForMcError, StartupValidationError
from ragformc.domain.models import AnswerResult
from ragformc.infrastructure.clients import DeepSeekChatClient, JinaEmbeddingClient
from ragformc.infrastructure.config import Settings
from ragformc.infrastructure.vector_store import LancePluginVectorStore
from ragformc.interfaces.langgraph.build import build_app


def build_session(settings: Settings) -> RagChatSession:
    vector_store = LancePluginVectorStore(
        settings.lance_db_dir,
        settings.lance_table_name,
        expected_embedding_dimension=settings.expected_embedding_dimension,
    )
    deepseek_client = DeepSeekChatClient(settings)
    embedding_client = JinaEmbeddingClient(settings)
    rewriter = QuestionRewriter(deepseek_client)
    retriever = Retriever(vector_store, embedding_client)
    answer_generator = AnswerGenerator(deepseek_client)
    multi_query_planner = MultiQueryPlanner(deepseek_client)
    plugin_decider = PluginDecisionMaker(deepseek_client)
    rag_chain = build_rag_chain(
        rewriter=rewriter,
        retriever=retriever,
        answer_generator=answer_generator,
        top_k=settings.retrieval_top_k,
        answer_top_k=settings.answer_top_k,
        citation_preview_chars=settings.citation_preview_chars,
    )
    graph_app = build_app(
        rewriter=rewriter,
        retriever=retriever,
        answer_generator=answer_generator,
        multi_query_planner=multi_query_planner,
        plugin_decider=plugin_decider,
        top_k=settings.retrieval_top_k,
        answer_top_k=settings.answer_top_k,
        citation_preview_chars=settings.citation_preview_chars,
    )
    return RagChatSession(
        settings=settings,
        vector_store=vector_store,
        rewriter=rewriter,
        retriever=retriever,
        answer_generator=answer_generator,
        rag_chain=rag_chain,
        graph_app=graph_app,
    )


def main() -> int:
    settings = Settings.from_env()
    session = build_session(settings)

    try:
        _validate_configuration(settings)
        stats = session.startup_validate()
    except (ConfigurationError, StartupValidationError) as exc:
        print(f"[startup error] {exc}")
        return 1

    print("RagForMc local RAG CLI")
    print(
        "Vector DB: "
        f"table={stats.table_name}, records={stats.record_count}, "
        f"embedding_dim={stats.embedding_dimension}"
    )
    print("Commands: exit | clear")

    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。")
            return 0

        if not question:
            continue
        lowered_question = question.lower()
        if lowered_question in {"exit", "quit"}:
            print("已退出。")
            return 0
        if lowered_question == "clear":
            session.clear_history()
            print("会话历史已清空。")
            continue

        try:
            result = session.ask(question)
        except RagForMcError as exc:
            print(f"[query error] {exc}")
            continue

        _print_answer_result(result)


def _validate_configuration(settings: Settings) -> None:
    if not settings.jina_api_key:
        raise ConfigurationError("缺少环境变量 RAG_JINA_API_KEY。")
    if not settings.deepseek_api_key:
        raise ConfigurationError("缺少环境变量 RAG_DEEPSEEK_API_KEY。")


def _print_answer_result(result: AnswerResult) -> None:
    print(f"\n回答：\n{result.answer}")

    print("\n引用摘要：")
    if not result.citations:
        print("- 无命中文档。")
        return

    for citation in result.citations:
        preview = citation.content.replace("\n", " ").strip()
        preview = preview[:200]
        print(
            "- "
            f"{citation.plugin_chinese_name} / "
            f"{citation.plugin_english_name} / "
            f"distance={citation.distance:.6f} / "
            f"match={citation.match_reason} / "
            f"preview={preview}"
        )
