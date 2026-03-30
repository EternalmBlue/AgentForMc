from __future__ import annotations

from agent_for_mc.application.chat_session import RagChatSession
from agent_for_mc.domain.errors import ConfigurationError, RagForMcError, StartupValidationError
from agent_for_mc.domain.models import AnswerResult
from agent_for_mc.infrastructure.clients import JinaEmbeddingClient
from agent_for_mc.infrastructure.config import Settings
from agent_for_mc.infrastructure.ranker import BceRanker
from agent_for_mc.infrastructure.vector_store import LancePluginVectorStore
from agent_for_mc.interfaces.deepagent.build import build_deep_agent
from agent_for_mc.application.retrieval import Retriever


def build_session(settings: Settings) -> RagChatSession:
    vector_store = LancePluginVectorStore(
        settings.lance_db_dir,
        settings.lance_table_name,
        expected_embedding_dimension=settings.expected_embedding_dimension,
    )
    embedding_client = JinaEmbeddingClient(settings)
    ranker = (
        BceRanker(settings.reranker_model_name_or_path)
        if settings.reranker_enabled
        else None
    )
    if ranker is not None:
        ranker.warmup()
    retriever = Retriever(vector_store, embedding_client, ranker=ranker)
    deep_agent = build_deep_agent(
        settings=settings,
        retriever=retriever,
    )
    return RagChatSession(
        settings=settings,
        vector_store=vector_store,
        deep_agent=deep_agent,
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

    print("DeepAgent local RAG CLI")
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
