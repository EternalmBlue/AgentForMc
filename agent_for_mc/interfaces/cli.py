from __future__ import annotations

from agent_for_mc.application.chat_session import RagChatSession
from agent_for_mc.application.memory_service import build_memory_service
from agent_for_mc.application.retrieval import Retriever
from agent_for_mc.domain.errors import ConfigurationError, RagForMcError, StartupValidationError
from agent_for_mc.domain.models import AnswerResult
from agent_for_mc.infrastructure.clients import JinaEmbeddingClient
from agent_for_mc.infrastructure.config import Settings
from agent_for_mc.infrastructure.ranker import BceRanker
from agent_for_mc.infrastructure.vector_store import LancePluginVectorStore
from agent_for_mc.interfaces.deepagent.build import (
    build_deep_agent,
    build_memory_maintenance_agent,
)


def build_session(settings: Settings, *, memory_scope_id: str) -> RagChatSession:
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
    memory_maintenance_agent = build_memory_maintenance_agent(settings=settings)
    memory_service = build_memory_service(
        settings,
        scope_id=memory_scope_id,
        maintenance_agent=memory_maintenance_agent,
    )
    deep_agent = build_deep_agent(
        settings=settings,
        retriever=retriever,
        ranker=ranker,
    )
    return RagChatSession(
        settings=settings,
        vector_store=vector_store,
        deep_agent=deep_agent,
        memory_service=memory_service,
    )


def main() -> int:
    settings = Settings.from_env()
    try:
        try:
            _validate_configuration(settings)
        except (ConfigurationError, StartupValidationError) as exc:
            print(f"[startup error] {exc}")
            return 1

        memory_scope_id = _prompt_memory_scope_id()
        session = build_session(settings, memory_scope_id=memory_scope_id)
        stats = session.startup_validate()

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
    finally:
        if "session" in locals():
            session.close()


def _validate_configuration(settings: Settings) -> None:
    if not settings.jina_api_key:
        raise ConfigurationError("缺少环境变量 RAG_JINA_API_KEY。")
    if not settings.deepseek_api_key:
        raise ConfigurationError("缺少环境变量 RAG_DEEPSEEK_API_KEY。")
    if settings.memory_enabled and settings.memory_consolidation_turns < 1:
        raise ConfigurationError("memory.consolidation_turns 必须大于 0。")


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


def _prompt_memory_scope_id() -> str:
    while True:
        try:
            scope_id = input("请输入记忆作用域ID（用户标识）: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。")
            raise SystemExit(0)

        if not scope_id:
            print("记忆作用域ID 不能为空。")
            continue
        return scope_id
