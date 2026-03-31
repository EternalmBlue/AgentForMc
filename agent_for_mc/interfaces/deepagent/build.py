from __future__ import annotations

from deepagents import create_deep_agent
from deepagents.middleware.subagents import SubAgent
from langchain_deepseek import ChatDeepSeek

from agent_for_mc.infrastructure.clients import DeepSeekChatClient, JinaEmbeddingClient
from agent_for_mc.application.retrieval import Retriever
from agent_for_mc.infrastructure.config import Settings
from agent_for_mc.infrastructure.plugin_config_vector_store import (
    LancePluginConfigVectorStore,
)
from agent_for_mc.infrastructure.ranker import BceRanker
from agent_for_mc.application.plugin_config_retrieval import PluginConfigRetriever
from agent_for_mc.interfaces.tools.multi_query import multi_query_retrieve_docs
from agent_for_mc.interfaces.tools.multi_query_rag import (
    MultiQueryRagToolContext,
    configure_multi_query_rag_tool,
    multi_query_rag,
)
from agent_for_mc.interfaces.tools.subquery_decomposition import (
    subquery_decomposition,
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
from agent_for_mc.interfaces.tools.retrieval import (
    RetrieveDocsToolContext,
    configure_retrieve_docs_tool,
    retrieve_docs,
)
from agent_for_mc.interfaces.tools.plugin_configs import (
    PluginConfigToolContext,
    configure_plugin_config_tool,
    retrieve_plugin_configs,
)
from agent_for_mc.interfaces.tools.plugin_config_routing import (
    PluginConfigRoutingToolContext,
    configure_plugin_config_routing_tool,
    route_plugin_config_request,
)
from agent_for_mc.interfaces.tools.server_plugins import get_server_plugins_list


DEEPAGENT_SYSTEM_PROMPT = """You are a Minecraft plugin assistant backed by a local RAG index.

Instructions:
- When deciding which backend to use, call `select_retrieval_tool` first.
- Use `query_expansion` for short or underspecified questions.
- Use `query_rewrite` when you need a standalone search query or pronoun resolution.
- Use `subquery_decomposition` for requests that mix multiple independent tasks or knowledge areas.
- Use `multi_query_rag` for broad or ambiguous questions that benefit from wider recall.
- Use `hyde_retrieve_docs` when direct retrieval is semantically weak and a hypothetical answer may help.
- For questions about plugin config files, defaults, file-specific settings, dependency wiring,
  or config-file differences, call `route_plugin_config_request` first.
- If the routing result points to `plugin_config_agent`, call the `task` tool with
  `subagent_type="plugin_config_agent"` and pass the routed query plus the relevant context.
- Do not call `retrieve_plugin_configs` directly from the main agent.
- Prefer composing primitive tools over monolithic workflow tools.
- Use `analyze_question` when you need a compact routing plan for multi-query and plugin checks.
- If `need_multi_query` is true, call `multi_query_retrieve_docs` with the planned queries.
- Otherwise call `retrieve_docs` with the standalone query.
- If retrieved docs look stale or incomplete, call `judge_retrieval_freshness`.
- After drafting an answer, call `judge_answer_quality` to score the answer as a whole.
- If `judge_answer_quality` returns `needs_retry=true` or `overall_score < 0.8`, inspect `retry_recommendation` and call the recommended tool before retrieving again.
- Use `query_rewrite` when the recommendation is `query_rewrite` or when the issue is pronoun resolution or a missing standalone query.
- Use `query_expansion` when the recommendation is `query_expansion` or when the issue is a short or underspecified question.
- Use `subquery_decomposition` when the recommendation is `subquery_decomposition` or when the answer misses one of several independent subtopics.
- Use `multi_query_rag` when the recommendation is `multi_query_rag` or when broader recall is needed.
- Use `hyde_retrieve_docs` when the recommendation is `hyde_retrieve_docs` or when direct retrieval is semantically weak.
- If the retrieval-freshness judge says fallback is needed, answer cautiously using model knowledge and say it is a supplement.
- Use `get_server_plugins_list` when `need_plugins` is true or when plugin availability changes the answer.
- If a long-term memory context is provided, treat it as stable user preference or project background, but still prioritize retrieved docs when they conflict.
- Answer in concise Chinese.
- Treat retrieved docs as primary evidence.
- Do not invent plugin names, versions, APIs, or dependencies that are not supported by the evidence.
- If you rely on general knowledge beyond the retrieved docs, say so explicitly.
"""


PLUGIN_CONFIG_AGENT_SYSTEM_PROMPT = """You are plugin_config_agent.

Your job is to answer questions about plugin configuration files, defaults, file paths,
dependency wiring, and config-file differences.

Use `retrieve_plugin_configs` as your primary and only retrieval tool.
Return concise Chinese answers with the key configuration evidence and file paths.
If the evidence is incomplete, say so directly and do not invent missing values.
"""


MEMORY_MAINTENANCE_SYSTEM_PROMPT = """You are memory_maintenance_agent.

You maintain long-term memory for a Minecraft plugin assistant.
Given a session transcript and the current memory snapshot, produce JSON only with:
{"session_summary": string, "actions": array}

Rules:
- session_summary should be concise and stable.
- actions may contain add, update, or delete entries for stable preferences, goals, constraints, and facts.
- Follow the same memory key rules as the memory subsystem:
  type must be one of preference, goal, constraint, fact.
  key must be snake_case.
  update/delete must reference an existing memory_id when applicable.
- Do not include extra prose or markdown.
""".strip()


def build_deep_agent(
    *,
    settings: Settings,
    retriever: Retriever,
    ranker: BceRanker | None = None,
) -> object | None:
    if not settings.deepseek_api_key:
        return None

    planning_client = DeepSeekChatClient(settings)
    embedding_client = JinaEmbeddingClient(settings)
    configure_select_retrieval_tool(
        SelectRetrievalToolContext(
            available_backends=("chunk",),
            default_backend="chunk",
        )
    )
    configure_plugin_config_routing_tool(
        PluginConfigRoutingToolContext(client=planning_client)
    )
    configure_query_expansion_tool(QueryExpansionToolContext(client=planning_client))
    configure_query_rewrite_tool(QueryRewriteToolContext(client=planning_client))
    configure_multi_query_rag_tool(MultiQueryRagToolContext())
    configure_hyde_tool(HydeToolContext())
    configure_planning_tool(PlanningToolContext(client=planning_client))
    configure_judge_retrieval_freshness_tool(JudgeRetrievalFreshnessToolContext())
    configure_judge_answer_quality_tool(JudgeAnswerQualityToolContext())
    plugin_config_vector_store = LancePluginConfigVectorStore(
        settings.plugin_config_db_dir,
        settings.plugin_config_table_name,
        expected_embedding_dimension=settings.expected_embedding_dimension,
    )
    configure_plugin_config_tool(
        PluginConfigToolContext(
            retriever=PluginConfigRetriever(
                plugin_config_vector_store,
                embedding_client,
                ranker=ranker,
            ),
            summarizer_client=planning_client,
            top_k=settings.plugin_config_top_k,
            preview_chars=settings.plugin_config_preview_chars,
            summary_max_chars=settings.plugin_config_summary_chars,
        )
    )
    configure_retrieve_docs_tool(
        RetrieveDocsToolContext(
            retriever=retriever,
            top_k=settings.retrieval_top_k,
            citation_preview_chars=settings.citation_preview_chars,
        )
    )

    model = ChatDeepSeek(
        model=settings.deepseek_model,
        api_key=settings.deepseek_api_key,
        api_base=settings.deepseek_api_base,
        temperature=0.1,
        timeout=settings.request_timeout_seconds,
    )
    plugin_config_model = ChatDeepSeek(
        model=settings.plugin_config_agent_model,
        api_key=settings.deepseek_api_key,
        api_base=settings.deepseek_api_base,
        temperature=0.1,
        timeout=settings.request_timeout_seconds,
    )
    plugin_config_agent: SubAgent = {
        "name": "plugin_config_agent",
        "description": (
            "Use this agent for plugin config files, defaults, file paths, dependency wiring, "
            "and config-file differences."
        ),
        "system_prompt": PLUGIN_CONFIG_AGENT_SYSTEM_PROMPT,
        "tools": [retrieve_plugin_configs],
        "model": plugin_config_model,
    }

    return create_deep_agent(
        model=model,
        tools=[
            select_retrieval_tool,
            route_plugin_config_request,
            query_expansion,
            query_rewrite,
            subquery_decomposition,
            multi_query_rag,
            hyde_retrieve_docs,
            analyze_question,
            judge_retrieval_freshness,
            judge_answer_quality,
            retrieve_docs,
            multi_query_retrieve_docs,
            get_server_plugins_list,
        ],
        subagents=[plugin_config_agent],
        system_prompt=DEEPAGENT_SYSTEM_PROMPT,
    )


def build_memory_maintenance_agent(*, settings: Settings) -> object | None:
    if not settings.deepseek_api_key:
        return None

    model = ChatDeepSeek(
        model=settings.memory_maintenance_agent_model,
        api_key=settings.deepseek_api_key,
        api_base=settings.deepseek_api_base,
        temperature=0.1,
        timeout=settings.request_timeout_seconds,
    )
    return create_deep_agent(
        model=model,
        system_prompt=MEMORY_MAINTENANCE_SYSTEM_PROMPT,
    )
