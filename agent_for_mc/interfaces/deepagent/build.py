from __future__ import annotations

from deepagents import create_deep_agent
from langchain_deepseek import ChatDeepSeek

from agent_for_mc.infrastructure.clients import DeepSeekChatClient
from agent_for_mc.application.retrieval import Retriever
from agent_for_mc.infrastructure.config import Settings
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
from agent_for_mc.interfaces.tools.server_plugins import get_server_plugins_list


DEEPAGENT_SYSTEM_PROMPT = """You are a Minecraft plugin assistant backed by a local RAG index.

Instructions:
- When deciding which backend to use, call `select_retrieval_tool` first.
- Use `query_expansion` for short or underspecified questions.
- Use `query_rewrite` when you need a standalone search query or pronoun resolution.
- Use `subquery_decomposition` for requests that mix multiple independent tasks or knowledge areas.
- Use `multi_query_rag` for broad or ambiguous questions that benefit from wider recall.
- Use `hyde_retrieve_docs` when direct retrieval is semantically weak and a hypothetical answer may help.
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
- Answer in concise Chinese.
- Treat retrieved docs as primary evidence.
- Do not invent plugin names, versions, APIs, or dependencies that are not supported by the evidence.
- If you rely on general knowledge beyond the retrieved docs, say so explicitly.
"""


def build_deep_agent(
    *,
    settings: Settings,
    retriever: Retriever,
) -> object | None:
    if not settings.deepseek_api_key:
        return None

    planning_client = DeepSeekChatClient(settings)
    configure_select_retrieval_tool(
        SelectRetrievalToolContext(
            available_backends=("chunk",),
            default_backend="chunk",
        )
    )
    configure_query_expansion_tool(QueryExpansionToolContext(client=planning_client))
    configure_query_rewrite_tool(QueryRewriteToolContext(client=planning_client))
    configure_multi_query_rag_tool(MultiQueryRagToolContext())
    configure_hyde_tool(HydeToolContext())
    configure_planning_tool(PlanningToolContext(client=planning_client))
    configure_judge_retrieval_freshness_tool(JudgeRetrievalFreshnessToolContext())
    configure_judge_answer_quality_tool(JudgeAnswerQualityToolContext())
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

    return create_deep_agent(
        model=model,
        tools=[
            select_retrieval_tool,
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
        system_prompt=DEEPAGENT_SYSTEM_PROMPT,
    )
