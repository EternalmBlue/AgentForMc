from __future__ import annotations

import json

from langchain_core.messages import BaseMessage

from agent_for_mc.application.prompts import format_history
from agent_for_mc.infrastructure.clients import DeepSeekChatClient


MULTI_QUERY_SYSTEM_PROMPT = """
你是一个多查询 RAG 路由器。
判断当前问题是否模糊、范围过大、信息缺失，或者是否适合用多个检索表达来提升召回。

如果适合多查询 RAG：
- 输出 need_multi_query 为 true
- 生成 3 到 4 条不同的检索表达
- 每条表达都尽量不同，覆盖同义改写、关键词压缩、上下文补全、不同检索角度

如果不适合：
- 输出 need_multi_query 为 false
- queries 为空数组

只输出 JSON，不要解释。
JSON 格式必须是 {"need_multi_query": true/false, "queries": []}。
"""


class MultiQueryPlanner:
    def __init__(self, client: DeepSeekChatClient):
        self._client = client

    def decide_and_generate_queries(
        self,
        history: list[BaseMessage],
        question: str,
        rewritten_question: str,
        retrieval_summary: str,
    ) -> tuple[bool, list[str]]:
        messages = [
            {"role": "system", "content": MULTI_QUERY_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"最近对话历史：\n{format_history(history)}\n\n"
                    f"用户原始问题：\n{question}\n\n"
                    f"检索问题：\n{rewritten_question}\n\n"
                    f"当前检索摘要：\n{retrieval_summary}\n"
                ),
            },
        ]
        content = self._client.chat(messages, temperature=0.0)
        return _parse_multi_query_decision(
            content,
            fallback_question=question,
            fallback_rewritten_question=rewritten_question,
        )


def _parse_multi_query_decision(
    content: str,
    *,
    fallback_question: str,
    fallback_rewritten_question: str,
) -> tuple[bool, list[str]]:
    normalized = content.strip()
    try:
        data = json.loads(normalized)
    except json.JSONDecodeError:
        lowered = normalized.lower()
        need_multi_query = lowered in {"true", "yes", "1", "need_multi_query=true"}
        return need_multi_query, []

    need_multi_query = _parse_bool(data.get("need_multi_query"))
    if not need_multi_query:
        return False, []

    raw_queries = data.get("queries") or data.get("multi_queries") or []
    if isinstance(raw_queries, str):
        raw_queries = [raw_queries]

    queries: list[str] = []
    if isinstance(raw_queries, list):
        queries = _normalize_queries(
            [str(query).strip() for query in raw_queries if str(query).strip()],
            fallback_question=fallback_question,
            fallback_rewritten_question=fallback_rewritten_question,
        )

    if len(queries) < 3:
        queries = _normalize_queries(
            queries,
            fallback_question=fallback_question,
            fallback_rewritten_question=fallback_rewritten_question,
        )

    return True, queries[:4]


def _parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1"}
    return bool(value)


def _normalize_queries(
    queries: list[str],
    *,
    fallback_question: str,
    fallback_rewritten_question: str,
) -> list[str]:
    unique_queries: list[str] = []
    seen: set[str] = set()

    for query in queries:
        normalized = " ".join(query.split()).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique_queries.append(normalized)

    for candidate in (
        fallback_rewritten_question,
        fallback_question,
        f"{fallback_question} 具体有哪些相关内容",
        f"{fallback_question} 相关插件/功能/配置",
    ):
        normalized = " ".join(candidate.split()).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique_queries.append(normalized)
        if len(unique_queries) >= 4:
            break

    return unique_queries
