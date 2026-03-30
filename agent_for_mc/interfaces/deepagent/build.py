from __future__ import annotations

from typing import Any

from deepagents import create_deep_agent
from langchain_deepseek import ChatDeepSeek

from agent_for_mc.application.retrieval import Retriever
from agent_for_mc.infrastructure.config import Settings
from agent_for_mc.interfaces.tools.retrieval import (
    RetrieveDocsToolContext,
    configure_retrieve_docs_tool,
    retrieve_docs,
)
from agent_for_mc.interfaces.tools.server_plugins import get_server_plugins_list


DEEPAGENT_SYSTEM_PROMPT = """You are a Minecraft plugin assistant backed by a local RAG index.

Instructions:
- Prefer calling `retrieve_docs` before answering.
- Use `get_server_plugins_list` when the user asks about installed server plugins or when plugin availability changes the answer.
- Answer in concise Chinese.
- Treat retrieved docs as primary evidence.
- Do not invent plugin names, versions, APIs, or dependencies that are not supported by the evidence.
- If you rely on general knowledge beyond the retrieved docs, say so explicitly.
"""


def build_deep_agent(
    *,
    settings: Settings,
    retriever: Retriever,
) -> Any | None:
    if not settings.deepseek_api_key:
        return None

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
        tools=[retrieve_docs, get_server_plugins_list],
        system_prompt=DEEPAGENT_SYSTEM_PROMPT,
    )
