from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages

from agent_for_mc.domain.models import RetrievedDoc


class RagGraphState(TypedDict, total=False):
    uuid: str
    messages: Annotated[list[BaseMessage], add_messages]
    need_plugins: bool
    server_plugins: list[str]
    need_multi_query: bool
    multi_query_variants: list[str]
    origin_question: str
    rewritten_question: str
    retrieved_docs: list[RetrievedDoc]
    retrieval_summary: str
    retry_count: int
    max_retries: int
    citations: list[RetrievedDoc]
    answer: str
