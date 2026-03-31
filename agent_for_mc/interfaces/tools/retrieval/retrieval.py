from __future__ import annotations

from langchain_core.tools import tool

from agent_for_mc.application.retrieval_tool import (
    RetrieveDocsToolContext,
    build_retrieve_docs_payload,
    configure_retrieve_docs_tool,
    get_retrieve_docs_tool_context,
)


@tool("retrieve_docs")
def retrieve_docs(search_query: str | None = None, query: str | None = None) -> str:
    """Retrieve the most relevant docs for a query and format them for an agent."""
    effective_search_query = search_query or query or ""
    docs, formatted_docs = build_retrieve_docs_payload(
        effective_search_query,
        context=get_retrieve_docs_tool_context(),
    )
    return formatted_docs
