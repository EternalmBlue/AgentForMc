from __future__ import annotations

from langchain_core.tools import tool

from agent_for_mc.application.retrieval_tool import (
    RetrieveDocsToolContext,
    build_retrieve_docs_payload,
    configure_retrieve_docs_tool,
    get_retrieve_docs_tool_context,
)


@tool("retrieve_docs")
def retrieve_docs(query: str) -> str:
    """Retrieve the most relevant docs for a query and format them for an agent."""
    _, formatted_docs = build_retrieve_docs_payload(
        query,
        context=get_retrieve_docs_tool_context(),
    )
    return formatted_docs
