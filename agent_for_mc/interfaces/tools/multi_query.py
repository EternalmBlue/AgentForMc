from __future__ import annotations

from langchain_core.tools import tool

from agent_for_mc.application.retrieval_tool import (
    build_multi_query_retrieve_docs_payload,
    format_docs_for_tool,
    get_retrieve_docs_tool_context,
)


@tool("multi_query_retrieve_docs")
def multi_query_retrieve_docs(queries: str) -> str:
    """Retrieve relevant docs for multiple queries and format them for an agent."""
    context = get_retrieve_docs_tool_context()
    normalized_queries: list[str] = []
    for line in queries.replace("\n", ",").replace(";", ",").split(","):
        item = line.strip()
        if item:
            normalized_queries.append(item)

    docs, summary = build_multi_query_retrieve_docs_payload(
        normalized_queries,
        context=context,
    )
    formatted_docs = format_docs_for_tool(
        docs,
        preview_chars=context.citation_preview_chars,
    )
    return f"{summary}\n\n{formatted_docs}"
