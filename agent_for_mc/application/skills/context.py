from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from agent_for_mc.domain.models import RetrievedDoc, SemanticMemoryDoc
from agent_for_mc.application.skills.web_research import (
    WebResearchProvider,
    WebResearchResult,
)


class PluginDocsRetriever(Protocol):
    def retrieve(self, search_query: str, *, top_k: int = 8) -> list[RetrievedDoc]:
        ...


class ServerConfigRetriever(Protocol):
    def retrieve(
        self,
        search_query: str,
        *,
        top_k: int = 8,
        server_id: str | None = None,
        plugin_name: str | None = None,
    ) -> list[SemanticMemoryDoc]:
        ...


class SkillInventory(Protocol):
    def list_skills(self, server_id: str | None = None) -> list[object]:
        ...


@dataclass(frozen=True, slots=True)
class SkillAuthoringContextSource:
    kind: str
    title: str
    content: str


@dataclass(frozen=True, slots=True)
class SkillAuthoringContext:
    query: str
    sources: tuple[SkillAuthoringContextSource, ...] = ()
    diagnostics: tuple[str, ...] = ()

    @property
    def has_content(self) -> bool:
        return bool(self.sources or self.diagnostics)

    def render(self) -> str:
        if not self.has_content:
            return ""

        parts = [
            "Skill authoring context for this Minecraft server.",
            "Use this context as supporting evidence. Do not invent facts not supported by the user's answers or these sources.",
            "If a source is incomplete, ask a follow-up question instead of guessing.",
            f"Search query: {self.query or '<empty>'}",
        ]
        for index, source in enumerate(self.sources, start=1):
            parts.append(
                "\n".join(
                    [
                        f"[Context {index}: {source.kind}] {source.title}",
                        source.content.strip(),
                    ]
                )
            )
        if self.diagnostics:
            parts.append(
                "Context diagnostics:\n"
                + "\n".join(f"- {diagnostic}" for diagnostic in self.diagnostics)
            )
        return "\n\n".join(part for part in parts if part.strip())


@dataclass(slots=True)
class SkillAuthoringContextBuilder:
    skill_inventory: SkillInventory
    plugin_docs_retriever: PluginDocsRetriever | None = None
    server_config_retriever: ServerConfigRetriever | None = None
    web_research_provider: WebResearchProvider | None = None
    plugin_docs_top_k: int = 4
    server_config_top_k: int = 6
    web_research_top_k: int = 5
    preview_chars: int = 500
    max_context_chars: int = 8000

    def build(
        self,
        *,
        server_id: str,
        messages: list[str],
    ) -> SkillAuthoringContext:
        query = _build_context_query(messages)
        sources: list[SkillAuthoringContextSource] = []
        diagnostics: list[str] = []

        self._add_existing_skills(
            server_id=server_id,
            sources=sources,
            diagnostics=diagnostics,
        )
        self._add_plugin_docs(
            query=query,
            sources=sources,
            diagnostics=diagnostics,
        )
        self._add_server_config(
            server_id=server_id,
            query=query,
            sources=sources,
            diagnostics=diagnostics,
        )
        self._add_web_research(
            query=query,
            sources=sources,
            diagnostics=diagnostics,
        )

        return SkillAuthoringContext(
            query=query,
            sources=tuple(_trim_sources(sources, max_chars=self.max_context_chars)),
            diagnostics=tuple(diagnostics),
        )

    def _add_existing_skills(
        self,
        *,
        server_id: str,
        sources: list[SkillAuthoringContextSource],
        diagnostics: list[str],
    ) -> None:
        try:
            skills = self.skill_inventory.list_skills(server_id)
        except Exception as exc:
            diagnostics.append(f"existing skill inventory unavailable: {exc}")
            return

        lines: list[str] = []
        for skill in skills:
            name = str(getattr(skill, "name", "")).strip()
            description = str(getattr(skill, "description", "")).strip()
            scope = getattr(getattr(skill, "scope", ""), "value", getattr(skill, "scope", ""))
            valid = bool(getattr(skill, "valid", False))
            usage = str(getattr(skill, "usage", "")).strip() or "qa"
            if not name:
                continue
            status = "valid" if valid else "invalid"
            lines.append(
                f"- [{scope or 'unknown'}] {name}: {description or '<no description>'} "
                f"(usage={usage}, {status})"
            )

        if lines:
            sources.append(
                SkillAuthoringContextSource(
                    kind="existing_skills",
                    title="Installed official/global/server skills",
                    content="\n".join(lines[:20]),
                )
            )

    def _add_plugin_docs(
        self,
        *,
        query: str,
        sources: list[SkillAuthoringContextSource],
        diagnostics: list[str],
    ) -> None:
        if self.plugin_docs_retriever is None or not query:
            return
        try:
            docs = self.plugin_docs_retriever.retrieve(
                query,
                top_k=max(1, int(self.plugin_docs_top_k)),
            )
        except Exception as exc:
            diagnostics.append(f"plugin docs retrieval unavailable: {exc}")
            return

        if not docs:
            diagnostics.append("plugin docs retrieval returned no matching documents")
            return
        sources.append(
            SkillAuthoringContextSource(
                kind="plugin_docs",
                title="Relevant plugin documentation",
                content=_format_plugin_docs(docs, preview_chars=self.preview_chars),
            )
        )

    def _add_server_config(
        self,
        *,
        server_id: str,
        query: str,
        sources: list[SkillAuthoringContextSource],
        diagnostics: list[str],
    ) -> None:
        if self.server_config_retriever is None or not query:
            return
        try:
            docs = self.server_config_retriever.retrieve(
                query,
                top_k=max(1, int(self.server_config_top_k)),
                server_id=server_id,
            )
        except Exception as exc:
            diagnostics.append(f"server config semantic retrieval unavailable: {exc}")
            return

        if not docs:
            diagnostics.append("server config semantic retrieval returned no matching memories")
            return
        sources.append(
            SkillAuthoringContextSource(
                kind="server_config_semantic_memory",
                title="Relevant uploaded server configuration memories",
                content=_format_server_config_docs(docs, preview_chars=self.preview_chars),
            )
        )

    def _add_web_research(
        self,
        *,
        query: str,
        sources: list[SkillAuthoringContextSource],
        diagnostics: list[str],
    ) -> None:
        if self.web_research_provider is None or not query:
            return
        try:
            response = self.web_research_provider.search(
                query,
                top_k=max(1, int(self.web_research_top_k)),
            )
        except Exception as exc:
            diagnostics.append(f"web research unavailable: {exc}")
            return

        diagnostics.extend(response.diagnostics)
        if not response.results:
            if not response.diagnostics:
                diagnostics.append("web research returned no matching results")
            return
        sources.append(
            SkillAuthoringContextSource(
                kind="web_research",
                title="External web search results",
                content=_format_web_research_results(
                    list(response.results),
                    preview_chars=self.preview_chars,
                ),
            )
        )


def _build_context_query(messages: list[str]) -> str:
    normalized = [
        _strip_authoring_message_prefix(" ".join(str(message or "").strip().split()))
        for message in messages
        if str(message or "").strip()
    ]
    normalized = [message for message in normalized if message]
    if not normalized:
        return ""
    return " ".join(normalized)[-1600:]


def _strip_authoring_message_prefix(message: str) -> str:
    for prefix in ("Initial requirement:", "User answer:"):
        if message.casefold().startswith(prefix.casefold()):
            return message[len(prefix) :].strip()
    return message


def _format_plugin_docs(docs: list[RetrievedDoc], *, preview_chars: int) -> str:
    parts: list[str] = []
    for index, doc in enumerate(docs, start=1):
        plugin_label = doc.plugin_english_name or doc.plugin_chinese_name or "unknown"
        preview = _preview(doc.content, preview_chars=preview_chars)
        parts.append(
            "\n".join(
                [
                    f"[{index}] plugin={plugin_label}",
                    f"match_reason={doc.match_reason}",
                    f"distance={doc.distance:.6f}",
                    f"preview={preview}",
                ]
            )
        )
    return "\n\n".join(parts)


def _format_server_config_docs(
    docs: list[SemanticMemoryDoc],
    *,
    preview_chars: int,
) -> str:
    parts: list[str] = []
    for index, doc in enumerate(docs, start=1):
        parts.append(
            "\n".join(
                [
                    f"[{index}] server={doc.server_id} plugin={doc.plugin_name}",
                    f"type={doc.memory_type}",
                    f"relation={doc.relation_type}",
                    f"match_reason={doc.match_reason}",
                    f"distance={doc.distance:.6f}",
                    f"preview={_preview(doc.memory_text, preview_chars=preview_chars)}",
                ]
            )
        )
    return "\n\n".join(parts)


def _format_web_research_results(
    results: list[WebResearchResult],
    *,
    preview_chars: int,
) -> str:
    parts = [
        "External public web results. Treat these as lower priority than the server owner's answers and server-local RAG evidence."
    ]
    for index, result in enumerate(results, start=1):
        lines = [
            f"[{index}] title={result.title}",
            f"url={result.url}",
            f"site={result.site_name or '<unknown>'}",
        ]
        if result.published_time:
            lines.append(f"published_time={result.published_time}")
        if result.icon:
            lines.append(f"icon={result.icon}")
        lines.append(f"summary={_preview(result.summary, preview_chars=preview_chars)}")
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


def _trim_sources(
    sources: list[SkillAuthoringContextSource],
    *,
    max_chars: int,
) -> list[SkillAuthoringContextSource]:
    if max_chars <= 0:
        return sources

    trimmed: list[SkillAuthoringContextSource] = []
    remaining = max_chars
    for source in sources:
        if remaining <= 0:
            break
        content = source.content
        if len(content) > remaining:
            content = content[:remaining].rstrip() + "\n[trimmed]"
        trimmed.append(
            SkillAuthoringContextSource(
                kind=source.kind,
                title=source.title,
                content=content,
            )
        )
        remaining -= len(content)
    return trimmed


def _preview(value: str, *, preview_chars: int) -> str:
    text = " ".join(str(value or "").strip().split())
    if preview_chars > 0 and len(text) > preview_chars:
        return text[:preview_chars].rstrip() + "..."
    return text
