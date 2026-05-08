from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event, Lock
from time import sleep

import pytest
from langchain_core.messages import AIMessage, SystemMessage

from agent_for_mc.application.chat_session import RagChatSession
from agent_for_mc.application.skills import (
    SkillAuthoringContextBuilder,
    SkillAuthoringService,
    SkillNotFoundError,
    SkillReadonlyError,
    SkillRegistry,
    SkillScope,
    SkillValidationError,
    WebResearchResponse,
    WebResearchResult,
)
from agent_for_mc.application.skills.service import render_skill_markdown
from agent_for_mc.domain.models import RetrievedDoc, SemanticMemoryDoc


class FakeSettings:
    rewrite_history_turns = 4


class FakeVectorStore:
    def validate(self):
        return None


class CapturingAgent:
    def __init__(self):
        self.payloads = []

    def invoke(self, payload):
        self.payloads.append(payload)
        return {"messages": [AIMessage(content="answer")]}


class FakeAuthoringClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.seen_messages = []

    def chat(self, messages, *, temperature: float = 0.1):
        self.seen_messages.append(messages)
        if not self.responses:
            raise AssertionError("unexpected chat call")
        response = self.responses.pop(0)
        if isinstance(response, str):
            return response
        return json.dumps(response, ensure_ascii=False)


def test_skill_registry_scans_three_scopes_and_marks_conflicts(tmp_path: Path):
    official_dir = tmp_path / "official"
    global_dir = tmp_path / "global"
    mc_root = tmp_path / "mc_servers"
    _write_skill(
        official_dir,
        "skill-creator",
        description="Use only during skill creation.",
        body="# Skill Creator\n\nAuthoring only.",
        usage="authoring",
    )
    _write_skill(
        global_dir,
        "permission-debugging",
        description="Use for LuckPerms permission denied questions.",
        body="# Permission Debugging\n\nCheck groups and explicit deny rules.",
    )
    _write_skill(
        mc_root / "lobby-1" / "skills",
        "permission-debugging",
        description="Duplicate server skill.",
        body="# Duplicate\n\nThis should be invalid because global already owns the name.",
    )
    _write_skill(
        mc_root / "lobby-1" / "skills",
        "economy-debugging",
        description="Use for Vault economy balance issues.",
        body="# Economy Debugging\n\nCheck Vault, economy plugin, and command context.",
    )

    registry = SkillRegistry(
        official_skills_dir=official_dir,
        global_skills_dir=global_dir,
        mc_servers_root=mc_root,
    )

    skills = registry.list_skills("lobby-1")
    by_name = {(skill.scope, skill.name): skill for skill in skills}

    assert by_name[(SkillScope.OFFICIAL, "skill-creator")].valid is True
    assert by_name[(SkillScope.GLOBAL, "permission-debugging")].valid is True
    duplicate = by_name[(SkillScope.SERVER, "permission-debugging")]
    assert duplicate.valid is False
    assert "conflicts" in duplicate.diagnostics[0]
    assert by_name[(SkillScope.SERVER, "economy-debugging")].deletable is True


def test_skill_registry_rejects_extra_files_and_selects_relevant_server_skill(
    tmp_path: Path,
):
    official_dir = tmp_path / "official"
    global_dir = tmp_path / "global"
    mc_root = tmp_path / "mc_servers"
    _write_skill(
        mc_root / "lobby-1" / "skills",
        "economy-debugging",
        description="Use for Vault economy balance issues.",
        body="# Economy Debugging\n\nCheck Vault and balance commands.",
    )
    extra_dir = mc_root / "lobby-1" / "skills" / "unsafe-skill"
    extra_dir.mkdir(parents=True)
    (extra_dir / "SKILL.md").write_text(
        render_skill_markdown(
            name="unsafe-skill",
            description="Use for unsafe resources.",
            body="# Unsafe\n\nBad.",
        ),
        encoding="utf-8",
    )
    (extra_dir / "script.py").write_text("print('no')\n", encoding="utf-8")

    registry = SkillRegistry(
        official_skills_dir=official_dir,
        global_skills_dir=global_dir,
        mc_servers_root=mc_root,
    )

    selected = registry.select_skills(
        server_id="lobby-1",
        question="玩家 Vault 余额不对，economy balance 怎么排查？",
    )
    invalid = registry.get_skill(server_id="lobby-1", skill_name="unsafe-skill")

    assert [skill.name for skill in selected] == ["economy-debugging"]
    assert invalid.valid is False
    assert "SKILL.md" in invalid.diagnostics[0]


def test_skill_authoring_creates_preview_and_installs_server_skill(tmp_path: Path):
    registry = SkillRegistry(
        official_skills_dir=tmp_path / "official",
        global_skills_dir=tmp_path / "global",
        mc_servers_root=tmp_path / "mc_servers",
    )
    service = SkillAuthoringService(
        registry=registry,
        client=FakeAuthoringClient(
            [
                {
                    "status": "needs_clarification",
                    "questions": ["Which plugin should this cover?"],
                },
                {
                    "status": "draft_ready",
                    "skill": {
                        "name": "luckperms-debugging",
                        "description": "Use for LuckPerms permission denied issues.",
                        "body": "# LuckPerms Debugging\n\nCheck group inheritance and explicit deny rules.",
                    },
                },
            ]
        ),
    )

    first = service.start(
        server_id="lobby-1",
        initial_requirement="帮我做一个权限排查 skill",
    )
    second = service.continue_draft(
        server_id="lobby-1",
        draft_id=first.draft_id,
        user_message="主要处理 LuckPerms 权限拒绝",
    )
    installed = service.confirm(server_id="lobby-1", draft_id=first.draft_id)

    assert first.status == "needs_clarification"
    assert first.questions == ("Which plugin should this cover?",)
    assert second.status == "draft_ready"
    assert second.skill is not None
    assert second.skill.valid is True
    assert installed.status == "installed"
    assert registry.get_skill(server_id="lobby-1", skill_name="luckperms-debugging").valid


def test_skill_authoring_injects_builder_context(tmp_path: Path):
    class FakePluginDocsRetriever:
        def __init__(self):
            self.seen_query = ""

        def retrieve(self, search_query: str, *, top_k: int = 8):
            self.seen_query = search_query
            return [
                RetrievedDoc(
                    id=1,
                    plugin_chinese_name="",
                    plugin_english_name="LuckPerms",
                    content="Use verbose mode to inspect permission checks.",
                    distance=0.12,
                    match_reason="vector",
                )
            ]

    class FakeServerConfigRetriever:
        def __init__(self):
            self.seen_server_id = ""

        def retrieve(
            self,
            search_query: str,
            *,
            top_k: int = 8,
            server_id: str | None = None,
            plugin_name: str | None = None,
        ):
            self.seen_server_id = server_id or ""
            return [
                SemanticMemoryDoc(
                    server_id="lobby-1",
                    plugin_name="LuckPerms",
                    memory_type="plugin_config",
                    relation_type="sets",
                    memory_text="Default group has no essentials.fly permission.",
                    distance=0.08,
                    match_reason="vector",
                )
            ]

    class FakeWebResearchProvider:
        def __init__(self):
            self.seen_top_k = 0

        def search(self, search_query: str, *, top_k: int = 5):
            self.seen_top_k = top_k
            return WebResearchResponse(
                results=(
                    WebResearchResult(
                        title="LuckPerms wiki",
                        url="https://example.com/luckperms-wiki",
                        summary="LuckPerms verbose mode helps inspect permission checks.",
                        site_name="Example Wiki",
                        raw_rank=1,
                    ),
                )
            )

    registry = SkillRegistry(
        official_skills_dir=tmp_path / "official",
        global_skills_dir=tmp_path / "global",
        mc_servers_root=tmp_path / "mc_servers",
    )
    _write_skill(
        tmp_path / "global",
        "permission-debugging",
        description="Use for permission issues.",
        body="# Permission Debugging\n\nCheck inheritance.",
    )
    plugin_docs = FakePluginDocsRetriever()
    server_config = FakeServerConfigRetriever()
    web_research = FakeWebResearchProvider()
    client = FakeAuthoringClient(
        [
            {
                "status": "draft_ready",
                "skill": {
                    "name": "luckperms-debugging",
                    "description": "Use for LuckPerms permission issues.",
                    "body": "# LuckPerms Debugging\n\nPrefer verbose checks and local group evidence.",
                },
            }
        ]
    )
    service = SkillAuthoringService(
        registry=registry,
        client=client,
        context_builder=SkillAuthoringContextBuilder(
            skill_inventory=registry,
            plugin_docs_retriever=plugin_docs,
            server_config_retriever=server_config,
            web_research_provider=web_research,
            plugin_docs_top_k=2,
            server_config_top_k=2,
            web_research_top_k=3,
        ),
    )

    result = service.start(server_id="lobby-1", initial_requirement="LuckPerms 权限排查")

    assert result.status == "draft_ready"
    assert result.skill is not None
    assert result.skill.valid is True
    assert "Initial requirement" not in plugin_docs.seen_query
    assert "LuckPerms" in plugin_docs.seen_query
    assert server_config.seen_server_id == "lobby-1"
    rendered_messages = "\n\n".join(message["content"] for message in client.seen_messages[0])
    assert "Skill authoring context for this Minecraft server" in rendered_messages
    assert "permission-debugging" in rendered_messages
    assert "LuckPerms" in rendered_messages
    assert "Default group has no essentials.fly permission" in rendered_messages
    assert "External public web results" in rendered_messages
    assert "https://example.com/luckperms-wiki" in rendered_messages
    assert web_research.seen_top_k == 3
    assert rendered_messages.index("Relevant uploaded server configuration memories") < (
        rendered_messages.index("External web search results")
    )


def test_skill_authoring_context_diagnostics_do_not_invalidate_preview(tmp_path: Path):
    class FailingPluginDocsRetriever:
        def retrieve(self, search_query: str, *, top_k: int = 8):
            raise RuntimeError("docs index offline")

    class EmptyServerConfigRetriever:
        def retrieve(
            self,
            search_query: str,
            *,
            top_k: int = 8,
            server_id: str | None = None,
            plugin_name: str | None = None,
        ):
            return []

    class FailingWebResearchProvider:
        def search(self, search_query: str, *, top_k: int = 5):
            return WebResearchResponse(
                diagnostics=("web research request failed: timeout",),
            )

    registry = SkillRegistry(
        official_skills_dir=tmp_path / "official",
        global_skills_dir=tmp_path / "global",
        mc_servers_root=tmp_path / "mc_servers",
    )
    service = SkillAuthoringService(
        registry=registry,
        client=FakeAuthoringClient(
            [
                {
                    "status": "draft_ready",
                    "skill": {
                        "name": "claims-debugging",
                        "description": "Use for claim permission issues.",
                        "body": "# Claims Debugging\n\nAsk for the claim plugin and region details.",
                    },
                }
            ]
        ),
        context_builder=SkillAuthoringContextBuilder(
            skill_inventory=registry,
            plugin_docs_retriever=FailingPluginDocsRetriever(),
            server_config_retriever=EmptyServerConfigRetriever(),
            web_research_provider=FailingWebResearchProvider(),
        ),
    )

    result = service.start(server_id="lobby-1", initial_requirement="领地权限排查")

    assert result.skill is not None
    assert result.skill.valid is True
    assert result.skill.diagnostics == ()
    assert "plugin docs retrieval unavailable: docs index offline" in result.diagnostics
    assert (
        "server config semantic retrieval returned no matching memories"
        in result.diagnostics
    )
    assert "web research request failed: timeout" in result.diagnostics


def test_skill_authoring_serializes_concurrent_continue_calls(tmp_path: Path):
    class BlockingAuthoringClient:
        def __init__(self):
            self.responses = [
                {
                    "status": "needs_clarification",
                    "questions": ["Which plugin should this cover?"],
                },
                {
                    "status": "needs_clarification",
                    "questions": ["What workflow should it follow?"],
                },
                {
                    "status": "draft_ready",
                    "skill": {
                        "name": "claims-debugging",
                        "description": "Use for land claim debugging.",
                        "body": "# Claims Debugging\n\nCheck claim ownership and trust rules.",
                    },
                },
            ]
            self.lock = Lock()
            self.first_continue_entered = Event()
            self.release_first_continue = Event()
            self.call_count = 0
            self.active_calls = 0
            self.max_active_calls = 0

        def chat(self, messages, *, temperature: float = 0.1):
            with self.lock:
                self.call_count += 1
                call_number = self.call_count
                self.active_calls += 1
                self.max_active_calls = max(self.max_active_calls, self.active_calls)
            try:
                if call_number == 2:
                    self.first_continue_entered.set()
                    assert self.release_first_continue.wait(timeout=2.0)
                with self.lock:
                    response = self.responses.pop(0)
                return json.dumps(response, ensure_ascii=False)
            finally:
                with self.lock:
                    self.active_calls -= 1

    registry = SkillRegistry(
        official_skills_dir=tmp_path / "official",
        global_skills_dir=tmp_path / "global",
        mc_servers_root=tmp_path / "mc_servers",
    )
    client = BlockingAuthoringClient()
    service = SkillAuthoringService(registry=registry, client=client)

    draft = service.start(server_id="lobby-1", initial_requirement="claim debugging")

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(
            service.continue_draft,
            server_id="lobby-1",
            draft_id=draft.draft_id,
            user_message="Towny claims",
        )
        second = executor.submit(
            service.continue_draft,
            server_id="lobby-1",
            draft_id=draft.draft_id,
            user_message="WorldGuard regions",
        )
        assert client.first_continue_entered.wait(timeout=2.0)
        sleep(0.05)
        assert client.max_active_calls == 1
        client.release_first_continue.set()

        statuses = {first.result(timeout=2.0).status, second.result(timeout=2.0).status}
        assert statuses == {"needs_clarification", "draft_ready"}
    assert client.max_active_calls == 1


def test_skill_authoring_includes_official_skill_creator_guidance(tmp_path: Path):
    _write_skill(
        tmp_path / "official",
        "skill-creator",
        description="Use only during skill authoring.",
        body="# Official Creator\n\nAsk focused questions.",
        usage="authoring",
    )
    registry = SkillRegistry(
        official_skills_dir=tmp_path / "official",
        global_skills_dir=tmp_path / "global",
        mc_servers_root=tmp_path / "mc_servers",
    )
    client = FakeAuthoringClient(
        [
            {
                "status": "needs_clarification",
                "questions": ["Which plugin should this cover?"],
            }
        ]
    )
    service = SkillAuthoringService(registry=registry, client=client)

    service.start(server_id="lobby-1", initial_requirement="权限排查")

    assert "# Official Creator" in client.seen_messages[0][0]["content"]


def test_skill_authoring_rejects_existing_name(tmp_path: Path):
    registry = SkillRegistry(
        official_skills_dir=tmp_path / "official",
        global_skills_dir=tmp_path / "global",
        mc_servers_root=tmp_path / "mc_servers",
    )
    _write_skill(
        tmp_path / "global",
        "luckperms-debugging",
        description="Use for global LuckPerms debugging.",
        body="# LuckPerms\n\nGlobal rules.",
    )
    service = SkillAuthoringService(
        registry=registry,
        client=FakeAuthoringClient(
            [
                {
                    "status": "draft_ready",
                    "skill": {
                        "name": "luckperms-debugging",
                        "description": "Use for LuckPerms permission denied issues.",
                        "body": "# LuckPerms Debugging\n\nCheck groups.",
                    },
                }
            ]
        ),
    )

    draft = service.start(server_id="lobby-1", initial_requirement="权限排查")

    assert draft.status == "needs_clarification"
    assert draft.skill is not None
    assert draft.skill.valid is False
    assert draft.diagnostics == ("skill already exists: luckperms-debugging",)
    assert "cannot be installed yet" in draft.questions[0]
    with pytest.raises(SkillValidationError):
        service.confirm(server_id="lobby-1", draft_id=draft.draft_id)


def test_skill_registry_reports_oversized_skill_file(tmp_path: Path):
    skill_dir = tmp_path / "mc_servers" / "lobby-1" / "skills" / "huge-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("x" * 2048, encoding="utf-8")
    registry = SkillRegistry(
        official_skills_dir=tmp_path / "official",
        global_skills_dir=tmp_path / "global",
        mc_servers_root=tmp_path / "mc_servers",
        max_skill_bytes=32,
    )

    record = registry.get_skill(server_id="lobby-1", skill_name="huge-skill")

    assert record.valid is False
    assert any("exceeds 1024 bytes" in diagnostic for diagnostic in record.diagnostics)


def test_delete_skill_only_allows_server_scope(tmp_path: Path):
    registry = SkillRegistry(
        official_skills_dir=tmp_path / "official",
        global_skills_dir=tmp_path / "global",
        mc_servers_root=tmp_path / "mc_servers",
    )
    _write_skill(
        tmp_path / "global",
        "global-rules",
        description="Use for shared backend rules.",
        body="# Global Rules\n\nRead only.",
    )
    _write_skill(
        tmp_path / "mc_servers" / "lobby-1" / "skills",
        "server-rules",
        description="Use for this server's rules.",
        body="# Server Rules\n\nCan be deleted.",
    )

    with pytest.raises(SkillReadonlyError):
        registry.delete_server_skill(server_id="lobby-1", skill_name="global-rules")

    result = registry.delete_server_skill(server_id="lobby-1", skill_name="server-rules")

    assert result.deleted is True
    assert result.archived_path is not None
    assert result.archived_path.exists()
    with pytest.raises(SkillNotFoundError):
        registry.get_skill(server_id="lobby-1", skill_name="server-rules")


def test_delete_server_skill_can_remove_invalid_conflicting_server_skill(tmp_path: Path):
    registry = SkillRegistry(
        official_skills_dir=tmp_path / "official",
        global_skills_dir=tmp_path / "global",
        mc_servers_root=tmp_path / "mc_servers",
    )
    _write_skill(
        tmp_path / "global",
        "shared-rules",
        description="Use for shared rules.",
        body="# Shared Rules\n\nGlobal.",
    )
    _write_skill(
        tmp_path / "mc_servers" / "lobby-1" / "skills",
        "shared-rules",
        description="Conflicting server copy.",
        body="# Shared Rules\n\nServer duplicate.",
    )

    duplicate = [
        skill
        for skill in registry.list_skills("lobby-1")
        if skill.scope == SkillScope.SERVER and skill.name == "shared-rules"
    ][0]
    result = registry.delete_server_skill(server_id="lobby-1", skill_name="shared-rules")

    assert duplicate.valid is False
    assert result.deleted is True


def test_registry_rejects_server_id_path_traversal(tmp_path: Path):
    registry = SkillRegistry(
        official_skills_dir=tmp_path / "official",
        global_skills_dir=tmp_path / "global",
        mc_servers_root=tmp_path / "mc_servers",
    )

    with pytest.raises(SkillValidationError):
        registry.list_skills("../other")


def test_rag_chat_session_injects_matching_skill_context(tmp_path: Path):
    mc_root = tmp_path / "mc_servers"
    _write_skill(
        mc_root / "lobby-1" / "skills",
        "economy-debugging",
        description="Use for Vault economy balance issues.",
        body="# Economy Debugging\n\nCheck Vault and balance commands.",
    )
    registry = SkillRegistry(
        official_skills_dir=tmp_path / "official",
        global_skills_dir=tmp_path / "global",
        mc_servers_root=mc_root,
    )
    agent = CapturingAgent()
    session = RagChatSession(
        FakeSettings(),
        FakeVectorStore(),
        agent,
        skill_registry=registry,
    )

    result = session.ask("Vault 余额不对怎么办？", server_id="lobby-1")

    messages = agent.payloads[0]["messages"]
    skill_messages = [
        message
        for message in messages
        if isinstance(message, SystemMessage) and "economy-debugging" in str(message.content)
    ]
    assert result.answer == "answer"
    assert len(skill_messages) == 1


def _write_skill(
    root: Path,
    name: str,
    *,
    description: str,
    body: str,
    usage: str = "qa",
) -> Path:
    skill_dir = root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_path = skill_dir / "SKILL.md"
    skill_path.write_text(
        render_skill_markdown(
            name=name,
            description=description,
            body=body,
            usage=usage,
        ),
        encoding="utf-8",
    )
    return skill_path
