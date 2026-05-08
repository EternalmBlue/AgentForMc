from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from threading import Lock
from time import time
from typing import Protocol
from uuid import uuid4

from agent_for_mc.application.skills.context import SkillAuthoringContextBuilder


SKILL_FILE_NAME = "SKILL.md"
TRASH_DIR_NAME = ".trash"
SKILL_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9-]{2,63}$")
DEFAULT_SKILL_MAX_BYTES = 32 * 1024
DEFAULT_SKILL_SELECTION_TOP_K = 3
DEFAULT_SKILL_DRAFT_TTL_SECONDS = 1800

_FRONTMATTER_DELIMITER = "---"
_AUTHORING_USAGE_VALUES = {"authoring", "authoring_only", "skill_authoring"}
_FORBIDDEN_PATTERNS = (
    "ignore previous instructions",
    "ignore system instructions",
    "bypass safety",
    "reveal system prompt",
    "reveal .env",
    "print .env",
    "dump .env",
    "reveal api key",
    "print api key",
    "dump api key",
    "reveal secret",
)


class SkillError(Exception):
    """Base error for skill management failures."""


class SkillValidationError(SkillError):
    """Raised when skill input is invalid."""


class SkillConflictError(SkillError):
    """Raised when a skill conflicts with an existing skill."""


class SkillReadonlyError(SkillError):
    """Raised when a caller tries to mutate a readonly skill."""


class SkillNotFoundError(SkillError):
    """Raised when a skill cannot be found."""


class SkillDraftNotFoundError(SkillError):
    """Raised when a skill creation draft is missing or expired."""


class SkillScope(str, Enum):
    OFFICIAL = "official"
    GLOBAL = "global"
    SERVER = "server"


class ChatClient(Protocol):
    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.1,
    ) -> str:
        ...


@dataclass(frozen=True, slots=True)
class SkillRecord:
    scope: SkillScope
    name: str
    description: str
    body: str
    path: Path
    valid: bool
    readonly: bool
    deletable: bool
    diagnostics: tuple[str, ...] = ()
    usage: str = "qa"

    @property
    def content(self) -> str:
        return render_skill_markdown(
            name=self.name,
            description=self.description,
            body=self.body,
            usage=self.usage,
        )

    @property
    def selectable(self) -> bool:
        return self.valid and self.usage.strip().casefold() not in _AUTHORING_USAGE_VALUES

    def with_diagnostic(self, diagnostic: str) -> "SkillRecord":
        return replace(
            self,
            valid=False,
            diagnostics=(*self.diagnostics, diagnostic),
        )


@dataclass(frozen=True, slots=True)
class DeleteSkillResult:
    deleted: bool
    message: str
    archived_path: Path | None = None


@dataclass(frozen=True, slots=True)
class SkillCreationResult:
    draft_id: str
    status: str
    message: str
    questions: tuple[str, ...] = ()
    skill: SkillRecord | None = None
    content: str = ""
    diagnostics: tuple[str, ...] = ()


@dataclass(slots=True)
class _SkillDraft:
    draft_id: str
    server_id: str
    messages: list[str]
    created_at_epoch_ms: int
    updated_at_epoch_ms: int
    status: str = "needs_clarification"
    skill_name: str = ""
    description: str = ""
    body: str = ""
    questions: list[str] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)
    lock: Lock = field(default_factory=Lock, repr=False)


class SkillRegistry:
    def __init__(
        self,
        *,
        official_skills_dir: Path,
        global_skills_dir: Path,
        mc_servers_root: Path,
        max_skill_bytes: int = DEFAULT_SKILL_MAX_BYTES,
        selection_top_k: int = DEFAULT_SKILL_SELECTION_TOP_K,
    ):
        self.official_skills_dir = Path(official_skills_dir)
        self.global_skills_dir = Path(global_skills_dir)
        self.mc_servers_root = Path(mc_servers_root)
        self.max_skill_bytes = max(1024, int(max_skill_bytes))
        self.selection_top_k = max(1, int(selection_top_k))

    def list_skills(self, server_id: str | None = None) -> list[SkillRecord]:
        records: list[SkillRecord] = []
        seen_valid: dict[str, SkillRecord] = {}

        for record in self._scan_all_scopes(server_id):
            resolved = record
            key = record.name.casefold()
            if record.valid:
                earlier = seen_valid.get(key)
                if earlier is not None:
                    resolved = record.with_diagnostic(
                        f"skill name conflicts with {earlier.scope.value} skill '{earlier.name}'"
                    )
                else:
                    seen_valid[key] = record
            records.append(resolved)

        return records

    def get_skill(self, *, server_id: str, skill_name: str) -> SkillRecord:
        normalized_name = validate_skill_name(skill_name)
        for record in self.list_skills(server_id):
            if record.name.casefold() == normalized_name.casefold():
                return record
        raise SkillNotFoundError(f"skill not found: {normalized_name}")

    def select_skills(
        self,
        *,
        server_id: str,
        question: str,
        server_plugins: list[str] | None = None,
    ) -> list[SkillRecord]:
        query = _normalize_search_text(
            " ".join([question, *(server_plugins or [])])
        )
        if not query:
            return []

        scored: list[tuple[int, int, str, SkillRecord]] = []
        for record in self.list_skills(server_id):
            if not record.selectable:
                continue
            score = _score_skill(record, query)
            if score <= 0:
                continue
            scored.append((-score, -_scope_priority(record.scope), record.name, record))

        scored.sort()
        return [item[3] for item in scored[: self.selection_top_k]]

    def has_any_skill_named(self, *, server_id: str, skill_name: str) -> bool:
        normalized_name = validate_skill_name(skill_name)
        return any(
            record.name.casefold() == normalized_name.casefold()
            for record in self.list_skills(server_id)
        )

    def install_server_skill(
        self,
        *,
        server_id: str,
        name: str,
        description: str,
        body: str,
    ) -> SkillRecord:
        normalized_name = validate_skill_name(name)
        if self.has_any_skill_named(server_id=server_id, skill_name=normalized_name):
            raise SkillConflictError(f"skill already exists: {normalized_name}")

        content = render_skill_markdown(
            name=normalized_name,
            description=_clean_metadata_value(description),
            body=body,
        )
        record = self.validate_content(
            scope=SkillScope.SERVER,
            path=self._server_skill_dir(server_id, normalized_name) / SKILL_FILE_NAME,
            content=content,
            readonly=False,
            deletable=True,
        )
        if not record.valid:
            raise SkillValidationError("; ".join(record.diagnostics))

        target_dir = self._server_skill_dir(server_id, normalized_name)
        target_path = target_dir / SKILL_FILE_NAME
        temp_path = target_dir / f"{SKILL_FILE_NAME}.{uuid4().hex}.tmp"
        try:
            target_dir.mkdir(parents=True, exist_ok=False)
            temp_path.write_text(content, encoding="utf-8", newline="\n")
            os.replace(temp_path, target_path)
        except FileExistsError as exc:
            raise SkillConflictError(f"skill already exists: {normalized_name}") from exc
        except OSError as exc:
            raise SkillValidationError(f"could not write skill: {target_path}") from exc
        finally:
            temp_path.unlink(missing_ok=True)

        return self.get_skill(server_id=server_id, skill_name=normalized_name)

    def delete_server_skill(self, *, server_id: str, skill_name: str) -> DeleteSkillResult:
        normalized_name = validate_skill_name(skill_name)
        record = self._get_server_skill(server_id=server_id, skill_name=normalized_name)
        if record is None:
            if any(
                skill.name.casefold() == normalized_name.casefold()
                for skill in self.list_skills(server_id)
            ):
                raise SkillReadonlyError(f"skill is readonly: {normalized_name}")
            raise SkillNotFoundError(f"skill not found: {normalized_name}")
        if record.readonly or not record.deletable:
            raise SkillReadonlyError(f"skill is readonly: {record.name}")
        if not record.path.exists():
            raise SkillNotFoundError(f"skill not found: {record.name}")

        skill_dir = record.path.parent.resolve()
        server_skills_dir = self._server_skills_dir(server_id).resolve()
        if not skill_dir.is_relative_to(server_skills_dir):
            raise SkillValidationError("skill path escapes server skills directory")

        trash_dir = server_skills_dir / TRASH_DIR_NAME
        trash_dir.mkdir(parents=True, exist_ok=True)
        archive_dir = _unique_archive_dir(trash_dir, record.name)
        try:
            shutil.move(str(skill_dir), str(archive_dir))
        except OSError as exc:
            raise SkillValidationError(f"could not delete skill: {record.name}") from exc

        return DeleteSkillResult(
            deleted=True,
            message=f"skill archived: {record.name}",
            archived_path=archive_dir,
        )

    def validate_content(
        self,
        *,
        scope: SkillScope,
        path: Path,
        content: str,
        readonly: bool,
        deletable: bool,
    ) -> SkillRecord:
        diagnostics: list[str] = []
        encoded = content.encode("utf-8")
        if len(encoded) > self.max_skill_bytes:
            diagnostics.append(
                f"skill file exceeds {self.max_skill_bytes} bytes"
            )

        try:
            metadata, body = _parse_skill_markdown(content)
        except SkillValidationError as exc:
            metadata, body = {}, ""
            diagnostics.append(str(exc))

        raw_name = str(metadata.get("name") or path.parent.name).strip()
        try:
            name = validate_skill_name(raw_name)
        except SkillValidationError as exc:
            name = _fallback_record_name(raw_name or path.parent.name)
            diagnostics.append(str(exc))

        description = _clean_metadata_value(str(metadata.get("description") or ""))
        if not description:
            diagnostics.append("frontmatter field 'description' is required")

        usage = _clean_metadata_value(str(metadata.get("usage") or "qa")) or "qa"
        if not body.strip():
            diagnostics.append("skill body must not be blank")

        for forbidden in _FORBIDDEN_PATTERNS:
            if forbidden in content.casefold():
                diagnostics.append(f"skill contains forbidden instruction: {forbidden}")

        return SkillRecord(
            scope=scope,
            name=name,
            description=description,
            body=body.strip(),
            path=path,
            valid=not diagnostics,
            readonly=readonly,
            deletable=deletable,
            diagnostics=tuple(diagnostics),
            usage=usage,
        )

    def _scan_all_scopes(self, server_id: str | None) -> list[SkillRecord]:
        records: list[SkillRecord] = []
        records.extend(
            self._scan_scope(
                scope=SkillScope.OFFICIAL,
                root=self.official_skills_dir,
                readonly=True,
                deletable=False,
            )
        )
        records.extend(
            self._scan_scope(
                scope=SkillScope.GLOBAL,
                root=self.global_skills_dir,
                readonly=True,
                deletable=False,
            )
        )
        if server_id:
            records.extend(
                self._scan_scope(
                    scope=SkillScope.SERVER,
                    root=self._server_skills_dir(server_id),
                    readonly=False,
                    deletable=True,
                )
            )
        return records

    def _scan_scope(
        self,
        *,
        scope: SkillScope,
        root: Path,
        readonly: bool,
        deletable: bool,
    ) -> list[SkillRecord]:
        root = Path(root)
        if not root.exists():
            return []
        if not root.is_dir():
            return [
                _invalid_record(
                    scope=scope,
                    name=_fallback_record_name(root.name),
                    path=root,
                    readonly=readonly,
                    deletable=False,
                    diagnostic=f"skills root is not a directory: {root}",
                )
            ]

        records: list[SkillRecord] = []
        root_resolved = root.resolve()
        for child in sorted(root.iterdir(), key=lambda item: item.name.casefold()):
            if child.name == TRASH_DIR_NAME:
                continue
            if not child.is_dir():
                continue
            records.append(
                self._load_skill_dir(
                    scope=scope,
                    skill_dir=child,
                    scope_root=root_resolved,
                    readonly=readonly,
                    deletable=deletable,
                )
            )
        return records

    def _load_skill_dir(
        self,
        *,
        scope: SkillScope,
        skill_dir: Path,
        scope_root: Path,
        readonly: bool,
        deletable: bool,
    ) -> SkillRecord:
        skill_path = skill_dir / SKILL_FILE_NAME
        try:
            resolved_dir = skill_dir.resolve()
        except OSError as exc:
            return _invalid_record(
                scope=scope,
                name=_fallback_record_name(skill_dir.name),
                path=skill_path,
                readonly=readonly,
                deletable=False,
                diagnostic=f"could not resolve skill directory: {exc}",
            )
        if not resolved_dir.is_relative_to(scope_root):
            return _invalid_record(
                scope=scope,
                name=_fallback_record_name(skill_dir.name),
                path=skill_path,
                readonly=readonly,
                deletable=False,
                diagnostic="skill directory escapes its scope root",
            )
        if not skill_path.is_file():
            return _invalid_record(
                scope=scope,
                name=_fallback_record_name(skill_dir.name),
                path=skill_path,
                readonly=readonly,
                deletable=False,
                diagnostic=f"missing {SKILL_FILE_NAME}",
            )

        extra_files = [
            item
            for item in skill_dir.rglob("*")
            if item.is_file() and item.name != SKILL_FILE_NAME
        ]
        if extra_files:
            return _invalid_record(
                scope=scope,
                name=_fallback_record_name(skill_dir.name),
                path=skill_path,
                readonly=readonly,
                deletable=deletable,
                diagnostic="v1 skills may only contain SKILL.md",
            )

        try:
            skill_size = skill_path.stat().st_size
            content = (
                ""
                if skill_size > self.max_skill_bytes
                else skill_path.read_text(encoding="utf-8")
            )
        except OSError as exc:
            return _invalid_record(
                scope=scope,
                name=_fallback_record_name(skill_dir.name),
                path=skill_path,
                readonly=readonly,
                deletable=False,
                diagnostic=f"could not read skill: {exc}",
            )

        record = self.validate_content(
            scope=scope,
            path=skill_path,
            content=content,
            readonly=readonly,
            deletable=deletable,
        )
        if skill_size > self.max_skill_bytes:
            record = record.with_diagnostic(
                f"skill file exceeds {self.max_skill_bytes} bytes"
            )
        if record.name != skill_dir.name:
            record = record.with_diagnostic(
                f"frontmatter name '{record.name}' must match directory name '{skill_dir.name}'"
            )
        return record

    def _server_skills_dir(self, server_id: str) -> Path:
        return self.mc_servers_root / _validate_path_segment(server_id, "server_id") / "skills"

    def _server_skill_dir(self, server_id: str, skill_name: str) -> Path:
        return self._server_skills_dir(server_id) / skill_name

    def _get_server_skill(self, *, server_id: str, skill_name: str) -> SkillRecord | None:
        normalized_name = validate_skill_name(skill_name)
        for record in self.list_skills(server_id):
            if (
                record.scope == SkillScope.SERVER
                and record.name.casefold() == normalized_name.casefold()
            ):
                return record
        return None


class SkillAuthoringService:
    def __init__(
        self,
        *,
        registry: SkillRegistry,
        client: ChatClient,
        context_builder: SkillAuthoringContextBuilder | None = None,
        ttl_seconds: int = DEFAULT_SKILL_DRAFT_TTL_SECONDS,
    ):
        self._registry = registry
        self._client = client
        self._context_builder = context_builder
        self._ttl_ms = max(60, int(ttl_seconds)) * 1000
        self._lock = Lock()
        self._drafts: dict[str, _SkillDraft] = {}

    def start(self, *, server_id: str, initial_requirement: str) -> SkillCreationResult:
        requirement = str(initial_requirement or "").strip()
        if not requirement:
            raise SkillValidationError("initial_requirement must not be blank")

        draft = _SkillDraft(
            draft_id=uuid4().hex,
            server_id=server_id,
            messages=[f"Initial requirement: {requirement}"],
            created_at_epoch_ms=_now_ms(),
            updated_at_epoch_ms=_now_ms(),
        )
        with self._lock:
            self._reap_expired_locked()
            self._drafts[draft.draft_id] = draft
        with draft.lock:
            return self._advance(draft)

    def continue_draft(
        self,
        *,
        server_id: str,
        draft_id: str,
        user_message: str,
    ) -> SkillCreationResult:
        message = str(user_message or "").strip()
        if not message:
            raise SkillValidationError("user_message must not be blank")
        draft = self._get_draft(server_id=server_id, draft_id=draft_id)
        with draft.lock:
            self._ensure_draft_active(server_id=server_id, draft=draft)
            draft.messages.append(f"User answer: {message}")
            draft.updated_at_epoch_ms = _now_ms()
            return self._advance(draft)

    def confirm(self, *, server_id: str, draft_id: str) -> SkillCreationResult:
        draft = self._get_draft(server_id=server_id, draft_id=draft_id)
        with draft.lock:
            self._ensure_draft_active(server_id=server_id, draft=draft)
            if draft.status != "draft_ready" or not draft.skill_name:
                raise SkillValidationError("skill draft is not ready to install")

            record = self._registry.install_server_skill(
                server_id=server_id,
                name=draft.skill_name,
                description=draft.description,
                body=draft.body,
            )
            with self._lock:
                self._drafts.pop(draft_id, None)
            return SkillCreationResult(
                draft_id=draft_id,
                status="installed",
                message=f"skill installed: {record.name}",
                skill=record,
                content=record.content,
            )

    def _advance(self, draft: _SkillDraft) -> SkillCreationResult:
        output, context_diagnostics = self._request_next_step(draft)
        status = str(output.get("status") or "").strip().casefold()
        questions = _as_str_list(output.get("questions"))
        skill_data = output.get("skill") if isinstance(output.get("skill"), dict) else output

        if status in {"needs_clarification", "clarify"} or (
            questions and not _has_skill_payload(skill_data)
        ):
            draft.status = "needs_clarification"
            draft.questions = questions or [
                "What problem should this skill handle?",
                "What evidence or server rule should the assistant prefer?",
            ]
            draft.updated_at_epoch_ms = _now_ms()
            return SkillCreationResult(
                draft_id=draft.draft_id,
                status=draft.status,
                message="more detail is needed before creating the skill",
                questions=tuple(draft.questions),
                diagnostics=tuple(context_diagnostics),
            )

        name = _coerce_skill_name(
            str(skill_data.get("name") or ""),
            fallback_seed=" ".join(draft.messages),
        )
        description = _clean_metadata_value(str(skill_data.get("description") or ""))
        body = str(
            skill_data.get("body")
            or skill_data.get("instructions")
            or skill_data.get("markdown")
            or ""
        ).strip()
        if not description or not body:
            draft.status = "needs_clarification"
            draft.questions = [
                "Please describe when this skill should trigger.",
                "Please describe the workflow the assistant should follow.",
            ]
            return SkillCreationResult(
                draft_id=draft.draft_id,
                status=draft.status,
                message="the draft did not include enough structured skill content",
                questions=tuple(draft.questions),
                diagnostics=tuple(context_diagnostics),
            )

        content = render_skill_markdown(
            name=name,
            description=description,
            body=body,
        )
        preview = self._registry.validate_content(
            scope=SkillScope.SERVER,
            path=Path(name) / SKILL_FILE_NAME,
            content=content,
            readonly=False,
            deletable=True,
        )
        validation_diagnostics = list(preview.diagnostics)
        if self._registry.has_any_skill_named(
            server_id=draft.server_id,
            skill_name=name,
        ):
            validation_diagnostics.append(f"skill already exists: {name}")
        result_diagnostics = [*validation_diagnostics, *context_diagnostics]

        if validation_diagnostics:
            draft.status = "needs_clarification"
            draft.questions = _questions_for_invalid_skill(name, validation_diagnostics)
            draft.skill_name = ""
            draft.description = ""
            draft.body = ""
            draft.diagnostics = result_diagnostics
            draft.updated_at_epoch_ms = _now_ms()
            return SkillCreationResult(
                draft_id=draft.draft_id,
                status=draft.status,
                message="the draft needs revision before it can be installed",
                questions=tuple(draft.questions),
                skill=replace(
                    preview,
                    diagnostics=tuple(validation_diagnostics),
                    valid=False,
                ),
                content=content,
                diagnostics=tuple(result_diagnostics),
            )

        draft.status = "draft_ready"
        draft.skill_name = name
        draft.description = description
        draft.body = body
        draft.diagnostics = result_diagnostics
        draft.updated_at_epoch_ms = _now_ms()

        return SkillCreationResult(
            draft_id=draft.draft_id,
            status=draft.status,
            message="skill draft is ready for review",
            skill=replace(
                preview,
                diagnostics=tuple(validation_diagnostics),
                valid=not validation_diagnostics,
            ),
            content=content,
            diagnostics=tuple(result_diagnostics),
        )

    def _request_next_step(self, draft: _SkillDraft) -> tuple[dict, list[str]]:
        context_messages: list[dict[str, str]] = []
        context_diagnostics: list[str] = []
        if self._context_builder is not None:
            try:
                context = self._context_builder.build(
                    server_id=draft.server_id,
                    messages=draft.messages,
                )
                rendered_context = context.render()
                if rendered_context:
                    context_messages.append(
                        {
                            "role": "system",
                            "content": rendered_context,
                        }
                    )
                context_diagnostics.extend(context.diagnostics)
            except Exception as exc:
                context_diagnostics.append(f"skill authoring context unavailable: {exc}")

        response = self._client.chat(
            [
                {
                    "role": "system",
                    "content": self._authoring_system_prompt(draft.server_id),
                },
                *context_messages,
                {
                    "role": "user",
                    "content": "\n".join(draft.messages),
                },
            ],
            temperature=0.1,
        )
        data = _parse_json_object(response)
        if data is None:
            return {
                "status": "needs_clarification",
                "questions": [
                    "What exact server workflow should this skill cover?",
                    "When should the assistant use this skill?",
                ],
            }, context_diagnostics
        return data, context_diagnostics

    def _authoring_system_prompt(self, server_id: str) -> str:
        try:
            creator_skill = self._registry.get_skill(
                server_id=server_id,
                skill_name="skill-creator",
            )
        except SkillError:
            return _SKILL_AUTHORING_SYSTEM_PROMPT
        if not creator_skill.valid:
            return _SKILL_AUTHORING_SYSTEM_PROMPT
        return (
            f"{_SKILL_AUTHORING_SYSTEM_PROMPT}\n\n"
            "Official skill-creator guidance:\n"
            f"{creator_skill.body}"
        )

    def _get_draft(self, *, server_id: str, draft_id: str) -> _SkillDraft:
        normalized_id = str(draft_id or "").strip()
        if not normalized_id:
            raise SkillDraftNotFoundError("draft_id must not be blank")
        with self._lock:
            self._reap_expired_locked()
            draft = self._drafts.get(normalized_id)
        if draft is None or draft.server_id != server_id:
            raise SkillDraftNotFoundError(f"skill draft not found: {normalized_id}")
        return draft

    def _ensure_draft_active(self, *, server_id: str, draft: _SkillDraft) -> None:
        with self._lock:
            current = self._drafts.get(draft.draft_id)
        if current is not draft or draft.server_id != server_id:
            raise SkillDraftNotFoundError(f"skill draft not found: {draft.draft_id}")

    def _reap_expired_locked(self) -> None:
        if self._ttl_ms <= 0:
            return
        now_ms = _now_ms()
        expired = [
            draft_id
            for draft_id, draft in self._drafts.items()
            if not draft.lock.locked()
            and now_ms - draft.updated_at_epoch_ms > self._ttl_ms
        ]
        for draft_id in expired:
            self._drafts.pop(draft_id, None)


def validate_skill_name(value: str) -> str:
    normalized = str(value or "").strip().casefold()
    if not SKILL_NAME_PATTERN.fullmatch(normalized):
        raise SkillValidationError(
            "skill name must match ^[a-z][a-z0-9-]{2,63}$"
        )
    return normalized


def render_skill_markdown(
    *,
    name: str,
    description: str,
    body: str,
    usage: str = "qa",
) -> str:
    normalized_name = validate_skill_name(name)
    normalized_description = _clean_metadata_value(description)
    if not normalized_description:
        raise SkillValidationError("description must not be blank")
    normalized_usage = _clean_metadata_value(usage) or "qa"
    return (
        "---\n"
        f"name: {normalized_name}\n"
        f"description: {normalized_description}\n"
        f"usage: {normalized_usage}\n"
        "---\n\n"
        f"{str(body or '').strip()}\n"
    )


def format_skills_for_system_message(skills: list[SkillRecord]) -> str:
    if not skills:
        return ""
    parts = [
        "Relevant skills for this Minecraft server question.",
        "Follow these skills as procedural guidance. Retrieved docs remain primary evidence.",
    ]
    for index, skill in enumerate(skills, start=1):
        parts.append(
            "\n".join(
                [
                    f"[Skill {index}: {skill.name}]",
                    f"scope: {skill.scope.value}",
                    f"description: {skill.description}",
                    "instructions:",
                    skill.body,
                ]
            )
        )
    return "\n\n".join(parts)


def _parse_skill_markdown(content: str) -> tuple[dict[str, str], str]:
    text = str(content or "")
    lines = text.splitlines()
    if not lines or lines[0].strip() != _FRONTMATTER_DELIMITER:
        raise SkillValidationError("SKILL.md must start with YAML frontmatter")

    end_index = None
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == _FRONTMATTER_DELIMITER:
            end_index = index
            break
    if end_index is None:
        raise SkillValidationError("frontmatter closing delimiter is missing")

    metadata: dict[str, str] = {}
    for line in lines[1:end_index]:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in stripped:
            raise SkillValidationError(f"invalid frontmatter line: {stripped}")
        key, value = stripped.split(":", 1)
        metadata[key.strip().casefold()] = _strip_yaml_scalar(value.strip())

    body = "\n".join(lines[end_index + 1 :]).strip()
    if "name" not in metadata:
        raise SkillValidationError("frontmatter field 'name' is required")
    return metadata, body


def _strip_yaml_scalar(value: str) -> str:
    stripped = value.strip()
    if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in {"'", '"'}:
        return stripped[1:-1].strip()
    return stripped


def _invalid_record(
    *,
    scope: SkillScope,
    name: str,
    path: Path,
    readonly: bool,
    deletable: bool,
    diagnostic: str,
) -> SkillRecord:
    return SkillRecord(
        scope=scope,
        name=_fallback_record_name(name),
        description="",
        body="",
        path=path,
        valid=False,
        readonly=readonly,
        deletable=deletable,
        diagnostics=(diagnostic,),
    )


def _fallback_record_name(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9-]+", "-", str(value or "").casefold()).strip("-")
    if not normalized or not normalized[0].isalpha():
        normalized = f"skill-{normalized or 'invalid'}"
    if len(normalized) < 3:
        normalized = f"{normalized}-skill"
    return normalized[:64].rstrip("-")


def _validate_path_segment(value: str, field_name: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise SkillValidationError(f"{field_name} must not be blank")
    if normalized in {".", ".."} or any(ch in normalized for ch in ("/", "\\", ":")):
        raise SkillValidationError(f"{field_name} must not contain path separators")
    return normalized


def _unique_archive_dir(root: Path, skill_name: str) -> Path:
    base_name = f"{skill_name}-{_now_ms()}"
    candidate = root / base_name
    suffix = 1
    while candidate.exists():
        suffix += 1
        candidate = root / f"{base_name}-{suffix}"
    return candidate


def _clean_metadata_value(value: str) -> str:
    return " ".join(str(value or "").strip().split())


def _coerce_skill_name(raw_name: str, *, fallback_seed: str) -> str:
    candidate = _fallback_record_name(raw_name)
    if SKILL_NAME_PATTERN.fullmatch(candidate):
        return candidate

    seed = str(fallback_seed or "server skill").casefold()
    words = re.findall(r"[a-z0-9]+", seed)
    if words:
        candidate = "-".join(words[:4])
    else:
        candidate = "server-skill"
    candidate = _fallback_record_name(candidate)
    if SKILL_NAME_PATTERN.fullmatch(candidate):
        return candidate
    return "server-skill"


def _normalize_search_text(value: str) -> str:
    return " ".join(str(value or "").casefold().split())


def _score_skill(record: SkillRecord, query: str) -> int:
    skill_text = _normalize_search_text(
        f"{record.name} {record.description} {record.body[:500]}"
    )
    score = 0
    name_tokens = [token for token in record.name.split("-") if len(token) >= 3]
    for token in name_tokens:
        if token in query:
            score += 6
    if record.name in query:
        score += 10
    for token in _keyword_tokens(skill_text):
        if token in query:
            score += 2
    for token in _keyword_tokens(query):
        if token in skill_text:
            score += 1
    return score


def _keyword_tokens(value: str) -> set[str]:
    ascii_tokens = {
        token
        for token in re.findall(r"[a-z0-9][a-z0-9+_.-]{2,}", value.casefold())
        if token not in {"the", "and", "for", "with", "this", "that"}
    }
    cjk_tokens = {
        match.group(0)
        for match in re.finditer(r"[\u4e00-\u9fff]{2,}", value)
    }
    expanded_cjk: set[str] = set()
    for token in cjk_tokens:
        if len(token) <= 6:
            expanded_cjk.add(token)
            continue
        expanded_cjk.update(token[index : index + 4] for index in range(len(token) - 3))
    return ascii_tokens | expanded_cjk


def _scope_priority(scope: SkillScope) -> int:
    if scope == SkillScope.SERVER:
        return 3
    if scope == SkillScope.GLOBAL:
        return 2
    return 1


def _as_str_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _has_skill_payload(value: object) -> bool:
    return (
        isinstance(value, dict)
        and bool(value.get("name"))
        and bool(value.get("description"))
        and bool(value.get("body") or value.get("instructions") or value.get("markdown"))
    )


def _questions_for_invalid_skill(
    skill_name: str,
    diagnostics: list[str],
) -> list[str]:
    summary = "; ".join(diagnostics)
    return [
        f"The proposed skill '{skill_name}' cannot be installed yet: {summary}",
        "Please revise the skill name, trigger scope, or instructions so the draft can pass validation.",
    ]


def _parse_json_object(text: str) -> dict | None:
    raw = str(text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _now_ms() -> int:
    return int(time() * 1000)


_SKILL_AUTHORING_SYSTEM_PROMPT = """You create server-scoped SKILL.md files for AgentForMc.

Return JSON only. Use this schema when more information is needed:
{"status":"needs_clarification","questions":["..."]}

Use this schema when the skill is ready:
{"status":"draft_ready","skill":{"name":"kebab-case-name","description":"When to use this skill.","body":"# Title\\n\\n## Workflow\\n..."}}

Rules:
- Create only instruction-only Markdown skills.
- Do not include scripts, tools, assets, secrets, or commands that bypass safety rules.
- The skill name must match ^[a-z][a-z0-9-]{2,63}$.
- Keep the skill concise and focused on Minecraft server operations.
- The body should explain trigger conditions, workflow, evidence preferences, and output rules.
"""
