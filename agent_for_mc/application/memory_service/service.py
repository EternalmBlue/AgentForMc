from __future__ import annotations

import hashlib
import json
import logging
import re
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import Any

from agent_for_mc.infrastructure.config import Settings
from agent_for_mc.infrastructure.observability import record_counter, trace_operation
from agent_for_mc.infrastructure.memory_store import (
    ALLOWED_MEMORY_TYPES,
    KIND_LABELS,
    MemoryAction,
    MemoryCandidate,
    MemoryRecord,
    SQLiteMemoryStore,
)


LOGGER = logging.getLogger(__name__)

KEY_RE = re.compile(r"^[a-z][a-z0-9_]{1,63}$")

_RESERVED_KEYS = {
    "action",
    "confidence",
    "content",
    "id",
    "key",
    "memory_id",
    "source_answer",
    "source_question",
    "type",
    "value",
}

_MEMORY_PATTERNS: list[tuple[str, float, str, re.Pattern[str]]] = [
    (
        "preference",
        0.96,
        "偏好",
        re.compile(
            r"(?:\bI\s+prefer\b|\bprefer\b|\bI\s+like\b|\bI\s+want\b|"
            r"我偏好|我喜欢|我想要|请记住我偏好)\s+(?P<value>.+?)(?:[。.!?,，；;]|$)",
            re.IGNORECASE,
        ),
    ),
    (
        "fact",
        0.92,
        "事实",
        re.compile(
            r"(?:\bmy\s+project\s+uses\b|\bwe\s+use\b|\bI\s+use\b|"
            r"我的项目使用|我们使用|我在用)\s+(?P<value>.+?)(?:[。.!?,，；;]|$)",
            re.IGNORECASE,
        ),
    ),
    (
        "constraint",
        0.9,
        "约束",
        re.compile(
            r"(?:\bmust\b|\bshould\b|\bneed\b|必须|需要|请始终)\s+(?P<value>.+?)(?:[。.!?,，；;]|$)",
            re.IGNORECASE,
        ),
    ),
    (
        "goal",
        0.9,
        "目标",
        re.compile(
            r"(?:\bmy\s+goal\s+is\b|\bgoal\b|我的目标是|我想学会|我想实现)\s+(?P<value>.+?)(?:[。.!?,，；;]|$)",
            re.IGNORECASE,
        ),
    ),
]


@dataclass(slots=True)
class SessionTurn:
    question: str
    answer: str


@dataclass(slots=True)
class MemoryMaintenanceResult:
    session_summary: str
    actions: list[MemoryAction] = field(default_factory=list)


@dataclass(slots=True)
class MemoryMaintenanceRunner:
    agent: Any

    def run(
        self,
        turns: list[SessionTurn],
        memories: list[MemoryRecord],
    ) -> MemoryMaintenanceResult:
        with trace_operation(
            "memory_maintenance.run",
            attributes={"component": "memory_maintenance", "turn.count": len(turns)},
            metric_name="rag_memory_maintenance_seconds",
        ):
            record_counter("rag_memory_maintenance_requests_total")
            payload = _render_memory_maintenance_prompt(turns, memories)
            raw = _invoke_memory_maintenance_agent(
                self.agent,
                system_prompt=_MEMORY_MAINTENANCE_SYSTEM_PROMPT,
                payload=payload,
            )
            try:
                return _parse_memory_maintenance_result(raw)
            except Exception as exc:
                repair_raw = _invoke_memory_maintenance_agent(
                    self.agent,
                    system_prompt=_MEMORY_MAINTENANCE_REPAIR_PROMPT,
                    payload=_render_repair_prompt(
                        task_name="memory maintenance",
                        previous_output=raw,
                        error_message=str(exc),
                        expected_schema=(
                            '{"session_summary":"...","actions":['
                            '{"action":"add","type":"preference","key":"gradle_style",'
                            '"value":"kotlin_dsl","confidence":0.94}]}'
                        ),
                    ),
                )
                return _parse_memory_maintenance_result(repair_raw)


@dataclass(slots=True)
class MemoryService:
    store: SQLiteMemoryStore
    recall_limit: int
    min_confidence: float
    consolidation_turns: int
    maintenance_runner: MemoryMaintenanceRunner
    _turn_ledger: list[SessionTurn] = field(default_factory=list)
    _next_consolidation_turn: int = field(init=False, repr=False)
    _executor: ThreadPoolExecutor = field(init=False, repr=False)
    _lock: Lock = field(init=False, repr=False)
    _inflight: Future[None] | None = field(default=None, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="memory-consolidation",
        )
        self._lock = Lock()
        self._next_consolidation_turn = max(1, int(self.consolidation_turns))

    def recall(self, question: str, *, history_text: str = "") -> list[MemoryRecord]:
        with trace_operation(
            "memory.recall",
            attributes={"component": "memory", "query.length": len(question.strip())},
            metric_name="rag_memory_recall_seconds",
        ):
            record_counter("rag_memory_recall_requests_total")
            query = _combine_query(question, history_text)
            return self.store.recall(query, limit=self.recall_limit)

    def observe_turn(self, question: str, answer: str) -> None:
        self.record_turn(question, answer)

    def record_turn(self, question: str, answer: str) -> None:
        if self._closed:
            return

        with trace_operation(
            "memory.record_turn",
            attributes={"component": "memory", "question.length": len(question.strip())},
            metric_name="rag_memory_record_turn_seconds",
        ):
            record_counter("rag_memory_record_turn_requests_total")
            question_text = question.strip()
            answer_text = answer.strip()
            if not question_text and not answer_text:
                return

            with self._lock:
                self._turn_ledger.append(
                    SessionTurn(question=question_text, answer=answer_text)
                )
                self._maybe_schedule_consolidation_locked()

    def wait_for_idle(self, timeout: float | None = None) -> None:
        remaining = timeout
        while True:
            with self._lock:
                future = self._inflight
            if future is None:
                return
            try:
                future.result(timeout=remaining)
            except Exception:
                return

    def close(self, *, wait: bool = True) -> None:
        with self._lock:
            self._closed = True
        if wait:
            self.wait_for_idle()
        self._executor.shutdown(wait=wait, cancel_futures=False)

    def _maybe_schedule_consolidation_locked(self) -> None:
        if self._closed or self.consolidation_turns <= 0:
            return
        if len(self._turn_ledger) < self._next_consolidation_turn:
            return
        if self._inflight is not None and not self._inflight.done():
            return

        snapshot = list(self._turn_ledger)
        self._next_consolidation_turn += self.consolidation_turns
        future = self._executor.submit(self._run_consolidation, snapshot)
        self._inflight = future
        future.add_done_callback(self._on_consolidation_done)

    def _on_consolidation_done(self, future: Future[None]) -> None:
        try:
            future.result()
        except Exception as exc:  # pragma: no cover - background failure
            LOGGER.warning("Memory consolidation worker failed: %s", exc, exc_info=True)

        with self._lock:
            if self._inflight is future:
                self._inflight = None
            if self._closed:
                return
            self._maybe_schedule_consolidation_locked()

    def _run_consolidation(self, turns: list[SessionTurn]) -> None:
        with trace_operation(
            "memory.consolidation",
            attributes={"component": "memory", "turn.count": len(turns)},
            metric_name="rag_memory_consolidation_seconds",
        ):
            try:
                existing_memories = self.store.list_all()
                result = self.maintenance_runner.run(turns, existing_memories)
                actions = result.actions
                validated_actions = validate_memory_actions(actions, existing_memories)
                validated_actions = [
                    action
                    for action in validated_actions
                    if action.action == "delete" or action.confidence >= self.min_confidence
                ]
                if not validated_actions:
                    return
                self.store.apply_actions(
                    validated_actions,
                    source_question=result.session_summary or "session_summary",
                    source_answer=_render_turn_ledger(turns),
                )
            except Exception as exc:  # pragma: no cover - background failure
                LOGGER.warning("Memory consolidation failed: %s", exc, exc_info=True)


def build_memory_service(
    settings: Settings,
    *,
    scope_id: str,
    maintenance_agent: Any | None = None,
) -> MemoryService | None:
    if not settings.memory_enabled:
        return None

    store = SQLiteMemoryStore(settings.memory_db_path, scope_id=scope_id)
    store.initialize()
    if maintenance_agent is None:
        from agent_for_mc.interfaces.deepagent import build_memory_maintenance_agent

        maintenance_agent = build_memory_maintenance_agent(settings=settings)
    if maintenance_agent is None:
        return None
    return MemoryService(
        store=store,
        recall_limit=settings.memory_recall_limit,
        min_confidence=settings.memory_min_confidence,
        consolidation_turns=settings.memory_consolidation_turns,
        maintenance_runner=MemoryMaintenanceRunner(agent=maintenance_agent),
    )


def extract_memory_candidates(question: str, answer: str = "") -> list[MemoryCandidate]:
    normalized_question = " ".join(question.strip().split())
    if not normalized_question:
        return []

    candidates: list[MemoryCandidate] = []
    for kind, confidence, label, pattern in _MEMORY_PATTERNS:
        match = pattern.search(normalized_question)
        if match is None:
            continue

        value = _clean_value(match.group("value"))
        if not _is_stable_value(value):
            continue

        for split_value in _split_compound_value(value):
            if not _is_stable_value(split_value):
                continue
            if kind == "preference" and _looks_like_fact_clause(split_value):
                continue
            candidates.append(
                MemoryCandidate(
                    kind=kind,
                    content=f"{label}：{split_value}",
                    source_question=question.strip(),
                    source_answer=answer.strip(),
                    confidence=confidence,
                )
            )

    return _dedupe_candidates(candidates)


def format_memory_context(memories: list[MemoryRecord]) -> str:
    if not memories:
        return ""

    parts: list[str] = []
    for index, memory in enumerate(memories, start=1):
        parts.append(
            f"[记忆 {index}] {memory.content} "
            f"(type={memory.kind}, key={memory.key}, id={memory.id}, "
            f"confidence={memory.confidence:.2f})"
        )
    return "\n".join(parts)


def validate_memory_actions(
    actions: list[MemoryAction],
    existing_memories: list[MemoryRecord],
) -> list[MemoryAction]:
    validated: list[MemoryAction] = []
    seen: set[tuple[Any, ...]] = set()
    working_by_id = {record.id: record for record in existing_memories}
    working_by_type_key = {
        (_normalize_text(record.kind), _normalize_text(record.key)): record
        for record in existing_memories
    }

    for raw_action in actions:
        action = _normalize_action(raw_action)
        canonical = _canonical_action(action)
        if canonical in seen:
            continue
        seen.add(canonical)

        _validate_action_shape(action)

        if action.action == "add":
            key = (_normalize_text(action.type), _normalize_text(action.key))
            existing = working_by_type_key.get(key)
            if existing is not None:
                if _normalize_text(existing.value) != _normalize_text(action.value):
                    raise ValueError(
                        f"重复 memory key 冲突: {action.type}/{action.key}"
                    )
            else:
                placeholder = MemoryRecord(
                    id=-1,
                    scope_id="",
                    kind=action.type,
                    key=action.key,
                    value=action.value,
                    content=_build_content(action.type, action.key, action.value),
                    source_question="",
                    source_answer="",
                    confidence=action.confidence,
                    created_at="",
                    updated_at="",
                    hit_count=1,
                )
                working_by_type_key[key] = placeholder
            validated.append(action)
            continue

        if action.action in {"update", "delete"}:
            if action.memory_id is None:
                raise ValueError(f"{action.action} action 缺少 memory_id")
            existing = working_by_id.get(int(action.memory_id))
            if existing is None:
                raise ValueError(f"{action.action} action 指向不存在的 memory_id")

            if _normalize_text(existing.kind) != _normalize_text(action.type):
                raise ValueError(
                    f"{action.action} action 的 type 与 memory_id 不匹配"
                )
            if _normalize_text(existing.key) != _normalize_text(action.key):
                raise ValueError(
                    f"{action.action} action 的 key 与 memory_id 不匹配"
                )

            if action.action == "update" and not _is_nonempty_text(action.value):
                raise ValueError("update action 的 value 不能为空")

            if action.action == "delete":
                validated.append(action)
                working_by_id.pop(int(action.memory_id), None)
                working_by_type_key.pop(
                    (_normalize_text(action.type), _normalize_text(action.key)),
                    None,
                )
                continue

            validated.append(action)
            updated_record = MemoryRecord(
                id=existing.id,
                scope_id=existing.scope_id,
                kind=action.type,
                key=action.key,
                value=action.value,
                content=_build_content(action.type, action.key, action.value),
                source_question=existing.source_question,
                source_answer=existing.source_answer,
                confidence=max(existing.confidence, action.confidence),
                created_at=existing.created_at,
                updated_at=existing.updated_at,
                hit_count=existing.hit_count,
            )
            working_by_id[existing.id] = updated_record
            working_by_type_key[(
                _normalize_text(action.type),
                _normalize_text(action.key),
            )] = updated_record
            continue

        raise ValueError(f"未知的 memory action: {action.action}")

    return validated


def _normalize_action(action: MemoryAction) -> MemoryAction:
    action_name = str(action.action).strip().lower()
    memory_type = str(action.type).strip().lower()
    key = str(action.key).strip()
    value = str(action.value).strip()
    memory_id = int(action.memory_id) if action.memory_id is not None else None
    confidence = float(action.confidence)
    return MemoryAction(
        action=action_name,
        type=memory_type,
        key=key,
        value=value,
        confidence=confidence,
        memory_id=memory_id,
    )


def _validate_action_shape(action: MemoryAction) -> None:
    if action.action not in {"add", "update", "delete"}:
        raise ValueError(f"非法的 memory action: {action.action}")
    if action.type not in ALLOWED_MEMORY_TYPES:
        raise ValueError(f"非法的 memory type: {action.type}")
    if not KEY_RE.fullmatch(action.key):
        raise ValueError(f"非法的 memory key: {action.key}")
    if action.key in _RESERVED_KEYS:
        raise ValueError(f"memory key 不能使用保留字段: {action.key}")
    if action.action in {"add", "update"} and not _is_nonempty_text(action.value):
        raise ValueError(f"{action.action} action 的 value 不能为空")
    if not (0.0 <= action.confidence <= 1.0):
        raise ValueError("confidence 必须在 0 到 1 之间")


def _canonical_action(action: MemoryAction) -> tuple[Any, ...]:
    if action.action == "add":
        return (action.action, action.type, action.key, action.value)
    return (action.action, action.memory_id, action.type, action.key, action.value)


def _render_memory_maintenance_prompt(
    turns: list[SessionTurn],
    memories: list[MemoryRecord],
) -> str:
    turn_lines: list[str] = [
        'You are memory_maintenance_agent. Produce JSON only.',
        'Summarize the session and reconcile it with existing long-term memory.',
    ]
    for index, turn in enumerate(turns, start=1):
        turn_lines.append(f'[Turn {index}] User: {turn.question}')
        turn_lines.append(f'[Turn {index}] Assistant: {turn.answer}')
    turn_lines.append('')
    turn_lines.append(
        'Return format: {"session_summary":"...","actions":[{"action":"add",'
        '"type":"preference","key":"gradle_style","value":"kotlin_dsl",'
        '"confidence":0.94}]}'
    )
    turn_lines.append('')
    turn_lines.append(
        'Rules: type must be one of preference / goal / constraint / fact; '
        'key must be snake_case; add/update must include a non-empty value; '
        'update/delete must point to an existing memory_id; '
        'identical (type, key, value) entries are duplicates; '
        'different values for the same (type, key) must be handled by update or delete+add.'
    )
    turn_lines.append('')
    turn_lines.append('Existing memory:')
    turn_lines.append(
        json.dumps(
            [
                {
                    'id': record.id,
                    'type': record.kind,
                    'key': record.key,
                    'value': record.value,
                    'confidence': record.confidence,
                    'hit_count': record.hit_count,
                }
                for record in memories
            ],
            ensure_ascii=False,
            indent=2,
        )
    )
    return "\n".join(turn_lines)


def _render_repair_prompt(
    *,
    task_name: str,
    previous_output: str,
    error_message: str,
    expected_schema: str,
) -> str:
    return (
        f"Previous {task_name} output was invalid.\n"
        f"Error: {error_message}\n"
        f"Original output: {previous_output}\n\n"
        f"Return valid JSON only. Expected schema example: {expected_schema}"
    )


def _invoke_memory_maintenance_agent(
    agent: Any,
    *,
    system_prompt: str,
    payload: str,
) -> str:
    state = agent.invoke(
        {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": payload},
            ]
        }
    )
    text = _extract_agent_text(state)
    if not text:
        raise ValueError("memory maintenance agent returned empty output")
    return text


def _extract_agent_text(state: object) -> str:
    if isinstance(state, str):
        return state.strip()

    if isinstance(state, dict):
        messages = state.get("messages")
        if isinstance(messages, list):
            for message in reversed(messages):
                content = getattr(message, "content", None)
                if isinstance(content, str) and content.strip():
                    return content.strip()
                if isinstance(message, dict):
                    content = message.get("content")
                    if isinstance(content, str) and content.strip():
                        return content.strip()

        output = state.get("output")
        if isinstance(output, str):
            return output.strip()

    return ""


def _render_turn_ledger(turns: list[SessionTurn]) -> str:
    lines: list[str] = []
    for index, turn in enumerate(turns, start=1):
        lines.append(f"[Turn {index}] User: {turn.question}")
        lines.append(f"[Turn {index}] Assistant: {turn.answer}")
    return "\n".join(lines)


def _parse_memory_maintenance_result(raw: str) -> MemoryMaintenanceResult:
    data = _load_json_payload(raw)
    session_summary = str(data.get("session_summary", "")).strip()
    if not session_summary:
        raise ValueError("session_summary 不能为空")

    actions: list[MemoryAction] = []
    for item in _ensure_list(data.get("actions", [])):
        parsed = _parse_memory_action(item)
        if parsed is not None:
            actions.append(parsed)

    return MemoryMaintenanceResult(session_summary=session_summary, actions=actions)


def _parse_memory_action(item: Any) -> MemoryAction | None:
    if not isinstance(item, dict):
        return None
    action = str(item.get("action", "")).strip().lower()
    memory_type = str(item.get("type", "")).strip().lower()
    key = str(item.get("key", "")).strip()
    value = str(item.get("value", "")).strip()
    confidence = float(item.get("confidence", 0.0) or 0.0)
    memory_id_value = item.get("memory_id")
    memory_id = None
    if memory_id_value not in (None, "", "null"):
        memory_id = int(memory_id_value)
    return MemoryAction(
        action=action,
        type=memory_type,
        key=key,
        value=value,
        confidence=confidence,
        memory_id=memory_id,
    )


def _load_json_payload(raw: str) -> dict[str, Any]:
    normalized = _extract_json_blob(raw)
    data = json.loads(normalized)
    if not isinstance(data, dict):
        raise ValueError("JSON 顶层必须是对象")
    return data


def _extract_json_blob(raw: str) -> str:
    text = str(raw).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"\s*```$", "", text).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start : end + 1]
    return text


def _ensure_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return []


def _combine_query(question: str, history_text: str) -> str:
    parts = [question.strip(), history_text.strip()]
    return "\n".join(part for part in parts if part)


def _dedupe_candidates(candidates: list[MemoryCandidate]) -> list[MemoryCandidate]:
    unique: list[MemoryCandidate] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = _normalize_text(candidate.content)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique.append(candidate)
    return unique


def _clean_value(value: str) -> str:
    cleaned = " ".join(value.strip().split())
    cleaned = cleaned.strip("。！？,.;；")
    cleaned = cleaned.strip("\"'“”‘’")
    cleaned = re.sub(
        r"^(?:that|to|the|a|an)\s+",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    return cleaned.strip()


def _split_compound_value(value: str) -> list[str]:
    parts = [
        part.strip()
        for part in re.split(
            r"\s+\b(?:and|then|also)\b\s+|\s+以及\s+|\s+并且\s+|\s+同时\s+",
            value,
            flags=re.IGNORECASE,
        )
        if part.strip()
    ]
    return parts or [value]


def _is_stable_value(value: str) -> bool:
    if not value:
        return False
    lowered = value.lower().strip()
    if lowered in {"it", "this", "that", "something", "anything", "anything else"}:
        return False
    if len(value) < 2:
        return False
    return True


def _looks_like_fact_clause(value: str) -> bool:
    lowered = value.lower()
    return any(
        phrase in lowered
        for phrase in (
            "my project uses",
            "i'm using",
            "i am using",
            "we use",
            "我在用",
            "我现在用",
            "我们使用",
        )
    )


def _is_nonempty_text(value: str) -> bool:
    return bool(str(value).strip())


def _normalize_text(value: str) -> str:
    return " ".join(str(value).strip().lower().split())


def _build_content(kind: str, key: str, value: str) -> str:
    label = KIND_LABELS.get(_normalize_text(kind), _normalize_text(kind) or "memory")
    return f"{label}：{value}" if value else f"{label}："


def _legacy_candidate_key(candidate: MemoryCandidate) -> str:
    raw = "|".join(
        [
            candidate.kind,
            candidate.content,
            candidate.source_question,
            candidate.source_answer,
        ]
    )
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"legacy_{_normalize_text(candidate.kind) or 'memory'}_{digest}"


def _legacy_value(content: str) -> str:
    content = str(content).strip()
    if not content:
        return ""
    for separator in ("：", ":"):
        if separator in content:
            _, value = content.split(separator, 1)
            return value.strip()
    return content


def _score_record(query: str, record: MemoryRecord) -> float:
    query_tokens = set(_tokenize(query))
    content_tokens = set(
        _tokenize(
            " ".join(
                [
                    record.kind,
                    record.key,
                    record.value,
                    record.content,
                    record.source_question,
                    record.source_answer,
                ]
            )
        )
    )

    overlap = len(query_tokens & content_tokens)
    normalized_content = _normalize_text(
        " ".join([record.kind, record.key, record.value, record.content])
    )
    normalized_query = _normalize_text(query)
    phrase_bonus = 2.5 if normalized_query and normalized_query in normalized_content else 0.0
    confidence_bonus = record.confidence * 0.5
    hit_bonus = min(record.hit_count, 5) * 0.1
    kind_bonus = {
        "preference": 0.25,
        "constraint": 0.2,
        "fact": 0.15,
        "goal": 0.1,
    }.get(_normalize_text(record.kind), 0.0)
    recency_bonus = 1.0 / (1.0 + max((_utc_now() - _parse_dt(record.updated_at)).days, 0))
    return (
        overlap * 2.0
        + phrase_bonus
        + confidence_bonus
        + hit_bonus
        + kind_bonus
        + recency_bonus
    )


def _tokenize(value: str) -> list[str]:
    return [token.lower() for token in re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]+", value)]


def _parse_dt(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return datetime(1970, 1, 1, tzinfo=timezone.utc)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


_MEMORY_MAINTENANCE_SYSTEM_PROMPT = """You are memory_maintenance_agent.
Your task is to produce one JSON object with exactly two top-level keys:
session_summary and actions.
The summary should capture stable preferences, goals, constraints, and facts.
The actions array should contain memory add/update/delete operations only.
Do not output any extra prose or markdown.
"""


_MEMORY_MAINTENANCE_REPAIR_PROMPT = """You are memory_maintenance_agent.
The previous output was invalid.
Return only valid JSON with exactly two top-level keys:
session_summary and actions.
"""
