from __future__ import annotations

import hashlib
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from agent_for_mc.domain.errors import ServiceError


TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]+")
KEY_RE = re.compile(r"^[a-z][a-z0-9_]{1,63}$")

ALLOWED_MEMORY_TYPES = {"preference", "goal", "constraint", "fact"}
KIND_LABELS = {
    "preference": "偏好",
    "goal": "目标",
    "constraint": "约束",
    "fact": "事实",
}

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


@dataclass(slots=True)
class MemoryCandidate:
    kind: str
    content: str
    source_question: str
    source_answer: str
    confidence: float


@dataclass(slots=True)
class MemoryAction:
    action: str
    type: str
    key: str
    value: str = ""
    confidence: float = 0.0
    memory_id: int | None = None


@dataclass(slots=True)
class MemoryRecord:
    id: int
    scope_id: str
    kind: str
    key: str
    value: str
    content: str
    source_question: str
    source_answer: str
    confidence: float
    created_at: str
    updated_at: str
    hit_count: int

    @property
    def memory_type(self) -> str:
        return self.kind

    @property
    def rewritten_question(self) -> str:
        return self.key


class SQLiteMemoryStore:
    def __init__(self, db_path: Path, *, scope_id: str):
        self._db_path = Path(db_path)
        self._scope_id = str(scope_id).strip() or "default"
        self._initialized = False

    def initialize(self) -> None:
        if self._initialized:
            return

        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self._connect() as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA foreign_keys=ON")
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS memory_items (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        scope_id TEXT NOT NULL DEFAULT '',
                        kind TEXT NOT NULL,
                        key TEXT NOT NULL DEFAULT '',
                        value TEXT NOT NULL DEFAULT '',
                        content TEXT NOT NULL DEFAULT '',
                        normalized_content TEXT NOT NULL DEFAULT '',
                        normalized_kind TEXT NOT NULL DEFAULT '',
                        normalized_key TEXT NOT NULL DEFAULT '',
                        source_question TEXT NOT NULL DEFAULT '',
                        source_answer TEXT NOT NULL DEFAULT '',
                        confidence REAL NOT NULL DEFAULT 0.0,
                        created_at TEXT NOT NULL DEFAULT '',
                        updated_at TEXT NOT NULL DEFAULT '',
                        hit_count INTEGER NOT NULL DEFAULT 1
                    )
                    """
                )
                self._ensure_columns(conn)
                self._backfill_rows(conn)
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_memory_items_updated_at
                    ON memory_items(updated_at DESC)
                    """
                )
                conn.execute(
                    """
                    CREATE UNIQUE INDEX IF NOT EXISTS idx_memory_items_kind_key
                    ON memory_items(scope_id, normalized_kind, normalized_key)
                    """
                )
        except sqlite3.Error as exc:  # pragma: no cover - storage failure
            raise ServiceError(f"Memory SQLite 初始化失败: {exc}") from exc

        self._initialized = True

    def list_all(self) -> list[MemoryRecord]:
        self.initialize()
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    """
                    SELECT
                        id,
                        scope_id,
                        kind,
                        key,
                        value,
                        content,
                        source_question,
                        source_answer,
                        confidence,
                        created_at,
                        updated_at,
                        hit_count
                    FROM memory_items
                    WHERE scope_id = ?
                    ORDER BY updated_at DESC, confidence DESC, id DESC
                    """,
                    (self._scope_id,),
                ).fetchall()
        except sqlite3.Error as exc:  # pragma: no cover - storage failure
            raise ServiceError(f"Memory SQLite 读取失败: {exc}") from exc

        return [_row_to_record(row) for row in rows]

    def get_by_id(self, memory_id: int) -> MemoryRecord | None:
        self.initialize()
        try:
            with self._connect() as conn:
                row = conn.execute(
                    """
                    SELECT
                        id,
                        scope_id,
                        kind,
                        key,
                        value,
                        content,
                        source_question,
                        source_answer,
                        confidence,
                        created_at,
                        updated_at,
                        hit_count
                    FROM memory_items
                    WHERE id = ? AND scope_id = ?
                    """,
                    (memory_id, self._scope_id),
                ).fetchone()
        except sqlite3.Error as exc:  # pragma: no cover - storage failure
            raise ServiceError(f"Memory SQLite 读取失败: {exc}") from exc

        return _row_to_record(row) if row is not None else None

    def save_candidates(self, candidates: list[MemoryCandidate]) -> None:
        if not candidates:
            return

        self.initialize()
        now = _utc_now()
        try:
            with self._connect() as conn:
                for candidate in candidates:
                    normalized_kind = _normalize_text(candidate.kind)
                    if normalized_kind not in ALLOWED_MEMORY_TYPES:
                        continue

                    value = _extract_candidate_value(candidate.content)
                    if not value:
                        continue

                    key = _legacy_candidate_key(candidate)
                    content = _build_content(candidate.kind, key, value)
                    self._upsert_row(
                        conn,
                        kind=candidate.kind,
                        key=key,
                        value=value,
                        content=content,
                        confidence=candidate.confidence,
                        source_question=candidate.source_question,
                        source_answer=candidate.source_answer,
                        now=now,
                    )
        except sqlite3.Error as exc:  # pragma: no cover - storage failure
            raise ServiceError(f"Memory SQLite 写入失败: {exc}") from exc

    def apply_actions(
        self,
        actions: list[MemoryAction],
        *,
        source_question: str,
        source_answer: str,
    ) -> None:
        if not actions:
            return

        self.initialize()
        now = _utc_now()

        try:
            with self._connect() as conn:
                for action in actions:
                    if action.action == "delete":
                        self._delete_action(
                            conn,
                            action,
                        )
                        continue

                    content = _build_content(action.type, action.key, action.value)
                    if action.action == "add":
                        self._upsert_row(
                            conn,
                            kind=action.type,
                            key=action.key,
                            value=action.value,
                            content=content,
                            confidence=action.confidence,
                            source_question=source_question,
                            source_answer=source_answer,
                            now=now,
                        )
                        continue

                    if action.action == "update":
                        self._update_row(
                            conn,
                            action,
                            content=content,
                            source_question=source_question,
                            source_answer=source_answer,
                            now=now,
                        )
                        continue

                    raise ServiceError(f"未知的 memory action: {action.action}")
        except sqlite3.Error as exc:  # pragma: no cover - storage failure
            raise ServiceError(f"Memory SQLite 写入失败: {exc}") from exc

    def recall(self, query: str, *, limit: int) -> list[MemoryRecord]:
        self.initialize()
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    """
                    SELECT
                        id,
                        scope_id,
                        kind,
                        key,
                        value,
                        content,
                        source_question,
                        source_answer,
                        confidence,
                        created_at,
                        updated_at,
                        hit_count
                    FROM memory_items
                    WHERE scope_id = ?
                    """,
                    (self._scope_id,),
                ).fetchall()
        except sqlite3.Error as exc:  # pragma: no cover - storage failure
            raise ServiceError(f"Memory SQLite 读取失败: {exc}") from exc

        records = [_row_to_record(row) for row in rows]
        if not records:
            return []

        normalized_query = _normalize_text(query)
        if not normalized_query:
            records.sort(
                key=lambda record: (_parse_dt(record.updated_at), record.confidence),
                reverse=True,
            )
            return records[:limit]

        scored_records = [
            (
                _score_record(normalized_query, record),
                _parse_dt(record.updated_at),
                record,
            )
            for record in records
        ]
        scored_records.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [record for _, _, record in scored_records[:limit]]

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_columns(self, conn: sqlite3.Connection) -> None:
        columns = _table_columns(conn, "memory_items")
        additions = {
            "scope_id": "TEXT NOT NULL DEFAULT ''",
            "key": "TEXT NOT NULL DEFAULT ''",
            "value": "TEXT NOT NULL DEFAULT ''",
            "content": "TEXT NOT NULL DEFAULT ''",
            "normalized_content": "TEXT NOT NULL DEFAULT ''",
            "normalized_kind": "TEXT NOT NULL DEFAULT ''",
            "normalized_key": "TEXT NOT NULL DEFAULT ''",
        }
        for column, definition in additions.items():
            if column in columns:
                continue
            conn.execute(f"ALTER TABLE memory_items ADD COLUMN {column} {definition}")

    def _backfill_rows(self, conn: sqlite3.Connection) -> None:
        rows = conn.execute(
            """
            SELECT
                id,
                scope_id,
                kind,
                key,
                value,
                content,
                normalized_content,
                normalized_kind,
                normalized_key
            FROM memory_items
            """
        ).fetchall()
        for row in rows:
            memory_id = int(row["id"])
            scope_id = str(row["scope_id"] or "").strip()
            kind = str(row["kind"]).strip() or "fact"
            key = str(row["key"] or "").strip()
            value = str(row["value"] or "").strip()
            content = str(row["content"] or "").strip()
            normalized_content = str(row["normalized_content"] or "").strip()
            normalized_kind = str(row["normalized_kind"] or "").strip()
            normalized_key = str(row["normalized_key"] or "").strip()

            updates: dict[str, str] = {}
            if not scope_id:
                scope_id = self._scope_id
                updates["scope_id"] = scope_id
            if not key:
                key = _legacy_key(kind, content or value or str(memory_id), memory_id)
                updates["key"] = key
            if not value:
                value = _legacy_value(content)
                updates["value"] = value
            if not content:
                content = _build_content(kind, key, value)
                updates["content"] = content
            if not normalized_kind:
                normalized_kind = _normalize_text(kind)
                updates["normalized_kind"] = normalized_kind
            if not normalized_key:
                normalized_key = _normalize_text(key)
                updates["normalized_key"] = normalized_key
            if not normalized_content:
                normalized_content = _normalize_text(" ".join([kind, key, value, content]))
                updates["normalized_content"] = normalized_content

            if not updates:
                continue

            assignments = ", ".join(f"{column} = ?" for column in updates)
            parameters = [*updates.values(), memory_id]
            conn.execute(
                f"UPDATE memory_items SET {assignments} WHERE id = ?",
                parameters,
            )

    def _upsert_row(
        self,
        conn: sqlite3.Connection,
        *,
        kind: str,
        key: str,
        value: str,
        content: str,
        confidence: float,
        source_question: str,
        source_answer: str,
        now: datetime,
    ) -> None:
        normalized_kind = _normalize_text(kind)
        normalized_key = _normalize_text(key)
        normalized_value = _normalize_text(value)
        normalized_content = _normalize_text(" ".join([kind, key, value, content]))

        existing = conn.execute(
            """
            SELECT id, scope_id, value, confidence, hit_count
            FROM memory_items
            WHERE scope_id = ? AND normalized_kind = ? AND normalized_key = ?
            """,
            (self._scope_id, normalized_kind, normalized_key),
        ).fetchone()

        if existing is None:
            conn.execute(
                """
                INSERT INTO memory_items (
                    scope_id,
                    kind,
                    key,
                    value,
                    content,
                    normalized_content,
                    normalized_kind,
                    normalized_key,
                    source_question,
                    source_answer,
                    confidence,
                    created_at,
                    updated_at,
                    hit_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self._scope_id,
                    kind,
                    key,
                    value,
                    content,
                    normalized_content,
                    normalized_kind,
                    normalized_key,
                    source_question,
                    source_answer,
                    confidence,
                    now.isoformat(),
                    now.isoformat(),
                    1,
                ),
            )
            return

        existing_value = _normalize_text(str(existing["value"]))
        if existing_value and existing_value != normalized_value:
            raise ServiceError(
                f"Memory 已存在相同 type/key 的不同 value: {kind}/{key}"
            )

        updated_confidence = max(float(existing["confidence"]), confidence)
        conn.execute(
            """
            UPDATE memory_items
            SET
                scope_id = ?,
                kind = ?,
                key = ?,
                value = ?,
                content = ?,
                normalized_content = ?,
                normalized_kind = ?,
                normalized_key = ?,
                source_question = ?,
                source_answer = ?,
                confidence = ?,
                updated_at = ?,
                hit_count = ?
            WHERE id = ?
            """,
            (
                self._scope_id,
                kind,
                key,
                value,
                content,
                normalized_content,
                normalized_kind,
                normalized_key,
                source_question,
                source_answer,
                updated_confidence,
                now.isoformat(),
                int(existing["hit_count"]) + 1,
                int(existing["id"]),
            ),
        )

    def _update_row(
        self,
        conn: sqlite3.Connection,
        action: MemoryAction,
        *,
        content: str,
        source_question: str,
        source_answer: str,
        now: datetime,
    ) -> None:
        if action.memory_id is None:
            raise ServiceError("update action 缺少 memory_id")

        row = conn.execute(
            """
            SELECT id, kind, key, value, confidence, hit_count
            FROM memory_items
            WHERE id = ? AND scope_id = ?
            """,
            (action.memory_id, self._scope_id),
        ).fetchone()
        if row is None:
            raise ServiceError(f"update action 指向不存在的 memory_id: {action.memory_id}")

        existing_kind = str(row["kind"])
        existing_key = str(row["key"])
        if _normalize_text(existing_kind) != _normalize_text(action.type):
            raise ServiceError(
                f"update action 的 type 与 memory_id={action.memory_id} 不匹配"
            )
        if _normalize_text(existing_key) != _normalize_text(action.key):
            raise ServiceError(
                f"update action 的 key 与 memory_id={action.memory_id} 不匹配"
            )
        if not _is_nonempty_text(action.value):
            raise ServiceError("update action 的 value 不能为空")

        normalized_content = _normalize_text(
            " ".join([action.type, action.key, action.value, content])
        )
        updated_confidence = max(float(row["confidence"]), action.confidence)
        conn.execute(
            """
            UPDATE memory_items
            SET
                kind = ?,
                key = ?,
                value = ?,
                content = ?,
                normalized_content = ?,
                normalized_kind = ?,
                normalized_key = ?,
                source_question = ?,
                source_answer = ?,
                confidence = ?,
                updated_at = ?,
                hit_count = ?
            WHERE id = ?
            """,
            (
                action.type,
                action.key,
                action.value,
                content,
                normalized_content,
                _normalize_text(action.type),
                _normalize_text(action.key),
                source_question,
                source_answer,
                updated_confidence,
                now.isoformat(),
                int(row["hit_count"]) + 1,
                int(row["id"]),
            ),
        )

    def _delete_action(self, conn: sqlite3.Connection, action: MemoryAction) -> None:
        if action.memory_id is None:
            raise ServiceError("delete action 缺少 memory_id")

        row = conn.execute(
            """
            SELECT id, kind, key
            FROM memory_items
            WHERE id = ? AND scope_id = ?
            """,
            (action.memory_id, self._scope_id),
        ).fetchone()
        if row is None:
            raise ServiceError(f"delete action 指向不存在的 memory_id: {action.memory_id}")

        if _normalize_text(str(row["kind"])) != _normalize_text(action.type):
            raise ServiceError(
                f"delete action 的 type 与 memory_id={action.memory_id} 不匹配"
            )
        if _normalize_text(str(row["key"])) != _normalize_text(action.key):
            raise ServiceError(
                f"delete action 的 key 与 memory_id={action.memory_id} 不匹配"
            )

        conn.execute("DELETE FROM memory_items WHERE id = ?", (action.memory_id,))


def _row_to_record(row: sqlite3.Row) -> MemoryRecord:
    return MemoryRecord(
        id=int(row["id"]),
        scope_id=str(row["scope_id"]),
        kind=str(row["kind"]),
        key=str(row["key"]),
        value=str(row["value"]),
        content=str(row["content"]),
        source_question=str(row["source_question"]),
        source_answer=str(row["source_answer"]),
        confidence=float(row["confidence"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
        hit_count=int(row["hit_count"]),
    )


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


def _table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {str(row[1]) for row in rows}


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


def _legacy_key(kind: str, content: str, memory_id: int) -> str:
    digest = hashlib.sha1(f"{kind}|{content}|{memory_id}".encode("utf-8")).hexdigest()[:12]
    return f"legacy_{_normalize_text(kind) or 'memory'}_{digest}"


def _legacy_value(content: str) -> str:
    content = str(content).strip()
    if not content:
        return ""
    for separator in ("：", ":"):
        if separator in content:
            _, value = content.split(separator, 1)
            return value.strip()
    return content


def _extract_candidate_value(content: str) -> str:
    return _legacy_value(content)


def _normalize_text(value: str) -> str:
    return " ".join(str(value).strip().lower().split())


def _tokenize(value: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(value)]


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


def _is_nonempty_text(value: str) -> bool:
    return bool(str(value).strip())
