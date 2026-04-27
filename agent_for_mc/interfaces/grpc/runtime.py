from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from threading import Lock
from time import time
from typing import Callable, Iterable
from uuid import uuid4

from agent_for_mc.application.chat_session import RagChatSession
from agent_for_mc.application.plugin_semantic_agent.file_rules import (
    is_semantic_indexable_relative_path,
    normalize_allowed_config_relative_path,
)
from agent_for_mc.application.plugin_semantic_agent import (
    RefreshProgressSnapshot,
    build_plugin_semantic_service,
)
from agent_for_mc.domain.errors import RagForMcError
from agent_for_mc.domain.models import AnswerResult, RetrievedDoc
from agent_for_mc.infrastructure.config import Settings
from agent_for_mc.infrastructure.observability import (
    configure_observability,
    record_counter,
    trace_operation,
)
from agent_for_mc.infrastructure.ranker import BceRanker
from agent_for_mc.infrastructure.vector_store import LancePluginVectorStore
from agent_for_mc.interfaces.session_factory import build_session


class BridgeRuntimeError(Exception):
    """Base error for backend bridge runtime failures."""


class InvalidRequestError(BridgeRuntimeError):
    """Raised when a caller sends invalid data."""


class NotFoundError(BridgeRuntimeError):
    """Raised when a referenced sync or resource does not exist."""


class ConflictError(BridgeRuntimeError):
    """Raised when the requested mutation conflicts with current state."""


class FailedPreconditionError(BridgeRuntimeError):
    """Raised when the request is well-formed but violates current state."""


class SyncState(str, Enum):
    PENDING = "pending"
    UPLOADING = "uploading"
    INDEXING = "indexing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ServerPlugin:
    name: str
    version: str
    enabled: bool


@dataclass(frozen=True, slots=True)
class AskCommand:
    server_id: str
    server_instance_id: str
    player_id: str
    player_name: str
    question: str
    request_id: str
    timestamp_ms: int
    installed_plugins: list[ServerPlugin] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class AskReply:
    request_id: str
    answer: str
    citations_summary: str
    backend_trace_id: str


@dataclass(frozen=True, slots=True)
class ManifestEntry:
    relative_path: str
    size: int
    sha256: str
    last_modified_epoch_ms: int


@dataclass(frozen=True, slots=True)
class RejectedPathEntry:
    relative_path: str
    reason: str


@dataclass(frozen=True, slots=True)
class PrepareSyncReply:
    sync_id: str
    required_paths: list[str]
    rejected_paths: list[RejectedPathEntry]


@dataclass(frozen=True, slots=True)
class UploadChunk:
    sync_id: str
    relative_path: str
    chunk_index: int
    total_chunks: int
    content_bytes: bytes
    sha256: str


@dataclass(frozen=True, slots=True)
class UploadReply:
    sync_id: str
    relative_path: str
    received_bytes: int
    received_chunks: int
    sha256_verified: bool
    message: str


@dataclass(frozen=True, slots=True)
class CommitSyncReply:
    sync_id: str
    accepted_count: int
    indexed_count: int
    refresh_started: bool
    message: str


@dataclass(frozen=True, slots=True)
class SyncStatusSnapshot:
    sync_id: str
    state: SyncState
    accepted_count: int
    indexed_count: int
    refresh_started: bool
    message: str
    updated_at_epoch_ms: int
    required_file_count: int
    uploaded_file_count: int
    total_upload_bytes: int
    uploaded_bytes: int
    current_upload_path: str
    refresh_total_bundles: int
    refresh_completed_bundles: int
    refresh_failed_bundles: int
    current_refresh_bundle: str
    current_refresh_phase: str


@dataclass(slots=True)
class _SessionHandle:
    session: RagChatSession
    lock: Lock = field(default_factory=Lock, repr=False)
    last_used_epoch_ms: int = field(default_factory=lambda: _now_ms())


class SessionRegistry:
    def __init__(self, *, factory: Callable[[str], RagChatSession], ttl_seconds: int):
        self._factory = factory
        self._ttl_ms = ttl_seconds * 1000
        self._lock = Lock()
        self._handles: dict[str, _SessionHandle] = {}

    def ask(
        self,
        memory_scope_id: str,
        question: str,
        *,
        server_plugins: list[str] | None = None,
    ) -> AnswerResult:
        handle = self._get_or_create(memory_scope_id)
        with handle.lock:
            handle.last_used_epoch_ms = _now_ms()
            return handle.session.ask(question, server_plugins=server_plugins)

    def close(self) -> None:
        with self._lock:
            handles = list(self._handles.values())
            self._handles.clear()
        for handle in handles:
            handle.session.close()

    def _get_or_create(self, memory_scope_id: str) -> _SessionHandle:
        with self._lock:
            self._reap_expired_locked()
            handle = self._handles.get(memory_scope_id)
            if handle is None:
                handle = _SessionHandle(session=self._factory(memory_scope_id))
                self._handles[memory_scope_id] = handle
            handle.last_used_epoch_ms = _now_ms()
            return handle

    def _reap_expired_locked(self) -> None:
        if self._ttl_ms <= 0:
            return
        now_ms = _now_ms()
        expired_keys = [
            key
            for key, handle in self._handles.items()
            if now_ms - handle.last_used_epoch_ms > self._ttl_ms
        ]
        for key in expired_keys:
            handle = self._handles.pop(key, None)
            if handle is not None:
                handle.session.close()


class ServerInstanceRegistry:
    def __init__(self, bindings_path: Path):
        self._bindings_path = bindings_path
        self._lock = Lock()

    def validate_or_bind(self, *, server_id: str, server_instance_id: str) -> tuple[str, str]:
        normalized_server_id = _validate_identifier(server_id, field_name="server_id")
        normalized_instance_id = _validate_identifier(
            server_instance_id,
            field_name="server_instance_id",
        )

        with self._lock:
            data = self._load_locked()
            bindings = data.setdefault("bindings", {})
            if not isinstance(bindings, dict):
                raise BridgeRuntimeError("server instance bindings file has invalid bindings format")

            existing = bindings.get(normalized_server_id)
            if isinstance(existing, dict):
                existing_instance_id = str(
                    existing.get("server_instance_id") or existing.get("instance_id") or ""
                ).strip()
                if existing_instance_id and existing_instance_id != normalized_instance_id:
                    raise FailedPreconditionError(
                        "server.id conflict: "
                        f"{normalized_server_id} is already bound to a different Minecraft server instance. "
                        "Use a unique server.id for this server, or delete the stale backend binding "
                        "after confirming the old server no longer uses it."
                    )
                if existing_instance_id == normalized_instance_id:
                    return normalized_server_id, normalized_instance_id

            bindings[normalized_server_id] = {
                "server_instance_id": normalized_instance_id,
                "updated_at_epoch_ms": _now_ms(),
            }
            data["version"] = 1
            self._write_locked(data)

        return normalized_server_id, normalized_instance_id

    def _load_locked(self) -> dict:
        if not self._bindings_path.exists():
            return {"version": 1, "bindings": {}}
        try:
            with self._bindings_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise BridgeRuntimeError(
                f"could not read server instance bindings: {self._bindings_path}"
            ) from exc
        if not isinstance(data, dict):
            raise BridgeRuntimeError("server instance bindings file must contain a JSON object")
        return data

    def _write_locked(self, data: dict) -> None:
        self._bindings_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self._bindings_path.with_name(f"{self._bindings_path.name}.{uuid4().hex}.tmp")
        try:
            with temp_path.open("w", encoding="utf-8") as handle:
                json.dump(data, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
            os.replace(temp_path, self._bindings_path)
        except OSError as exc:
            raise BridgeRuntimeError(
                f"could not write server instance bindings: {self._bindings_path}"
            ) from exc
        finally:
            temp_path.unlink(missing_ok=True)


@dataclass(slots=True)
class _SyncOperation:
    sync_id: str
    server_id: str
    server_instance_id: str
    manifest: dict[str, ManifestEntry]
    required_paths: set[str]
    rejected_paths: list[RejectedPathEntry]
    received_paths: set[str] = field(default_factory=set)
    state: SyncState = SyncState.PENDING
    accepted_count: int = 0
    indexed_count: int = 0
    refresh_started: bool = False
    message: str = ""
    updated_at_epoch_ms: int = field(default_factory=lambda: _now_ms())
    required_file_count: int = 0
    uploaded_file_count: int = 0
    total_upload_bytes: int = 0
    uploaded_bytes: int = 0
    current_upload_path: str = ""
    refresh_total_bundles: int = 0
    refresh_completed_bundles: int = 0
    refresh_failed_bundles: int = 0
    current_refresh_bundle: str = ""
    current_refresh_phase: str = ""


class AgentBridgeRuntime:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._lock = Lock()
        self._syncs: dict[str, _SyncOperation] = {}
        self._server_instance_registry = ServerInstanceRegistry(settings.server_instance_bindings_path)

        configure_observability()
        self._settings.plugin_semantic_mc_servers_root.mkdir(parents=True, exist_ok=True)
        self._settings.grpc_upload_tmp_dir.mkdir(parents=True, exist_ok=True)

        self._shared_ranker = (
            BceRanker(settings.reranker_model_name_or_path)
            if settings.reranker_enabled
            else None
        )
        if self._shared_ranker is not None:
            self._shared_ranker.warmup()

        self._plugin_semantic_service = build_plugin_semantic_service(settings)
        self._session_registry = SessionRegistry(
            factory=self._create_session,
            ttl_seconds=settings.grpc_session_ttl_seconds,
        )

    def validate_startup(self):
        store = LancePluginVectorStore(
            self._settings.plugin_docs_vector_db_dir,
            self._settings.plugin_docs_table_name,
            expected_embedding_dimension=self._settings.expected_embedding_dimension,
        )
        stats = store.validate()
        if (
            self._settings.plugin_docs_bm25_enabled
            and self._settings.plugin_docs_bm25_auto_create_index
        ):
            store.ensure_bm25_index()
        return stats

    def probe(self, *, server_id: str, server_instance_id: str) -> None:
        self._validate_server_identity(
            server_id=server_id,
            server_instance_id=server_instance_id,
        )

    def close(self) -> None:
        self._session_registry.close()
        if self._plugin_semantic_service is not None:
            self._plugin_semantic_service.close()

    def ask(self, command: AskCommand) -> AskReply:
        with trace_operation(
            "grpc_bridge.ask",
            attributes={"component": "grpc_bridge", "question.length": len(command.question.strip())},
            metric_name="rag_grpc_bridge_ask_seconds",
        ):
            record_counter("rag_grpc_bridge_ask_requests_total")
            question = command.question.strip()
            if not question:
                raise InvalidRequestError("question must not be blank")

            server_id, _ = self._validate_server_identity(
                server_id=command.server_id,
                server_instance_id=command.server_instance_id,
            )
            player_scope = command.player_id.strip() or command.player_name.strip()
            if not player_scope:
                raise InvalidRequestError("player_id or player_name must be provided")

            scope_id = f"{server_id}:{player_scope}"
            server_plugins = _format_installed_plugins(command.installed_plugins)
            try:
                result = self._session_registry.ask(
                    scope_id,
                    question,
                    server_plugins=server_plugins,
                )
            except RagForMcError as exc:
                raise BridgeRuntimeError(str(exc)) from exc

            return AskReply(
                request_id=command.request_id.strip() or uuid4().hex,
                answer=result.answer,
                citations_summary=_summarize_citations(result.citations),
                backend_trace_id=command.request_id.strip(),
            )

    def prepare_sync(
        self,
        *,
        server_id: str,
        server_instance_id: str,
        manifest: Iterable[ManifestEntry],
    ) -> PrepareSyncReply:
        with trace_operation(
            "grpc_bridge.prepare_sync",
            attributes={"component": "grpc_bridge"},
            metric_name="rag_grpc_bridge_prepare_sync_seconds",
        ):
            record_counter("rag_grpc_bridge_prepare_sync_requests_total")
            normalized_server_id, normalized_server_instance_id = self._validate_server_identity(
                server_id=server_id,
                server_instance_id=server_instance_id,
            )
            manifest_map: dict[str, ManifestEntry] = {}
            rejected: list[RejectedPathEntry] = []

            for entry in manifest:
                try:
                    normalized_path = _normalize_allowed_relative_path(entry.relative_path)
                except InvalidRequestError as exc:
                    rejected.append(
                        RejectedPathEntry(
                            relative_path=entry.relative_path,
                            reason=str(exc),
                        )
                    )
                    continue

                if normalized_path in manifest_map:
                    rejected.append(
                        RejectedPathEntry(
                            relative_path=normalized_path,
                            reason="manifest contains duplicate paths",
                        )
                    )
                    continue

                if entry.size < 0:
                    rejected.append(
                        RejectedPathEntry(
                            relative_path=normalized_path,
                            reason="file size must be >= 0",
                        )
                    )
                    continue

                sha256_value = entry.sha256.strip().lower()
                if len(sha256_value) != 64 or any(ch not in "0123456789abcdef" for ch in sha256_value):
                    rejected.append(
                        RejectedPathEntry(
                            relative_path=normalized_path,
                            reason="sha256 must be a 64-character lowercase hex string",
                        )
                    )
                    continue

                manifest_map[normalized_path] = ManifestEntry(
                    relative_path=normalized_path,
                    size=entry.size,
                    sha256=sha256_value,
                    last_modified_epoch_ms=entry.last_modified_epoch_ms,
                )

            required_paths = sorted(
                path
                for path, entry in manifest_map.items()
                if not self._matches_existing_file(normalized_server_id, entry)
            )

            operation = _SyncOperation(
                sync_id=uuid4().hex,
                server_id=normalized_server_id,
                server_instance_id=normalized_server_instance_id,
                manifest=manifest_map,
                required_paths=set(required_paths),
                rejected_paths=rejected,
                message="Manifest received. Waiting for uploads.",
                required_file_count=len(required_paths),
                total_upload_bytes=sum(manifest_map[path].size for path in required_paths),
            )

            with self._lock:
                self._reap_syncs_locked()
                self._syncs[operation.sync_id] = operation

            return PrepareSyncReply(
                sync_id=operation.sync_id,
                required_paths=required_paths,
                rejected_paths=rejected,
            )

    def upload_file(self, chunks: Iterable[UploadChunk]) -> UploadReply:
        with trace_operation(
            "grpc_bridge.upload_file",
            attributes={"component": "grpc_bridge"},
            metric_name="rag_grpc_bridge_upload_seconds",
        ):
            record_counter("rag_grpc_bridge_upload_requests_total")
            iterator = iter(chunks)
            try:
                first = next(iterator)
            except StopIteration as exc:
                raise InvalidRequestError("upload stream must not be empty") from exc

            sync_id = first.sync_id.strip()
            if not sync_id:
                raise InvalidRequestError("sync_id must not be blank")

            normalized_path = _normalize_allowed_relative_path(first.relative_path)
            expected_sha256 = first.sha256.strip().lower()
            if len(expected_sha256) != 64 or any(ch not in "0123456789abcdef" for ch in expected_sha256):
                raise InvalidRequestError("sha256 must be a 64-character lowercase hex string")
            if first.total_chunks < 1:
                raise InvalidRequestError("total_chunks must be > 0")

            operation = self._get_sync(sync_id)
            with self._lock:
                self._assert_upload_allowed(operation, normalized_path)
                operation.state = SyncState.UPLOADING
                operation.message = f"Receiving {normalized_path}"
                operation.current_upload_path = normalized_path
                operation.updated_at_epoch_ms = _now_ms()

            temp_dir = self._settings.grpc_upload_tmp_dir / sync_id
            temp_dir.mkdir(parents=True, exist_ok=True)
            temp_path = temp_dir / f"{uuid4().hex}.part"
            digest = hashlib.sha256()
            received_bytes = 0
            received_chunks = 0
            expected_chunk_index = 0

            try:
                with temp_path.open("wb") as handle:
                    for chunk in _chain_first(first, iterator):
                        self._validate_chunk(
                            chunk=chunk,
                            sync_id=sync_id,
                            relative_path=normalized_path,
                            total_chunks=first.total_chunks,
                            sha256=expected_sha256,
                            expected_chunk_index=expected_chunk_index,
                        )
                        payload = bytes(chunk.content_bytes)
                        handle.write(payload)
                        digest.update(payload)
                        received_bytes += len(payload)
                        received_chunks += 1
                        expected_chunk_index += 1

                if received_chunks != first.total_chunks:
                    raise FailedPreconditionError(
                        f"incomplete upload stream: expected {first.total_chunks} chunks, got {received_chunks}"
                    )

                actual_sha256 = digest.hexdigest()
                if actual_sha256 != expected_sha256:
                    raise FailedPreconditionError("sha256 verification failed")

                target_path = self._resolve_server_target(operation.server_id, normalized_path)
                target_path.parent.mkdir(parents=True, exist_ok=True)
                os.replace(temp_path, target_path)

                with self._lock:
                    operation.received_paths.add(normalized_path)
                    operation.uploaded_file_count = len(operation.received_paths)
                    operation.uploaded_bytes += received_bytes
                    operation.current_upload_path = ""
                    operation.message = f"Uploaded {normalized_path}"
                    operation.updated_at_epoch_ms = _now_ms()

                return UploadReply(
                    sync_id=sync_id,
                    relative_path=normalized_path,
                    received_bytes=received_bytes,
                    received_chunks=received_chunks,
                    sha256_verified=True,
                    message="File upload completed.",
                )
            except BridgeRuntimeError as exc:
                self._mark_sync_failed(sync_id, str(exc))
                raise
            except Exception as exc:
                self._mark_sync_failed(sync_id, f"file upload failed: {exc}")
                raise BridgeRuntimeError(f"file upload failed: {exc}") from exc
            finally:
                if temp_path.exists():
                    temp_path.unlink(missing_ok=True)

    def commit_sync(
        self,
        *,
        sync_id: str,
        server_id: str,
        server_instance_id: str,
        uploaded_paths: Iterable[str],
    ) -> CommitSyncReply:
        with trace_operation(
            "grpc_bridge.commit_sync",
            attributes={"component": "grpc_bridge"},
            metric_name="rag_grpc_bridge_commit_sync_seconds",
        ):
            record_counter("rag_grpc_bridge_commit_sync_requests_total")
            operation = self._get_sync(sync_id)
            normalized_server_id, normalized_server_instance_id = self._validate_server_identity(
                server_id=server_id,
                server_instance_id=server_instance_id,
            )
            if operation.server_id != normalized_server_id:
                raise FailedPreconditionError("sync_id and server_id do not match")
            if operation.server_instance_id != normalized_server_instance_id:
                raise FailedPreconditionError("sync_id and server_instance_id do not match")

            normalized_uploaded_paths = {
                _normalize_allowed_relative_path(path) for path in uploaded_paths
            }
            missing_paths = operation.required_paths - operation.received_paths
            unexpected_paths = normalized_uploaded_paths - operation.received_paths
            if missing_paths:
                raise FailedPreconditionError(
                    "missing uploaded files: " + ", ".join(sorted(missing_paths))
                )
            if unexpected_paths:
                raise InvalidRequestError(
                    "commit included paths that were not uploaded successfully: "
                    + ", ".join(sorted(unexpected_paths))
                )
            if normalized_uploaded_paths != operation.received_paths:
                raise FailedPreconditionError(
                    "commit uploaded_paths does not match the successfully uploaded files"
                )

            with self._lock:
                operation.accepted_count = len(operation.received_paths)
                operation.indexed_count = 0
                operation.current_upload_path = ""
                operation.updated_at_epoch_ms = _now_ms()

            if not operation.required_paths:
                return self._complete_without_refresh(
                    operation,
                    message="No changed files required semantic refresh.",
                )

            if self._plugin_semantic_service is None:
                return self._complete_without_refresh(
                    operation,
                    message="Files were committed, but plugin semantic refresh is disabled.",
                )

            refresh_status = self._plugin_semantic_service.request_refresh_status(full=False)
            with self._lock:
                operation.refresh_started = refresh_status in {"started", "already_running"}
                operation.state = (
                    SyncState.INDEXING if operation.refresh_started else SyncState.FAILED
                )
                operation.message = _render_refresh_message(refresh_status)
                operation.current_refresh_phase = "queued" if operation.refresh_started else "not_started"
                operation.updated_at_epoch_ms = _now_ms()
                self._apply_refresh_progress_locked(
                    operation,
                    self._plugin_semantic_service.get_refresh_progress_snapshot(),
                )

            if not operation.refresh_started:
                raise FailedPreconditionError(operation.message)

            return CommitSyncReply(
                sync_id=operation.sync_id,
                accepted_count=operation.accepted_count,
                indexed_count=operation.indexed_count,
                refresh_started=operation.refresh_started,
                message=operation.message,
            )

    def get_sync_status(self, sync_id: str) -> SyncStatusSnapshot:
        with trace_operation(
            "grpc_bridge.get_sync_status",
            attributes={"component": "grpc_bridge"},
            metric_name="rag_grpc_bridge_status_seconds",
        ):
            record_counter("rag_grpc_bridge_status_requests_total")
            operation = self._get_sync(sync_id)
            with self._lock:
                self._refresh_indexing_status_locked(operation)
                return SyncStatusSnapshot(
                    sync_id=operation.sync_id,
                    state=operation.state,
                    accepted_count=operation.accepted_count,
                    indexed_count=operation.indexed_count,
                    refresh_started=operation.refresh_started,
                    message=operation.message,
                    updated_at_epoch_ms=operation.updated_at_epoch_ms,
                    required_file_count=operation.required_file_count,
                    uploaded_file_count=operation.uploaded_file_count,
                    total_upload_bytes=operation.total_upload_bytes,
                    uploaded_bytes=operation.uploaded_bytes,
                    current_upload_path=operation.current_upload_path,
                    refresh_total_bundles=operation.refresh_total_bundles,
                    refresh_completed_bundles=operation.refresh_completed_bundles,
                    refresh_failed_bundles=operation.refresh_failed_bundles,
                    current_refresh_bundle=operation.current_refresh_bundle,
                    current_refresh_phase=operation.current_refresh_phase,
                )

    def _validate_server_identity(
        self,
        *,
        server_id: str,
        server_instance_id: str,
    ) -> tuple[str, str]:
        return self._server_instance_registry.validate_or_bind(
            server_id=server_id,
            server_instance_id=server_instance_id,
        )

    def _create_session(self, memory_scope_id: str) -> RagChatSession:
        return build_session(
            self._settings,
            memory_scope_id=memory_scope_id,
            ranker=self._shared_ranker,
            plugin_semantic_service=self._plugin_semantic_service,
            attach_plugin_semantic_service_to_session=False,
            configure_runtime_observability=False,
        )

    def _matches_existing_file(self, server_id: str, entry: ManifestEntry) -> bool:
        target_path = self._resolve_server_target(server_id, entry.relative_path)
        if not target_path.is_file():
            return False
        try:
            if target_path.stat().st_size != entry.size:
                return False
            return _sha256_path(target_path) == entry.sha256
        except OSError:
            return False

    def _resolve_server_target(self, server_id: str, relative_path: str) -> Path:
        server_root = (self._settings.plugin_semantic_mc_servers_root / server_id).resolve()
        server_root.mkdir(parents=True, exist_ok=True)
        target = (server_root / relative_path.replace("/", os.sep)).resolve()
        try:
            target.relative_to(server_root)
        except ValueError as exc:
            raise InvalidRequestError("target path escapes the server root") from exc
        return target

    def _get_sync(self, sync_id: str) -> _SyncOperation:
        with self._lock:
            self._reap_syncs_locked()
            operation = self._syncs.get(sync_id)
            if operation is None:
                raise NotFoundError(f"no sync operation found for sync_id={sync_id}")
            return operation

    def _assert_upload_allowed(self, operation: _SyncOperation, relative_path: str) -> None:
        if operation.state in {SyncState.COMPLETED, SyncState.FAILED}:
            raise FailedPreconditionError("sync operation has already finished")
        if relative_path not in operation.required_paths:
            raise FailedPreconditionError("path is not required by this sync operation")
        if relative_path in operation.received_paths:
            raise ConflictError("path has already been uploaded")

    def _validate_chunk(
        self,
        *,
        chunk: UploadChunk,
        sync_id: str,
        relative_path: str,
        total_chunks: int,
        sha256: str,
        expected_chunk_index: int,
    ) -> None:
        if chunk.sync_id.strip() != sync_id:
            raise FailedPreconditionError("sync_id changed within the upload stream")
        if _normalize_allowed_relative_path(chunk.relative_path) != relative_path:
            raise FailedPreconditionError("relative_path changed within the upload stream")
        if chunk.total_chunks != total_chunks:
            raise FailedPreconditionError("total_chunks changed within the upload stream")
        if chunk.sha256.strip().lower() != sha256:
            raise FailedPreconditionError("sha256 changed within the upload stream")
        if chunk.chunk_index != expected_chunk_index:
            raise FailedPreconditionError(
                f"unexpected chunk order: expected {expected_chunk_index}, got {chunk.chunk_index}"
            )

    def _complete_without_refresh(
        self,
        operation: _SyncOperation,
        *,
        message: str,
    ) -> CommitSyncReply:
        with self._lock:
            operation.state = SyncState.COMPLETED
            operation.indexed_count = _count_indexable_paths(operation.received_paths)
            operation.refresh_started = False
            operation.message = message
            operation.current_refresh_phase = "not_started"
            operation.updated_at_epoch_ms = _now_ms()
            reply = CommitSyncReply(
                sync_id=operation.sync_id,
                accepted_count=operation.accepted_count,
                indexed_count=operation.indexed_count,
                refresh_started=operation.refresh_started,
                message=operation.message,
            )
        self._cleanup_sync_temp_dir(operation.sync_id)
        return reply

    def _refresh_indexing_status_locked(self, operation: _SyncOperation) -> None:
        if operation.state != SyncState.INDEXING:
            return
        if self._plugin_semantic_service is None:
            operation.state = SyncState.COMPLETED
            operation.indexed_count = _count_indexable_paths(operation.received_paths)
            operation.refresh_started = False
            operation.message = "plugin semantic refresh is unavailable"
            operation.current_refresh_phase = "unavailable"
            operation.updated_at_epoch_ms = _now_ms()
            return

        progress = self._plugin_semantic_service.get_refresh_progress_snapshot()
        self._apply_refresh_progress_locked(operation, progress)
        operation.message = progress.message or operation.message
        operation.updated_at_epoch_ms = _now_ms()
        if progress.running:
            return

        if progress.failed_bundles > 0:
            operation.state = SyncState.FAILED
            operation.indexed_count = 0
        else:
            operation.state = SyncState.COMPLETED
            operation.indexed_count = _count_indexable_paths(operation.received_paths)
        self._cleanup_sync_temp_dir(operation.sync_id)

    def _mark_sync_failed(self, sync_id: str, message: str) -> None:
        with self._lock:
            operation = self._syncs.get(sync_id)
            if operation is None:
                return
            operation.state = SyncState.FAILED
            operation.message = message
            operation.current_upload_path = ""
            operation.updated_at_epoch_ms = _now_ms()
        self._cleanup_sync_temp_dir(sync_id)

    def _apply_refresh_progress_locked(
        self,
        operation: _SyncOperation,
        progress: RefreshProgressSnapshot,
    ) -> None:
        operation.refresh_total_bundles = progress.total_bundles
        operation.refresh_completed_bundles = progress.completed_bundles
        operation.refresh_failed_bundles = progress.failed_bundles
        operation.current_refresh_bundle = progress.current_bundle
        operation.current_refresh_phase = progress.current_phase

    def _reap_syncs_locked(self) -> None:
        ttl_ms = self._settings.grpc_sync_ttl_seconds * 1000
        if ttl_ms <= 0:
            return
        now_ms = _now_ms()
        expired_ids = [
            sync_id
            for sync_id, operation in self._syncs.items()
            if now_ms - operation.updated_at_epoch_ms > ttl_ms
        ]
        for sync_id in expired_ids:
            self._syncs.pop(sync_id, None)
            self._cleanup_sync_temp_dir(sync_id)

    def _cleanup_sync_temp_dir(self, sync_id: str) -> None:
        temp_dir = self._settings.grpc_upload_tmp_dir / sync_id
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


def _chain_first(first, iterator):
    yield first
    yield from iterator


def _now_ms() -> int:
    return int(time() * 1000)


def _validate_identifier(value: str, *, field_name: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise InvalidRequestError(f"{field_name} must not be blank")
    if normalized in {".", ".."}:
        raise InvalidRequestError(f"{field_name} must not be '.' or '..'")
    if any(ch in normalized for ch in ("/", "\\", ":")):
        raise InvalidRequestError(f"{field_name} must not contain path separators or drive markers")
    return normalized


def _normalize_allowed_relative_path(relative_path: str) -> str:
    try:
        return normalize_allowed_config_relative_path(relative_path)
    except ValueError as exc:
        raise InvalidRequestError(str(exc)) from exc


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _summarize_citations(citations: list[RetrievedDoc]) -> str:
    summary_parts: list[str] = []
    seen: set[tuple[str, str]] = set()
    for citation in citations:
        label = citation.plugin_chinese_name.strip() or citation.plugin_english_name.strip()
        if not label:
            continue
        key = (label, citation.match_reason)
        if key in seen:
            continue
        seen.add(key)
        summary_parts.append(f"{label}({citation.match_reason})")
        if len(summary_parts) >= 4:
            break
    return ", ".join(summary_parts)


def _format_installed_plugins(plugins: Iterable[ServerPlugin]) -> list[str]:
    formatted: list[str] = []
    seen: set[str] = set()
    for plugin in plugins:
        name = plugin.name.strip()
        if not name:
            continue
        version = plugin.version.strip()
        status = "enabled" if plugin.enabled else "disabled"
        label = f"{name} {version} ({status})" if version else f"{name} ({status})"
        dedupe_key = label.casefold()
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        formatted.append(label)
    return formatted


def _render_refresh_message(refresh_status: str) -> str:
    if refresh_status == "started":
        return "Files committed. Semantic refresh started."
    if refresh_status == "already_running":
        return "Files committed. Semantic refresh is already running."
    if refresh_status == "closed":
        return "Semantic refresh service is closed."
    return f"Semantic refresh returned an unknown state: {refresh_status}"


def _count_indexable_paths(paths: Iterable[str]) -> int:
    count = 0
    for relative_path in paths:
        if _is_semantic_indexable_path(relative_path):
            count += 1
    return count


def _is_semantic_indexable_path(relative_path: str) -> bool:
    return is_semantic_indexable_relative_path(relative_path)
