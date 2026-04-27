from __future__ import annotations

from collections.abc import Iterable, Iterator
from importlib.metadata import PackageNotFoundError, version as package_version

import grpc

from . import agent_bridge_pb2, agent_bridge_pb2_grpc
from agent_for_mc.interfaces.grpc.runtime import (
    AgentBridgeRuntime,
    AskCommand,
    BridgeRuntimeError,
    CommitSyncReply,
    ConflictError,
    FailedPreconditionError,
    InvalidRequestError,
    ManifestEntry,
    NotFoundError,
    PrepareSyncReply,
    ServerPlugin,
    AskStreamEvent,
    SyncState,
    SyncStatusSnapshot,
    UploadChunk,
    UploadReply,
)


BRIDGE_PROTOCOL_VERSION = 1
BACKEND_NAME = "AgentForMc"
CAPABILITY_ASK_STREAM_PROGRESS = "ask_stream_progress"


class AgentBridgeService(agent_bridge_pb2_grpc.AgentBridgeServiceServicer):
    def __init__(self, *, runtime: AgentBridgeRuntime, auth_token: str):
        self._runtime = runtime
        self._auth_token = auth_token.strip()
        self._backend_version = _resolve_backend_version()

    def Probe(self, request, context):
        try:
            self._runtime.probe(
                server_id=request.server_id,
                server_instance_id=request.server_instance_id,
            )
        except Exception as exc:  # pragma: no cover - exercised via gRPC tests
            self._abort_from_exception(context, exc)

        return agent_bridge_pb2.ProbeResponse(
            ack=True,
            message="AgentForMc gRPC bridge is ready",
            backend_name=BACKEND_NAME,
            backend_version=self._backend_version,
            protocol_version=BRIDGE_PROTOCOL_VERSION,
            capabilities=[CAPABILITY_ASK_STREAM_PROGRESS],
        )

    def Ask(self, request, context):
        self._require_authorization(context)
        try:
            reply = self._runtime.ask(_build_ask_command(request))
        except Exception as exc:  # pragma: no cover - exercised via gRPC tests
            self._abort_from_exception(context, exc)

        return _build_ask_response(reply)

    def AskStream(self, request, context):
        self._require_authorization(context)
        try:
            for event in self._runtime.ask_stream(_build_ask_command(request)):
                yield _build_ask_event(event)
        except Exception as exc:  # pragma: no cover - exercised via gRPC tests
            self._abort_from_exception(context, exc)

    def PrepareSync(self, request, context):
        self._require_authorization(context)
        manifest = [
            ManifestEntry(
                relative_path=item.relative_path,
                size=item.size,
                sha256=item.sha256,
                last_modified_epoch_ms=item.last_modified_epoch_ms,
            )
            for item in request.manifest
        ]
        try:
            reply = self._runtime.prepare_sync(
                server_id=request.server_id,
                server_instance_id=request.server_instance_id,
                manifest=manifest,
            )
        except Exception as exc:  # pragma: no cover - exercised via gRPC tests
            self._abort_from_exception(context, exc)

        return _build_prepare_sync_response(reply)

    def UploadFileChunk(self, request_iterator, context):
        self._require_authorization(context)
        try:
            reply = self._runtime.upload_file(_iter_upload_chunks(request_iterator))
        except Exception as exc:  # pragma: no cover - exercised via gRPC tests
            self._abort_from_exception(context, exc)

        return _build_upload_response(reply)

    def CommitSync(self, request, context):
        self._require_authorization(context)
        try:
            reply = self._runtime.commit_sync(
                sync_id=request.sync_id,
                server_id=request.server_id,
                server_instance_id=request.server_instance_id,
                uploaded_paths=request.uploaded_paths,
            )
        except Exception as exc:  # pragma: no cover - exercised via gRPC tests
            self._abort_from_exception(context, exc)

        return _build_commit_sync_response(reply)

    def GetSyncStatus(self, request, context):
        self._require_authorization(context)
        try:
            reply = self._runtime.get_sync_status(request.sync_id)
        except Exception as exc:  # pragma: no cover - exercised via gRPC tests
            self._abort_from_exception(context, exc)

        return _build_sync_status_response(reply)

    def _require_authorization(self, context) -> None:
        metadata = dict(context.invocation_metadata())
        authorization = metadata.get("authorization", "")
        if not authorization:
            context.abort(grpc.StatusCode.UNAUTHENTICATED, "missing authorization metadata")
        if not authorization.startswith("Bearer "):
            context.abort(grpc.StatusCode.UNAUTHENTICATED, "authorization must use a Bearer token")
        token = authorization[len("Bearer ") :].strip()
        if token != self._auth_token:
            context.abort(grpc.StatusCode.UNAUTHENTICATED, "invalid authentication token")

    def _abort_from_exception(self, context, exc: Exception):
        if isinstance(exc, InvalidRequestError):
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        if isinstance(exc, NotFoundError):
            context.abort(grpc.StatusCode.NOT_FOUND, str(exc))
        if isinstance(exc, ConflictError):
            context.abort(grpc.StatusCode.ALREADY_EXISTS, str(exc))
        if isinstance(exc, FailedPreconditionError):
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))
        if isinstance(exc, BridgeRuntimeError):
            context.abort(grpc.StatusCode.INTERNAL, str(exc))
        context.abort(grpc.StatusCode.INTERNAL, f"unhandled backend error: {exc}")


def _resolve_backend_version() -> str:
    for distribution_name in ("agent-for-mc", "agent_for_mc"):
        try:
            return package_version(distribution_name)
        except PackageNotFoundError:
            continue
    return "dev"


def _build_ask_command(request) -> AskCommand:
    return AskCommand(
        server_id=request.server_id,
        server_instance_id=request.server_instance_id,
        player_id=request.player_id,
        player_name=request.player_name,
        question=request.question,
        request_id=request.request_id,
        timestamp_ms=request.timestamp,
        installed_plugins=[
            ServerPlugin(
                name=plugin.name,
                version=plugin.version,
                enabled=plugin.enabled,
            )
            for plugin in request.installed_plugins
        ],
    )


def _build_ask_response(reply) -> agent_bridge_pb2.AskResponse:
    return agent_bridge_pb2.AskResponse(
        request_id=reply.request_id,
        answer=reply.answer,
        citations_summary=reply.citations_summary,
        backend_trace_id=reply.backend_trace_id,
    )


def _build_ask_event(event: AskStreamEvent) -> agent_bridge_pb2.AskEvent:
    if event.reply is not None:
        return agent_bridge_pb2.AskEvent(response=_build_ask_response(event.reply))
    if event.progress is not None:
        progress = event.progress
        return agent_bridge_pb2.AskEvent(
            progress=agent_bridge_pb2.AskProgress(
                request_id=progress.request_id,
                stage=progress.stage,
                message=progress.message,
                elapsed_ms=progress.elapsed_ms,
                sequence=progress.sequence,
            )
        )
    raise BridgeRuntimeError("empty ask stream event")


def _iter_upload_chunks(
    request_iterator: Iterable[agent_bridge_pb2.FileChunkUploadRequest],
) -> Iterator[UploadChunk]:
    for request in request_iterator:
        yield UploadChunk(
            sync_id=request.sync_id,
            relative_path=request.relative_path,
            chunk_index=request.chunk_index,
            total_chunks=request.total_chunks,
            content_bytes=request.content_bytes,
            sha256=request.sha256,
        )


def _build_prepare_sync_response(reply: PrepareSyncReply) -> agent_bridge_pb2.SyncPrepareResponse:
    return agent_bridge_pb2.SyncPrepareResponse(
        sync_id=reply.sync_id,
        required_paths=reply.required_paths,
        rejected_paths=[
            agent_bridge_pb2.RejectedPath(
                relative_path=item.relative_path,
                reason=item.reason,
            )
            for item in reply.rejected_paths
        ],
    )


def _build_upload_response(reply: UploadReply) -> agent_bridge_pb2.FileChunkUploadResponse:
    return agent_bridge_pb2.FileChunkUploadResponse(
        sync_id=reply.sync_id,
        relative_path=reply.relative_path,
        received_bytes=reply.received_bytes,
        received_chunks=reply.received_chunks,
        sha256_verified=reply.sha256_verified,
        message=reply.message,
    )


def _build_commit_sync_response(reply: CommitSyncReply) -> agent_bridge_pb2.SyncCommitResponse:
    return agent_bridge_pb2.SyncCommitResponse(
        sync_id=reply.sync_id,
        accepted_count=reply.accepted_count,
        indexed_count=reply.indexed_count,
        refresh_started=reply.refresh_started,
        message=reply.message,
    )


def _build_sync_status_response(reply: SyncStatusSnapshot) -> agent_bridge_pb2.SyncStatusResponse:
    return agent_bridge_pb2.SyncStatusResponse(
        sync_id=reply.sync_id,
        state=_sync_state_to_proto(reply.state),
        accepted_count=reply.accepted_count,
        indexed_count=reply.indexed_count,
        refresh_started=reply.refresh_started,
        message=reply.message,
        updated_at_epoch_ms=reply.updated_at_epoch_ms,
        required_file_count=reply.required_file_count,
        uploaded_file_count=reply.uploaded_file_count,
        total_upload_bytes=reply.total_upload_bytes,
        uploaded_bytes=reply.uploaded_bytes,
        current_upload_path=reply.current_upload_path,
        refresh_total_bundles=reply.refresh_total_bundles,
        refresh_completed_bundles=reply.refresh_completed_bundles,
        refresh_failed_bundles=reply.refresh_failed_bundles,
        current_refresh_bundle=reply.current_refresh_bundle,
        current_refresh_phase=reply.current_refresh_phase,
    )


def _sync_state_to_proto(state: SyncState) -> agent_bridge_pb2.SyncState:
    if state == SyncState.PENDING:
        return agent_bridge_pb2.SYNC_STATE_PENDING
    if state == SyncState.UPLOADING:
        return agent_bridge_pb2.SYNC_STATE_UPLOADING
    if state == SyncState.INDEXING:
        return agent_bridge_pb2.SYNC_STATE_INDEXING
    if state == SyncState.COMPLETED:
        return agent_bridge_pb2.SYNC_STATE_COMPLETED
    if state == SyncState.FAILED:
        return agent_bridge_pb2.SYNC_STATE_FAILED
    return agent_bridge_pb2.SYNC_STATE_UNSPECIFIED
