from __future__ import annotations

import hashlib
from concurrent import futures
from contextlib import contextmanager
from pathlib import Path

import grpc
import pytest

from agent_for_mc.infrastructure.config import Settings
from agent_for_mc.interfaces.grpc import agent_bridge_pb2, agent_bridge_pb2_grpc
from agent_for_mc.interfaces.grpc.runtime import (
    AgentBridgeRuntime,
    AskCommand,
    AskReply,
    CommitSyncReply,
    FailedPreconditionError,
    ManifestEntry,
    PrepareSyncReply,
    RefreshProgressSnapshot,
    RejectedPathEntry,
    ServerPlugin,
    ServerInstanceRegistry,
    SyncState,
    SyncStatusSnapshot,
    UploadChunk,
    UploadReply,
)
from agent_for_mc.interfaces.grpc.service import AgentBridgeService


class FakeRuntime:
    def __init__(self):
        self.seen_ask_command: AskCommand | None = None
        self.seen_probe_identity: tuple[str, str] | None = None
        self.seen_prepare_identity: tuple[str, str] | None = None
        self.seen_commit_identity: tuple[str, str] | None = None

    def probe(self, *, server_id: str, server_instance_id: str) -> None:
        self.seen_probe_identity = (server_id, server_instance_id)

    def ask(self, command: AskCommand) -> AskReply:
        self.seen_ask_command = command
        return AskReply(
            request_id=command.request_id,
            answer="test answer",
            citations_summary="Essentials(vector)",
            backend_trace_id="trace-123",
        )

    def prepare_sync(self, *, server_id: str, server_instance_id: str, manifest):
        self.seen_prepare_identity = (server_id, server_instance_id)
        return PrepareSyncReply(
            sync_id="sync-1",
            required_paths=["plugins/TestPlugin/config.yml"],
            rejected_paths=[RejectedPathEntry(relative_path="bad.bin", reason="unsupported")],
        )

    def upload_file(self, chunks):
        chunk_list = list(chunks)
        return UploadReply(
            sync_id=chunk_list[0].sync_id,
            relative_path=chunk_list[0].relative_path,
            received_bytes=sum(len(item.content_bytes) for item in chunk_list),
            received_chunks=len(chunk_list),
            sha256_verified=True,
            message="uploaded",
        )

    def commit_sync(
        self,
        *,
        sync_id: str,
        server_id: str,
        server_instance_id: str,
        uploaded_paths,
    ):
        self.seen_commit_identity = (server_id, server_instance_id)
        return CommitSyncReply(
            sync_id=sync_id,
            accepted_count=len(list(uploaded_paths)),
            indexed_count=1,
            refresh_started=True,
            message="refresh started",
        )

    def get_sync_status(self, sync_id: str):
        return SyncStatusSnapshot(
            sync_id=sync_id,
            state=SyncState.COMPLETED,
            accepted_count=1,
            indexed_count=1,
            refresh_started=True,
            message="done",
            updated_at_epoch_ms=123,
            required_file_count=2,
            uploaded_file_count=2,
            total_upload_bytes=300,
            uploaded_bytes=300,
            current_upload_path="",
            refresh_total_bundles=2,
            refresh_completed_bundles=2,
            refresh_failed_bundles=0,
            current_refresh_bundle="",
            current_refresh_phase="completed",
        )


class FakeRefreshService:
    def __init__(self):
        self.running = False

    def request_refresh_status(self, *, full: bool = False) -> str:
        self.running = True
        return "started"

    def is_refresh_running(self) -> bool:
        return self.running

    def get_refresh_progress_snapshot(self) -> RefreshProgressSnapshot:
        return RefreshProgressSnapshot(
            running=self.running,
            total_bundles=1,
            completed_bundles=0 if self.running else 1,
            failed_bundles=0,
            current_bundle="lobby-1/__server_core__" if self.running else "",
            current_phase="refreshing_bundle" if self.running else "completed",
            message="refreshing" if self.running else "completed",
        )

    def close(self) -> None:
        self.running = False


@contextmanager
def running_grpc_server(runtime, *, token: str = "secret-token"):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    agent_bridge_pb2_grpc.add_AgentBridgeServiceServicer_to_server(
        AgentBridgeService(runtime=runtime, auth_token=token),
        server,
    )
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    channel = grpc.insecure_channel(f"127.0.0.1:{port}")
    try:
        yield agent_bridge_pb2_grpc.AgentBridgeServiceStub(channel)
    finally:
        channel.close()
        server.stop(None)


def make_settings(tmp_path: Path) -> Settings:
    return Settings(
        plugin_docs_vector_db_dir=tmp_path / "plugin_docs_vector_db",
        plugin_docs_table_name="plugin_docs",
        deepseek_api_key="deepseek-key",
        deepseek_model="deepseek-chat",
        deepseek_chat_url="https://example.invalid/chat/completions",
        expected_embedding_dimension=1024,
        rewrite_history_turns=4,
        retrieval_top_k=8,
        answer_top_k=4,
        citation_preview_chars=200,
        request_timeout_seconds=30,
        model_cache_dir=tmp_path / ".cache" / "models",
        reranker_enabled=False,
        reranker_model_name_or_path="unused",
        plugin_config_agent_model="deepseek-chat",
        memory_maintenance_agent_model="deepseek-chat",
        memory_enabled=False,
        user_semantic_memory_db_path=tmp_path / "data" / "user_semantic_memory.sqlite3",
        memory_recall_limit=5,
        memory_min_confidence=0.75,
        memory_consolidation_turns=4,
        plugin_semantic_mc_servers_root=tmp_path / "mc_servers",
        plugin_semantic_agent_model="deepseek-chat",
        plugin_semantic_agent_scan_on_startup=False,
        plugin_semantic_agent_refresh_interval_seconds=1800,
        plugin_semantic_agent_max_file_chars=12000,
        plugin_semantic_agent_max_files_per_plugin=20,
        server_config_semantic_vector_db_dir=tmp_path / "server_config_semantic_vector_db",
        server_config_semantic_table_name="server_config_semantic_memories",
        server_config_semantic_top_k=8,
        server_config_semantic_preview_chars=220,
        grpc_host="127.0.0.1",
        grpc_port=50051,
        grpc_auth_token="secret-token",
        grpc_max_workers=4,
        grpc_session_ttl_seconds=60,
        grpc_sync_ttl_seconds=60,
        grpc_upload_tmp_dir=tmp_path / ".cache" / "grpc_uploads",
        server_instance_bindings_path=tmp_path / "data" / "server_instance_bindings.json",
        embedding_api_key="zhipu-key",
        embedding_api_key_env="RAG_ZHIPU_API_KEY",
        embedding_url="https://open.bigmodel.cn/api/paas/v4/embeddings",
        embedding_model="embedding-3",
    )


def test_server_instance_registry_rejects_reused_server_id(tmp_path: Path):
    registry = ServerInstanceRegistry(tmp_path / "server_instance_bindings.json")

    assert registry.validate_or_bind(
        server_id="lobby-1",
        server_instance_id="instance-1",
    ) == ("lobby-1", "instance-1")
    assert registry.validate_or_bind(
        server_id="lobby-1",
        server_instance_id="instance-1",
    ) == ("lobby-1", "instance-1")

    with pytest.raises(FailedPreconditionError, match="server.id conflict"):
        registry.validate_or_bind(
            server_id="lobby-1",
            server_instance_id="instance-2",
        )


def test_grpc_service_requires_authorization():
    runtime = FakeRuntime()
    with running_grpc_server(runtime) as stub:
        with pytest.raises(grpc.RpcError) as exc_info:
            stub.GetSyncStatus(agent_bridge_pb2.SyncStatusRequest(sync_id="sync-1"))

    assert exc_info.value.code() == grpc.StatusCode.UNAUTHENTICATED


def test_grpc_probe_returns_ack_without_authorization():
    runtime = FakeRuntime()
    with running_grpc_server(runtime) as stub:
        response = stub.Probe(
            agent_bridge_pb2.ProbeRequest(
                server_id="lobby-1",
                server_instance_id="instance-1",
                plugin_name="Agent4Minecraft",
                plugin_version="1.0.0",
                protocol_version=1,
            )
        )

    assert response.ack is True
    assert response.backend_name == "AgentForMc"
    assert response.protocol_version == 1
    assert runtime.seen_probe_identity == ("lobby-1", "instance-1")


def test_grpc_probe_rejects_server_id_conflict_without_authorization():
    class ConflictingRuntime(FakeRuntime):
        def probe(self, *, server_id: str, server_instance_id: str) -> None:
            raise FailedPreconditionError("server.id conflict: lobby-1")

    with running_grpc_server(ConflictingRuntime()) as stub:
        with pytest.raises(grpc.RpcError) as exc_info:
            stub.Probe(
                agent_bridge_pb2.ProbeRequest(
                    server_id="lobby-1",
                    server_instance_id="instance-2",
                    plugin_name="Agent4Minecraft",
                    plugin_version="1.0.0",
                    protocol_version=1,
                )
            )

    assert exc_info.value.code() == grpc.StatusCode.FAILED_PRECONDITION
    assert "server.id conflict" in exc_info.value.details()


def test_grpc_service_maps_ask_request_and_response():
    runtime = FakeRuntime()
    metadata = (("authorization", "Bearer secret-token"),)
    with running_grpc_server(runtime) as stub:
        response = stub.Ask(
            agent_bridge_pb2.AskRequest(
                server_id="lobby-1",
                server_instance_id="instance-1",
                player_id="player-123",
                player_name="Steve",
                question="How do I configure homes?",
                request_id="req-1",
                timestamp=123456,
                installed_plugins=[
                    agent_bridge_pb2.ServerPlugin(
                        name="EssentialsX",
                        version="2.20.1",
                        enabled=True,
                    )
                ],
            ),
            metadata=metadata,
        )

    assert response.request_id == "req-1"
    assert response.answer == "test answer"
    assert response.citations_summary == "Essentials(vector)"
    assert response.backend_trace_id == "trace-123"
    assert runtime.seen_ask_command is not None
    assert runtime.seen_ask_command.server_id == "lobby-1"
    assert runtime.seen_ask_command.server_instance_id == "instance-1"
    assert runtime.seen_ask_command.player_id == "player-123"
    assert runtime.seen_ask_command.installed_plugins == [
        ServerPlugin(name="EssentialsX", version="2.20.1", enabled=True)
    ]


def test_grpc_service_maps_sync_identity():
    runtime = FakeRuntime()
    metadata = (("authorization", "Bearer secret-token"),)
    with running_grpc_server(runtime) as stub:
        prepare_response = stub.PrepareSync(
            agent_bridge_pb2.SyncPrepareRequest(
                server_id="lobby-1",
                server_instance_id="instance-1",
                manifest=[
                    agent_bridge_pb2.FileManifestEntry(
                        relative_path="plugins/TestPlugin/config.yml",
                        size=1,
                        sha256="a" * 64,
                        last_modified_epoch_ms=1,
                    )
                ],
            ),
            metadata=metadata,
        )
        commit_response = stub.CommitSync(
            agent_bridge_pb2.SyncCommitRequest(
                sync_id=prepare_response.sync_id,
                server_id="lobby-1",
                server_instance_id="instance-1",
                uploaded_paths=["plugins/TestPlugin/config.yml"],
            ),
            metadata=metadata,
        )

    assert runtime.seen_prepare_identity == ("lobby-1", "instance-1")
    assert runtime.seen_commit_identity == ("lobby-1", "instance-1")
    assert commit_response.accepted_count == 1


def test_grpc_service_maps_sync_status_progress():
    runtime = FakeRuntime()
    metadata = (("authorization", "Bearer secret-token"),)
    with running_grpc_server(runtime) as stub:
        response = stub.GetSyncStatus(
            agent_bridge_pb2.SyncStatusRequest(sync_id="sync-1"),
            metadata=metadata,
        )

    assert response.sync_id == "sync-1"
    assert response.required_file_count == 2
    assert response.uploaded_file_count == 2
    assert response.total_upload_bytes == 300
    assert response.uploaded_bytes == 300
    assert response.refresh_total_bundles == 2
    assert response.refresh_completed_bundles == 2
    assert response.current_refresh_phase == "completed"


def test_runtime_prepare_sync_rejects_invalid_and_duplicate_paths(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(
        "agent_for_mc.interfaces.grpc.runtime.build_plugin_semantic_service",
        lambda settings: None,
    )
    runtime = AgentBridgeRuntime(make_settings(tmp_path))
    try:
        reply = runtime.prepare_sync(
            server_id="lobby-1",
            server_instance_id="instance-1",
            manifest=[
                ManifestEntry(
                    relative_path="plugins/TestPlugin/config.yml",
                    size=10,
                    sha256="a" * 64,
                    last_modified_epoch_ms=1,
                ),
                ManifestEntry(
                    relative_path="plugins/TestPlugin/config.yml",
                    size=10,
                    sha256="a" * 64,
                    last_modified_epoch_ms=1,
                ),
                ManifestEntry(
                    relative_path="../evil.yml",
                    size=1,
                    sha256="b" * 64,
                    last_modified_epoch_ms=1,
                ),
            ],
        )
    finally:
        runtime.close()

    assert reply.required_paths == ["plugins/TestPlugin/config.yml"]
    assert len(reply.rejected_paths) == 2
    assert {item.relative_path for item in reply.rejected_paths} == {
        "plugins/TestPlugin/config.yml",
        "../evil.yml",
    }


def test_runtime_prepare_upload_commit_and_complete_status(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "agent_for_mc.interfaces.grpc.runtime.build_plugin_semantic_service",
        lambda settings: FakeRefreshService(),
    )
    runtime = AgentBridgeRuntime(make_settings(tmp_path))
    plugin_bytes = b"key: value\n"
    core_bytes = b"motd=hello\n"
    plugin_sha = hashlib.sha256(plugin_bytes).hexdigest()
    core_sha = hashlib.sha256(core_bytes).hexdigest()

    try:
        prepare_reply = runtime.prepare_sync(
            server_id="lobby-1",
            server_instance_id="instance-1",
            manifest=[
                ManifestEntry(
                    relative_path="plugins/TestPlugin/config.yml",
                    size=len(plugin_bytes),
                    sha256=plugin_sha,
                    last_modified_epoch_ms=1,
                ),
                ManifestEntry(
                    relative_path="server.properties",
                    size=len(core_bytes),
                    sha256=core_sha,
                    last_modified_epoch_ms=1,
                ),
            ],
        )

        plugin_upload = runtime.upload_file(
            [
                UploadChunk(
                    sync_id=prepare_reply.sync_id,
                    relative_path="plugins/TestPlugin/config.yml",
                    chunk_index=0,
                    total_chunks=1,
                    content_bytes=plugin_bytes,
                    sha256=plugin_sha,
                )
            ]
        )
        core_upload = runtime.upload_file(
            [
                UploadChunk(
                    sync_id=prepare_reply.sync_id,
                    relative_path="server.properties",
                    chunk_index=0,
                    total_chunks=1,
                    content_bytes=core_bytes,
                    sha256=core_sha,
                )
            ]
        )

        commit_reply = runtime.commit_sync(
            sync_id=prepare_reply.sync_id,
            server_id="lobby-1",
            server_instance_id="instance-1",
            uploaded_paths=[
                "plugins/TestPlugin/config.yml",
                "server.properties",
            ],
        )
        status_during_refresh = runtime.get_sync_status(prepare_reply.sync_id)
        runtime._plugin_semantic_service.running = False
        final_status = runtime.get_sync_status(prepare_reply.sync_id)
    finally:
        runtime.close()

    assert plugin_upload.sha256_verified is True
    assert core_upload.sha256_verified is True
    assert commit_reply.accepted_count == 2
    assert commit_reply.refresh_started is True
    assert status_during_refresh.state == SyncState.INDEXING
    assert status_during_refresh.required_file_count == 2
    assert status_during_refresh.uploaded_file_count == 2
    assert status_during_refresh.total_upload_bytes == len(plugin_bytes) + len(core_bytes)
    assert status_during_refresh.uploaded_bytes == len(plugin_bytes) + len(core_bytes)
    assert status_during_refresh.refresh_total_bundles == 1
    assert status_during_refresh.refresh_completed_bundles == 0
    assert status_during_refresh.current_refresh_bundle == "lobby-1/__server_core__"
    assert final_status.state == SyncState.COMPLETED
    assert final_status.indexed_count == 2
    assert final_status.refresh_completed_bundles == 1
    assert final_status.current_refresh_phase == "completed"
    assert (tmp_path / "mc_servers" / "lobby-1" / "plugins" / "TestPlugin" / "config.yml").read_bytes() == plugin_bytes
    assert (tmp_path / "mc_servers" / "lobby-1" / "server.properties").read_bytes() == core_bytes
