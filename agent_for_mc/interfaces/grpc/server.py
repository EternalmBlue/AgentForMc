from __future__ import annotations

import logging
from concurrent import futures

import grpc

from agent_for_mc.domain.errors import ConfigurationError, StartupValidationError
from agent_for_mc.infrastructure.config import Settings
from agent_for_mc.interfaces.grpc.runtime import AgentBridgeRuntime
from agent_for_mc.interfaces.grpc.service import AgentBridgeService
from agent_for_mc.interfaces.runtime_validation import validate_runtime_settings

from . import agent_bridge_pb2_grpc


LOGGER = logging.getLogger(__name__)


def serve(settings: Settings | None = None) -> None:
    resolved_settings = settings or Settings.from_env()
    validate_runtime_settings(resolved_settings, require_grpc=True)

    runtime = AgentBridgeRuntime(resolved_settings)
    runtime.validate_startup()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=resolved_settings.grpc_max_workers))
    agent_bridge_pb2_grpc.add_AgentBridgeServiceServicer_to_server(
        AgentBridgeService(
            runtime=runtime,
            auth_token=resolved_settings.grpc_auth_token or "",
        ),
        server,
    )

    listen_address = f"{resolved_settings.grpc_host}:{resolved_settings.grpc_port}"
    bound_port = server.add_insecure_port(listen_address)
    if bound_port == 0:
        runtime.close()
        raise StartupValidationError(f"gRPC 服务监听失败: {listen_address}")

    LOGGER.info("AgentForMc gRPC bridge listening on %s", listen_address)
    server.start()
    try:
        server.wait_for_termination()
    finally:
        server.stop(grace=None)
        runtime.close()


def main() -> int:
    try:
        serve()
    except (ConfigurationError, StartupValidationError) as exc:
        print(f"[grpc startup error] {exc}")
        return 1
    except KeyboardInterrupt:
        print("\n已停止 gRPC 服务。")
        return 0
    return 0
