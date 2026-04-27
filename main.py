from __future__ import annotations

import argparse
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AgentForMc gRPC backend")
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="verify packaged runtime files without starting the gRPC service",
    )
    args = parser.parse_args(argv)

    if args.self_check:
        from agent_for_mc.infrastructure.runtime_paths import run_runtime_self_check

        print(run_runtime_self_check().render())
        return 0

    from agent_for_mc.interfaces.grpc.server import main as grpc_main

    return grpc_main()


if __name__ == "__main__":
    raise SystemExit(main())
