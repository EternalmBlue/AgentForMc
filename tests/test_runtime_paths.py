from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

from agent_for_mc.infrastructure import dotenv, runtime_paths


def test_runtime_paths_use_source_root_when_not_frozen(monkeypatch):
    monkeypatch.delattr(sys, "frozen", raising=False)
    monkeypatch.delattr(sys, "_MEIPASS", raising=False)

    assert runtime_paths.runtime_base_dir() == runtime_paths.SOURCE_ROOT
    assert runtime_paths.bundled_resource_dir() == runtime_paths.SOURCE_ROOT
    assert runtime_paths.default_config_path() == runtime_paths.SOURCE_ROOT / "config.toml"


def test_runtime_paths_use_executable_dir_when_frozen(tmp_path, monkeypatch):
    executable_dir = tmp_path / "runtime"
    bundle_dir = tmp_path / "bundle"
    executable = executable_dir / ("AgentForMc.exe" if os.name == "nt" else "AgentForMc")
    executable_dir.mkdir()
    bundle_dir.mkdir()
    executable.touch()

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(executable))
    monkeypatch.setattr(sys, "_MEIPASS", str(bundle_dir), raising=False)

    assert runtime_paths.runtime_base_dir() == executable_dir.resolve()
    assert runtime_paths.bundled_resource_dir() == bundle_dir.resolve()
    assert runtime_paths.default_config_path() == executable_dir.resolve() / "config.toml"


def test_ensure_external_runtime_layout_copies_config_and_keeps_existing(
    tmp_path,
    monkeypatch,
):
    executable_dir = tmp_path / "runtime"
    bundle_dir = tmp_path / "bundle"
    executable = executable_dir / "AgentForMc.exe"
    executable_dir.mkdir()
    bundle_dir.mkdir()
    executable.touch()
    (bundle_dir / "config.toml").write_text("[grpc]\nport = 50052\n", encoding="utf-8")

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(executable))
    monkeypatch.setattr(sys, "_MEIPASS", str(bundle_dir), raising=False)

    config_path = runtime_paths.ensure_external_runtime_layout()

    assert config_path == executable_dir.resolve() / "config.toml"
    assert config_path.read_text(encoding="utf-8") == "[grpc]\nport = 50052\n"
    assert (executable_dir / "data").is_dir()

    config_path.write_text("custom = true\n", encoding="utf-8")
    runtime_paths.ensure_external_runtime_layout()

    assert config_path.read_text(encoding="utf-8") == "custom = true\n"


def test_load_dotenv_defaults_to_runtime_base_dir(tmp_path, monkeypatch):
    executable_dir = tmp_path / "runtime"
    executable = executable_dir / "AgentForMc.exe"
    executable_dir.mkdir()
    executable.touch()
    (executable_dir / ".env").write_text(
        "RAG_GRPC_AUTH_TOKEN=runtime-token\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(executable))
    monkeypatch.delenv("RAG_ENV_FILE", raising=False)
    monkeypatch.delenv("RAG_GRPC_AUTH_TOKEN", raising=False)

    loaded_path = dotenv.load_dotenv()

    assert loaded_path == executable_dir.resolve() / ".env"
    assert os.environ["RAG_GRPC_AUTH_TOKEN"] == "runtime-token"


def test_load_runtime_config_source_uses_executable_dir_when_frozen(tmp_path, monkeypatch):
    executable_dir = tmp_path / "runtime"
    bundle_dir = tmp_path / "bundle"
    executable = executable_dir / "AgentForMc.exe"
    executable_dir.mkdir()
    bundle_dir.mkdir()
    executable.touch()
    (bundle_dir / "config.toml").write_text("[grpc]\nport = 50053\n", encoding="utf-8")

    import agent_for_mc.infrastructure.config as config_module

    with monkeypatch.context() as frozen_context:
        frozen_context.setattr(sys, "frozen", True, raising=False)
        frozen_context.setattr(sys, "executable", str(executable))
        frozen_context.setattr(sys, "_MEIPASS", str(bundle_dir), raising=False)
        frozen_context.delenv("RAG_CONFIG_TOML", raising=False)
        frozen_context.delenv("RAG_ENV_FILE", raising=False)
        frozen_config = importlib.reload(config_module)

        source = frozen_config.load_runtime_config_source()

        assert source.path == executable_dir.resolve() / "config.toml"
        assert source.base_dir == executable_dir.resolve()
        assert source.data["grpc"]["port"] == 50053
        assert (executable_dir / "data").is_dir()

    importlib.reload(config_module)
