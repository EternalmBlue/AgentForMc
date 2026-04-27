from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_FILE_NAME = "config.toml"
ENV_EXAMPLE_FILE_NAME = ".env.example"
DATA_DIR_NAME = "data"


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def runtime_base_dir() -> Path:
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return SOURCE_ROOT


def bundled_resource_dir() -> Path:
    if is_frozen() and hasattr(sys, "_MEIPASS"):
        return Path(getattr(sys, "_MEIPASS")).resolve()
    return SOURCE_ROOT


def bundled_resource_path(relative_path: str | Path) -> Path:
    return (bundled_resource_dir() / relative_path).resolve()


def default_config_path() -> Path:
    return runtime_base_dir() / CONFIG_FILE_NAME


def default_dotenv_path() -> Path:
    return runtime_base_dir() / ".env"


def resolve_runtime_path(path_value: str | Path, *, base_dir: Path | None = None) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = (base_dir or runtime_base_dir()) / path
    return path.resolve()


def ensure_external_runtime_layout(
    *,
    config_path: Path | None = None,
    copy_default_config: bool = True,
) -> Path:
    target_config_path = (config_path or default_config_path()).resolve()
    target_base_dir = target_config_path.parent
    target_base_dir.mkdir(parents=True, exist_ok=True)
    (target_base_dir / DATA_DIR_NAME).mkdir(parents=True, exist_ok=True)

    if copy_default_config and not target_config_path.exists():
        bundled_config = bundled_resource_path(CONFIG_FILE_NAME)
        if not bundled_config.is_file():
            raise FileNotFoundError(f"Bundled config template not found: {bundled_config}")
        shutil.copyfile(bundled_config, target_config_path)

    return target_config_path


@dataclass(frozen=True, slots=True)
class RuntimeSelfCheck:
    frozen: bool
    runtime_base_dir: Path
    bundled_resource_dir: Path
    config_path: Path
    data_dir: Path
    bundled_config_path: Path

    def render(self) -> str:
        return "\n".join(
            [
                "AgentForMc self-check OK",
                f"frozen={self.frozen}",
                f"runtime_base_dir={self.runtime_base_dir}",
                f"bundled_resource_dir={self.bundled_resource_dir}",
                f"config_path={self.config_path}",
                f"data_dir={self.data_dir}",
                f"bundled_config_path={self.bundled_config_path}",
            ]
        )


def run_runtime_self_check() -> RuntimeSelfCheck:
    config_path = ensure_external_runtime_layout()
    data_dir = config_path.parent / DATA_DIR_NAME
    bundled_config = bundled_resource_path(CONFIG_FILE_NAME)

    if not config_path.is_file():
        raise RuntimeError(f"External config file is not available: {config_path}")
    if not data_dir.is_dir():
        raise RuntimeError(f"External data directory is not available: {data_dir}")
    if not bundled_config.is_file():
        raise RuntimeError(f"Bundled config template is not available: {bundled_config}")

    return RuntimeSelfCheck(
        frozen=is_frozen(),
        runtime_base_dir=runtime_base_dir(),
        bundled_resource_dir=bundled_resource_dir(),
        config_path=config_path,
        data_dir=data_dir,
        bundled_config_path=bundled_config,
    )
