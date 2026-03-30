from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: Path | None = None) -> Path | None:
    dotenv_path = path or _default_dotenv_path()
    if dotenv_path is None or not dotenv_path.exists():
        return None

    _load_dotenv_file(dotenv_path)
    return dotenv_path


def _default_dotenv_path() -> Path | None:
    base_dir = Path(__file__).resolve().parent.parent.parent
    env_file = os.getenv("RAG_ENV_FILE")
    if env_file:
        return Path(env_file).expanduser()
    return base_dir / ".env"


def _load_dotenv_file(dotenv_path: Path) -> None:
    base_dir = dotenv_path.parent
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" in line:
            key, value = line.split("=", 1)
        else:
            key, value = line, ""
        key = key.strip()
        if not key or key in os.environ:
            continue
        os.environ[key] = _normalize_value(key, value.strip(), base_dir)


def _normalize_value(key: str, value: str, base_dir: Path) -> str:
    unquoted = _strip_quotes(value)
    if not unquoted:
        return ""

    if _looks_like_path_key(key):
        return str(_resolve_path(unquoted, base_dir))
    return unquoted


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def _looks_like_path_key(key: str) -> bool:
    normalized = key.upper()
    return normalized.endswith("_DIR") or normalized.endswith("_PATH") or normalized in {
        "RAG_CONFIG_TOML",
    }


def _resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()
