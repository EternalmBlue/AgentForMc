from __future__ import annotations

from pathlib import Path, PurePosixPath


TEXT_CONFIG_EXTENSIONS = frozenset({".yml", ".yaml", ".json", ".properties", ".txt", ".md"})
CORE_SERVER_CONFIG_FILES = frozenset({"server.properties", "bukkit.yml", "spigot.yml"})
PAPER_CONFIG_PREFIX = "paper"
PAPER_CONFIG_SUFFIX = ".yml"


def is_text_config_file(file_name: str | Path) -> bool:
    return Path(str(file_name)).suffix.lower() in TEXT_CONFIG_EXTENSIONS


def is_server_core_config_file(file_name: str) -> bool:
    lowered = str(file_name).lower()
    return lowered in CORE_SERVER_CONFIG_FILES or (
        lowered.startswith(PAPER_CONFIG_PREFIX) and lowered.endswith(PAPER_CONFIG_SUFFIX)
    )


def normalize_allowed_config_relative_path(relative_path: str) -> str:
    normalized = str(relative_path or "").replace("\\", "/").strip()
    if not normalized:
        raise ValueError("relative_path must not be blank")
    if normalized.startswith("/"):
        raise ValueError("relative_path must be relative")

    raw_path = PurePosixPath(normalized)
    parts = raw_path.parts
    if not parts:
        raise ValueError("relative_path must not be blank")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("relative_path must not contain empty segments, '.' or '..'")
    if any(":" in part for part in parts):
        raise ValueError("relative_path must not contain drive markers")

    normalized_path = "/".join(parts)
    if parts[0] == "plugins":
        if not is_text_config_file(parts[-1]):
            raise ValueError("only text config files under plugins/ are allowed")
        return normalized_path

    if len(parts) == 1 and is_server_core_config_file(parts[0]):
        return normalized_path

    raise ValueError("only plugin config files and allowed core server configs may be uploaded")


def is_semantic_indexable_relative_path(relative_path: str) -> bool:
    try:
        normalize_allowed_config_relative_path(relative_path)
    except ValueError:
        return False
    return True
