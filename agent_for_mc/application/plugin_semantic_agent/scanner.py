from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


TEXT_EXTENSIONS = {".yml", ".yaml", ".json", ".properties", ".txt", ".md"}


@dataclass(slots=True)
class PluginSemanticSourceFile:
    relative_path: str
    content: str


@dataclass(slots=True)
class PluginSemanticBundle:
    server_id: str
    plugin_name: str
    plugin_dir: Path
    files: list[PluginSemanticSourceFile]


def scan_plugin_semantic_bundles(
    mc_servers_root: Path,
    *,
    max_file_chars: int,
    max_files_per_plugin: int,
) -> list[PluginSemanticBundle]:
    root = Path(mc_servers_root)
    if not root.exists():
        return []

    bundles: list[PluginSemanticBundle] = []
    for server_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        plugins_dir = server_dir / "plugins"
        if not plugins_dir.exists():
            continue
        for plugin_dir in sorted(path for path in plugins_dir.iterdir() if path.is_dir()):
            files = _scan_plugin_files(
                plugin_dir,
                max_file_chars=max_file_chars,
                max_files_per_plugin=max_files_per_plugin,
            )
            if not files:
                continue
            bundles.append(
                PluginSemanticBundle(
                    server_id=server_dir.name,
                    plugin_name=plugin_dir.name,
                    plugin_dir=plugin_dir,
                    files=files,
                )
            )
    return bundles


def _scan_plugin_files(
    plugin_dir: Path,
    *,
    max_file_chars: int,
    max_files_per_plugin: int,
) -> list[PluginSemanticSourceFile]:
    files: list[PluginSemanticSourceFile] = []
    for file_path in sorted(path for path in plugin_dir.rglob("*") if path.is_file()):
        if file_path.suffix.lower() not in TEXT_EXTENSIONS:
            continue
        content = _read_text_limited(file_path, max_file_chars=max_file_chars)
        files.append(
            PluginSemanticSourceFile(
                relative_path=str(file_path.relative_to(plugin_dir)).replace("\\", "/"),
                content=content,
            )
        )
        if 0 < max_files_per_plugin <= len(files):
            break
    return files


def _read_text_limited(path: Path, *, max_file_chars: int) -> str:
    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""

    normalized = content.strip()
    if max_file_chars > 0 and len(normalized) > max_file_chars:
        normalized = normalized[:max_file_chars].rstrip() + "\n...[truncated]"
    return normalized
