from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


TEXT_EXTENSIONS = {".yml", ".yaml", ".json", ".properties", ".txt", ".md"}


@dataclass(slots=True)
class PluginSemanticSourceFile:
    relative_path: str
    content: str


@dataclass(slots=True)
class PluginSemanticSourceFileSpec:
    relative_path: str
    file_path: Path
    size: int
    mtime_ns: int


@dataclass(slots=True)
class PluginSemanticBundleSpec:
    server_id: str
    plugin_name: str
    plugin_dir: Path
    files: list[PluginSemanticSourceFileSpec]
    fingerprint: str


@dataclass(slots=True)
class PluginSemanticBundle:
    server_id: str
    plugin_name: str
    plugin_dir: Path
    files: list[PluginSemanticSourceFile]


def discover_plugin_semantic_bundle_specs(
    mc_servers_root: Path,
    *,
    max_files_per_plugin: int,
) -> list[PluginSemanticBundleSpec]:
    root = Path(mc_servers_root)
    if not root.exists():
        return []

    bundles: list[PluginSemanticBundleSpec] = []
    for server_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        plugins_dir = server_dir / "plugins"
        if not plugins_dir.exists():
            continue
        for plugin_dir in sorted(path for path in plugins_dir.iterdir() if path.is_dir()):
            files = _discover_plugin_files(
                plugin_dir,
                max_files_per_plugin=max_files_per_plugin,
            )
            if not files:
                continue
            bundles.append(
                PluginSemanticBundleSpec(
                    server_id=server_dir.name,
                    plugin_name=plugin_dir.name,
                    plugin_dir=plugin_dir,
                    files=files,
                    fingerprint=_fingerprint_bundle(server_dir.name, plugin_dir.name, files),
                )
            )
    return bundles


def load_plugin_semantic_bundle(
    spec: PluginSemanticBundleSpec,
    *,
    max_file_chars: int,
) -> PluginSemanticBundle:
    return PluginSemanticBundle(
        server_id=spec.server_id,
        plugin_name=spec.plugin_name,
        plugin_dir=spec.plugin_dir,
        files=[
            PluginSemanticSourceFile(
                relative_path=file_spec.relative_path,
                content=_read_text_limited(file_spec.file_path, max_file_chars=max_file_chars),
            )
            for file_spec in spec.files
        ],
    )


def scan_plugin_semantic_bundles(
    mc_servers_root: Path,
    *,
    max_file_chars: int,
    max_files_per_plugin: int,
) -> list[PluginSemanticBundle]:
    specs = discover_plugin_semantic_bundle_specs(
        mc_servers_root,
        max_files_per_plugin=max_files_per_plugin,
    )
    return [
        load_plugin_semantic_bundle(spec, max_file_chars=max_file_chars)
        for spec in specs
    ]


def _discover_plugin_files(
    plugin_dir: Path,
    *,
    max_files_per_plugin: int,
) -> list[PluginSemanticSourceFileSpec]:
    files: list[PluginSemanticSourceFileSpec] = []
    for file_path in sorted(path for path in plugin_dir.rglob("*") if path.is_file()):
        if file_path.suffix.lower() not in TEXT_EXTENSIONS:
            continue
        try:
            stat = file_path.stat()
        except OSError:
            continue
        files.append(
            PluginSemanticSourceFileSpec(
                relative_path=str(file_path.relative_to(plugin_dir)).replace("\\", "/"),
                file_path=file_path,
                size=stat.st_size,
                mtime_ns=stat.st_mtime_ns,
            )
        )
        if 0 < max_files_per_plugin <= len(files):
            break
    return files


def _fingerprint_bundle(
    server_id: str,
    plugin_name: str,
    files: list[PluginSemanticSourceFileSpec],
) -> str:
    digest = hashlib.sha256()
    digest.update(server_id.encode("utf-8"))
    digest.update(b"\0")
    digest.update(plugin_name.encode("utf-8"))
    for file_spec in files:
        digest.update(b"\0")
        digest.update(file_spec.relative_path.encode("utf-8"))
        digest.update(b"|")
        digest.update(str(file_spec.size).encode("utf-8"))
        digest.update(b"|")
        digest.update(str(file_spec.mtime_ns).encode("utf-8"))
    return digest.hexdigest()


def _read_text_limited(path: Path, *, max_file_chars: int) -> str:
    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""

    normalized = content.strip()
    if max_file_chars > 0 and len(normalized) > max_file_chars:
        normalized = normalized[:max_file_chars].rstrip() + "\n...[truncated]"
    return normalized
