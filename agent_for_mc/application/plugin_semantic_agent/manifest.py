from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class PluginSemanticBundleState:
    server_id: str
    plugin_name: str
    fingerprint: str
    updated_at: str


@dataclass(slots=True)
class PluginSemanticManifest:
    bundles: dict[tuple[str, str], PluginSemanticBundleState] = field(
        default_factory=dict
    )

    @classmethod
    def load(cls, path: Path) -> "PluginSemanticManifest":
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return cls()
        if not isinstance(data, dict):
            return cls()
        bundles_data = data.get("bundles", [])
        bundles: dict[tuple[str, str], PluginSemanticBundleState] = {}
        if isinstance(bundles_data, list):
            for item in bundles_data:
                state = _parse_state(item)
                if state is None:
                    continue
                bundles[(state.server_id, state.plugin_name)] = state
        return cls(bundles=bundles)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "bundles": [
                {
                    "server_id": state.server_id,
                    "plugin_name": state.plugin_name,
                    "fingerprint": state.fingerprint,
                    "updated_at": state.updated_at,
                }
                for state in sorted(
                    self.bundles.values(),
                    key=lambda item: (item.server_id, item.plugin_name),
                )
            ],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def get(self, server_id: str, plugin_name: str) -> PluginSemanticBundleState | None:
        return self.bundles.get((server_id, plugin_name))

    def set(self, server_id: str, plugin_name: str, fingerprint: str) -> None:
        self.bundles[(server_id, plugin_name)] = PluginSemanticBundleState(
            server_id=server_id,
            plugin_name=plugin_name,
            fingerprint=fingerprint,
            updated_at=_utc_now(),
        )

    def remove(self, server_id: str, plugin_name: str) -> None:
        self.bundles.pop((server_id, plugin_name), None)

    def keys(self) -> set[tuple[str, str]]:
        return set(self.bundles)


def _parse_state(item: Any) -> PluginSemanticBundleState | None:
    if not isinstance(item, dict):
        return None
    server_id = str(item.get("server_id") or "").strip()
    plugin_name = str(item.get("plugin_name") or "").strip()
    fingerprint = str(item.get("fingerprint") or "").strip()
    updated_at = str(item.get("updated_at") or "").strip() or _utc_now()
    if not server_id or not plugin_name or not fingerprint:
        return None
    return PluginSemanticBundleState(
        server_id=server_id,
        plugin_name=plugin_name,
        fingerprint=fingerprint,
        updated_at=updated_at,
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
