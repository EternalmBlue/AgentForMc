from __future__ import annotations

from pathlib import Path

import pytest

from agent_for_mc.application.plugin_semantic_agent.file_rules import (
    is_semantic_indexable_relative_path,
    normalize_allowed_config_relative_path,
)
from agent_for_mc.application.plugin_semantic_agent.scanner import (
    PLUGIN_BUNDLE_KIND,
    SERVER_CORE_BUNDLE_KIND,
    SERVER_CORE_PLUGIN_NAME,
    discover_plugin_semantic_bundle_specs,
)


def test_scanner_discovers_server_core_and_plugin_bundles(tmp_path: Path):
    server_dir = tmp_path / "lobby-1"
    plugins_dir = server_dir / "plugins" / "Essentials"
    plugins_dir.mkdir(parents=True)

    (server_dir / "server.properties").write_text("motd=hello\n", encoding="utf-8")
    (server_dir / "paper-global.yml").write_text("verbose: false\n", encoding="utf-8")
    (server_dir / "logs.txt").write_text("should be ignored\n", encoding="utf-8")
    (plugins_dir / "config.yml").write_text("homes: true\n", encoding="utf-8")

    specs = discover_plugin_semantic_bundle_specs(tmp_path, max_files_per_plugin=20)
    spec_map = {(spec.server_id, spec.plugin_name): spec for spec in specs}

    server_core_spec = spec_map[("lobby-1", SERVER_CORE_PLUGIN_NAME)]
    assert server_core_spec.bundle_kind == SERVER_CORE_BUNDLE_KIND
    assert [file.relative_path for file in server_core_spec.files] == [
        "paper-global.yml",
        "server.properties",
    ]

    plugin_spec = spec_map[("lobby-1", "Essentials")]
    assert plugin_spec.bundle_kind == PLUGIN_BUNDLE_KIND
    assert [file.relative_path for file in plugin_spec.files] == ["config.yml"]


def test_shared_file_rules_normalize_upload_paths():
    assert (
        normalize_allowed_config_relative_path(r"plugins\Essentials\config.yml")
        == "plugins/Essentials/config.yml"
    )
    assert normalize_allowed_config_relative_path("paper-global.yml") == "paper-global.yml"
    assert is_semantic_indexable_relative_path("plugins/Essentials/config.yml") is True
    assert is_semantic_indexable_relative_path("plugins/Essentials/plugin.jar") is False

    with pytest.raises(ValueError, match="drive markers"):
        normalize_allowed_config_relative_path("C:/server.properties")

    with pytest.raises(ValueError, match="must not contain"):
        normalize_allowed_config_relative_path("../server.properties")
