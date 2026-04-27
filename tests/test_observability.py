from __future__ import annotations

from pathlib import Path

from agent_for_mc.infrastructure.observability import (
    configure_observability,
    load_observability_config,
    trace_operation,
)


def _write_observability_config(path: Path) -> None:
    path.write_text(
        """
[observability]
langsmith_enabled = true
langsmith_project = "agent-for-mc"
langsmith_endpoint = "https://example.com"
otel_enabled = true
otel_service_name = "agent-for-mc"
otel_exporter_otlp_endpoint = "https://otel.example.com"
otel_exporter_otlp_protocol = "http/protobuf"
otel_console_export = false
""".strip(),
        encoding="utf-8",
    )


def test_load_observability_config_reads_config_and_secret_env(tmp_path, monkeypatch):
    config_path = tmp_path / "config.toml"
    env_path = tmp_path / ".env"
    _write_observability_config(config_path)
    env_path.write_text("", encoding="utf-8")

    monkeypatch.setenv("RAG_CONFIG_TOML", str(config_path))
    monkeypatch.setenv("RAG_ENV_FILE", str(env_path))
    monkeypatch.setenv("RAG_LANGSMITH_API_KEY", "ls-key")
    monkeypatch.setenv("RAG_OTEL_EXPORTER_OTLP_HEADERS", "authorization=Bearer secret")

    config = load_observability_config()

    assert config.langsmith_enabled is True
    assert config.langsmith_api_key == "ls-key"
    assert config.langsmith_project == "agent-for-mc"
    assert config.langsmith_endpoint == "https://example.com"
    assert config.otel_enabled is True
    assert config.otel_service_name == "agent-for-mc"
    assert config.otel_exporter_otlp_endpoint == "https://otel.example.com"
    assert config.otel_exporter_otlp_headers == "authorization=Bearer secret"
    assert config.otel_console_export is False


def test_configure_observability_is_idempotent(tmp_path, monkeypatch):
    config_path = tmp_path / "config.toml"
    env_path = tmp_path / ".env"
    _write_observability_config(config_path)
    env_path.write_text("", encoding="utf-8")

    monkeypatch.setenv("RAG_CONFIG_TOML", str(config_path))
    monkeypatch.setenv("RAG_ENV_FILE", str(env_path))
    monkeypatch.setenv("RAG_LANGSMITH_API_KEY", "ls-key")
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)

    config = configure_observability(service_name="agent-for-mc")
    assert config.langsmith_enabled is True
    assert config.otel_enabled is True
    assert config.otel_service_name == "agent-for-mc"
    assert config.langsmith_project == "agent-for-mc"
    assert config.langsmith_endpoint == "https://example.com"
    assert config.otel_console_export is False

    with trace_operation("test.operation"):
        pass
