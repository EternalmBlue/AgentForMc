from __future__ import annotations

from agent_for_mc.infrastructure.observability import (
    configure_observability,
    load_observability_config,
    trace_operation,
)


def test_load_observability_config_reads_env(monkeypatch):
    monkeypatch.setenv("RAG_LANGSMITH_ENABLED", "true")
    monkeypatch.setenv("RAG_LANGSMITH_API_KEY", "ls-key")
    monkeypatch.setenv("RAG_LANGSMITH_PROJECT", "agent-for-mc")
    monkeypatch.setenv("RAG_LANGSMITH_ENDPOINT", "https://example.com")
    monkeypatch.setenv("RAG_OTEL_ENABLED", "true")
    monkeypatch.setenv("RAG_OTEL_SERVICE_NAME", "agent-for-mc")
    monkeypatch.setenv("RAG_OTEL_CONSOLE_EXPORT", "false")

    config = load_observability_config()

    assert config.langsmith_enabled is True
    assert config.langsmith_api_key == "ls-key"
    assert config.langsmith_project == "agent-for-mc"
    assert config.langsmith_endpoint == "https://example.com"
    assert config.otel_enabled is True
    assert config.otel_service_name == "agent-for-mc"
    assert config.otel_console_export is False


def test_configure_observability_is_idempotent(monkeypatch):
    monkeypatch.setenv("RAG_LANGSMITH_ENABLED", "true")
    monkeypatch.setenv("RAG_LANGSMITH_API_KEY", "ls-key")
    monkeypatch.setenv("RAG_LANGSMITH_PROJECT", "agent-for-mc")
    monkeypatch.setenv("RAG_LANGSMITH_ENDPOINT", "https://example.com")
    monkeypatch.setenv("RAG_OTEL_ENABLED", "true")
    monkeypatch.setenv("RAG_OTEL_SERVICE_NAME", "agent-for-mc")
    monkeypatch.setenv("RAG_OTEL_CONSOLE_EXPORT", "false")
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
