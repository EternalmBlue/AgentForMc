from __future__ import annotations

import logging
import os
import re
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Iterator

from opentelemetry import _logs, metrics, trace
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor, ConsoleLogExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import ConsoleMetricExporter, PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter


LOGGER = logging.getLogger(__name__)

DEFAULT_SERVICE_NAME = "agent_for_mc"
DEFAULT_LANGSMITH_PROJECT = "AgentForMc"
DEFAULT_LANGSMITH_ENDPOINT = "https://api.smith.langchain.com"

_CONFIG_LOCK = threading.Lock()
_CONFIGURED = False
_REQUESTS_INSTRUMENTED = False
_LOG_HANDLER_INSTALLED = False


@dataclass(frozen=True, slots=True)
class ObservabilityConfig:
    langsmith_enabled: bool
    langsmith_api_key: str | None
    langsmith_project: str
    langsmith_endpoint: str
    otel_enabled: bool
    otel_service_name: str
    otel_exporter_otlp_endpoint: str | None
    otel_exporter_otlp_headers: str | None
    otel_exporter_otlp_protocol: str
    otel_console_export: bool


def load_observability_config() -> ObservabilityConfig:
    langsmith_api_key = _get_env(
        "RAG_LANGSMITH_API_KEY",
        "LANGSMITH_API_KEY",
        "LANGCHAIN_API_KEY",
    )
    langsmith_enabled = _parse_bool(
        _get_env(
            "RAG_LANGSMITH_ENABLED",
            "LANGSMITH_TRACING",
            "LANGCHAIN_TRACING_V2",
        ),
        default=bool(langsmith_api_key),
    )
    langsmith_project = _get_env(
        "RAG_LANGSMITH_PROJECT",
        "LANGSMITH_PROJECT",
        "LANGCHAIN_PROJECT",
        default=DEFAULT_LANGSMITH_PROJECT,
    )
    langsmith_endpoint = _get_env(
        "RAG_LANGSMITH_ENDPOINT",
        "LANGSMITH_ENDPOINT",
        "LANGCHAIN_ENDPOINT",
        default=DEFAULT_LANGSMITH_ENDPOINT,
    )

    otel_exporter_otlp_endpoint = _get_env(
        "RAG_OTEL_EXPORTER_OTLP_ENDPOINT",
        "OTEL_EXPORTER_OTLP_ENDPOINT",
    )
    otel_console_export = _parse_bool(
        _get_env("RAG_OTEL_CONSOLE_EXPORT", "OTEL_CONSOLE_EXPORT"),
        default=False,
    )
    raw_otel_enabled = _get_env("RAG_OTEL_ENABLED")
    if raw_otel_enabled is not None:
        otel_enabled = _parse_bool(raw_otel_enabled)
    else:
        raw_sdk_disabled = _get_env("OTEL_SDK_DISABLED")
        if raw_sdk_disabled is not None:
            otel_enabled = not _parse_bool(raw_sdk_disabled)
        else:
            otel_enabled = bool(otel_exporter_otlp_endpoint) or otel_console_export

    return ObservabilityConfig(
        langsmith_enabled=langsmith_enabled,
        langsmith_api_key=langsmith_api_key,
        langsmith_project=langsmith_project,
        langsmith_endpoint=langsmith_endpoint,
        otel_enabled=otel_enabled,
        otel_service_name=_get_env(
            "RAG_OTEL_SERVICE_NAME",
            "OTEL_SERVICE_NAME",
            default=DEFAULT_SERVICE_NAME,
        ),
        otel_exporter_otlp_endpoint=otel_exporter_otlp_endpoint,
        otel_exporter_otlp_headers=_get_env(
            "RAG_OTEL_EXPORTER_OTLP_HEADERS",
            "OTEL_EXPORTER_OTLP_HEADERS",
        ),
        otel_exporter_otlp_protocol=_get_env(
            "RAG_OTEL_EXPORTER_OTLP_PROTOCOL",
            "OTEL_EXPORTER_OTLP_PROTOCOL",
            default="http/protobuf",
        ),
        otel_console_export=otel_console_export,
    )


def configure_observability(*, service_name: str = DEFAULT_SERVICE_NAME) -> ObservabilityConfig:
    config = load_observability_config()
    with _CONFIG_LOCK:
        _configure_langsmith(config)
        _configure_otel(config, service_name=service_name)
    return config


def get_tracer(name: str = "agent_for_mc.observability"):
    return trace.get_tracer(name)


def get_meter(name: str = "agent_for_mc.observability"):
    return metrics.get_meter(name)


@contextmanager
def trace_operation(
    name: str,
    *,
    attributes: dict[str, Any] | None = None,
    metric_name: str | None = None,
) -> Iterator[Any]:
    start = perf_counter()
    tracer = get_tracer()
    with tracer.start_as_current_span(name) as span:
        _set_attributes(span, attributes)
        try:
            yield span
        finally:
            if metric_name:
                record_duration(metric_name, perf_counter() - start, attributes=attributes)


def record_counter(
    name: str,
    amount: float = 1.0,
    *,
    attributes: dict[str, Any] | None = None,
) -> None:
    counter = get_meter().create_counter(name)
    counter.add(amount, attributes=attributes)


def record_duration(
    name: str,
    duration_seconds: float,
    *,
    attributes: dict[str, Any] | None = None,
) -> None:
    histogram = get_meter().create_histogram(name, unit="s")
    histogram.record(duration_seconds, attributes=attributes)


def _configure_langsmith(config: ObservabilityConfig) -> None:
    if not config.langsmith_enabled or not config.langsmith_api_key:
        return

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = config.langsmith_api_key
    os.environ["LANGSMITH_API_KEY"] = config.langsmith_api_key
    os.environ["LANGCHAIN_PROJECT"] = config.langsmith_project
    os.environ["LANGSMITH_PROJECT"] = config.langsmith_project
    os.environ["LANGCHAIN_ENDPOINT"] = config.langsmith_endpoint
    os.environ["LANGSMITH_ENDPOINT"] = config.langsmith_endpoint


def _configure_otel(config: ObservabilityConfig, *, service_name: str) -> None:
    global _CONFIGURED, _REQUESTS_INSTRUMENTED, _LOG_HANDLER_INSTALLED

    if not config.otel_enabled or _CONFIGURED:
        return

    resource_attributes: dict[str, Any] = {
        "service.name": service_name or config.otel_service_name or DEFAULT_SERVICE_NAME,
        "service.namespace": "agent_for_mc",
    }
    resource = Resource.create(resource_attributes)
    headers = _parse_headers(config.otel_exporter_otlp_headers)

    tracer_provider = TracerProvider(resource=resource)
    span_exporter = _build_trace_exporter(config, headers=headers)
    if span_exporter is not None:
        tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter))
    trace.set_tracer_provider(tracer_provider)

    metric_readers = _build_metric_readers(config, headers=headers)
    metrics.set_meter_provider(MeterProvider(metric_readers=metric_readers, resource=resource))

    logger_provider = LoggerProvider(resource=resource)
    log_exporter = _build_log_exporter(config, headers=headers)
    if log_exporter is not None:
        logger_provider.add_log_record_processor(BatchLogRecordProcessor(log_exporter))
        _install_logging_handler(logger_provider)
    _logs.set_logger_provider(logger_provider)

    if not _REQUESTS_INSTRUMENTED:
        RequestsInstrumentor().instrument()
        _REQUESTS_INSTRUMENTED = True

    _CONFIGURED = True


def _build_trace_exporter(
    config: ObservabilityConfig,
    *,
    headers: dict[str, str] | None,
) -> Any | None:
    exporters: list[Any] = []
    if config.otel_exporter_otlp_endpoint:
        exporters.append(
            OTLPSpanExporter(
                endpoint=config.otel_exporter_otlp_endpoint,
                headers=headers,
            )
        )
    if config.otel_console_export:
        exporters.append(ConsoleSpanExporter())
    return _combine_exporters(exporters)


def _build_metric_readers(
    config: ObservabilityConfig,
    *,
    headers: dict[str, str] | None,
) -> list[PeriodicExportingMetricReader]:
    readers: list[PeriodicExportingMetricReader] = []
    if config.otel_exporter_otlp_endpoint:
        readers.append(
            PeriodicExportingMetricReader(
                OTLPMetricExporter(
                    endpoint=config.otel_exporter_otlp_endpoint,
                    headers=headers,
                )
            )
        )
    if config.otel_console_export:
        readers.append(
            PeriodicExportingMetricReader(ConsoleMetricExporter())
        )
    return readers


def _build_log_exporter(
    config: ObservabilityConfig,
    *,
    headers: dict[str, str] | None,
) -> Any | None:
    exporters: list[Any] = []
    if config.otel_exporter_otlp_endpoint:
        exporters.append(
            OTLPLogExporter(
                endpoint=config.otel_exporter_otlp_endpoint,
                headers=headers,
            )
        )
    if config.otel_console_export:
        exporters.append(ConsoleLogExporter())
    return _combine_exporters(exporters)


def _combine_exporters(exporters: list[Any]) -> Any | None:
    if not exporters:
        return None
    if len(exporters) == 1:
        return exporters[0]

    class _CompositeExporter:
        def __init__(self, children: list[Any]):
            self._children = children

        def export(self, items: Any) -> Any:
            result = None
            for exporter in self._children:
                result = exporter.export(items)
            return result

        def force_flush(self, timeout_millis: int | None = None) -> bool:
            flushed = True
            for exporter in self._children:
                force_flush = getattr(exporter, "force_flush", None)
                if callable(force_flush):
                    flushed = bool(force_flush(timeout_millis=timeout_millis)) and flushed
            return flushed

        def shutdown(self) -> None:
            for exporter in self._children:
                shutdown = getattr(exporter, "shutdown", None)
                if callable(shutdown):
                    shutdown()

    return _CompositeExporter(exporters)


def _install_logging_handler(logger_provider: LoggerProvider) -> None:
    global _LOG_HANDLER_INSTALLED
    if _LOG_HANDLER_INSTALLED:
        return

    handler = LoggingHandler(level=logging.NOTSET, logger_provider=logger_provider)
    logging.getLogger().addHandler(handler)
    _LOG_HANDLER_INSTALLED = True


def _set_attributes(span: Any, attributes: dict[str, Any] | None) -> None:
    if attributes is None:
        return
    for key, value in attributes.items():
        if value is None:
            continue
        if isinstance(value, (str, bool, int, float)):
            span.set_attribute(key, value)
        else:
            span.set_attribute(key, str(value))


def _get_env(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value is not None and value.strip():
            return value.strip()
    return default


def _parse_bool(value: str | None, *, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_headers(value: str | None) -> dict[str, str] | None:
    if not value:
        return None

    headers: dict[str, str] = {}
    for chunk in re.split(r"[;,]", value):
        item = chunk.strip()
        if not item:
            continue
        key, separator, header_value = item.partition("=")
        if not separator:
            continue
        key = key.strip()
        header_value = header_value.strip()
        if key and header_value:
            headers[key] = header_value
    return headers or None
