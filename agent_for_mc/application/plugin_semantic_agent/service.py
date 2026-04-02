from __future__ import annotations

import json
import logging
import re
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any

from agent_for_mc.application.plugin_semantic_agent.manifest import (
    PluginSemanticManifest,
)
from agent_for_mc.application.plugin_semantic_agent.scanner import (
    PluginSemanticBundle,
    PluginSemanticBundleSpec,
    discover_plugin_semantic_bundle_specs,
    load_plugin_semantic_bundle,
)
from agent_for_mc.domain.models import SemanticMemoryEntry
from agent_for_mc.infrastructure.clients import JinaEmbeddingClient
from agent_for_mc.infrastructure.config import Settings
from agent_for_mc.infrastructure.observability import record_counter, trace_operation
from agent_for_mc.infrastructure.semantic_memory_vector_store import (
    ALLOWED_MEMORY_TYPES,
    ALLOWED_RELATION_TYPES,
    LanceSemanticMemoryVectorStore,
)
from agent_for_mc.interfaces.deepagent.prompts import PLUGIN_SEMANTIC_AGENT_SYSTEM_PROMPT


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class PluginSemanticExtractionResult:
    entries: list[SemanticMemoryEntry] = field(default_factory=list)


@dataclass(slots=True)
class PluginSemanticExtractionRunner:
    agent: Any

    def run(self, bundle: PluginSemanticBundle) -> PluginSemanticExtractionResult:
        with trace_operation(
            "plugin_semantic_agent.run",
            attributes={
                "component": "plugin_semantic_agent",
                "plugin": bundle.plugin_name,
                "file.count": len(bundle.files),
            },
            metric_name="rag_plugin_semantic_agent_seconds",
        ):
            record_counter("rag_plugin_semantic_agent_requests_total")
            payload = _render_plugin_semantic_agent_prompt(bundle)
            raw = _invoke_plugin_semantic_agent(
                self.agent,
                system_prompt=PLUGIN_SEMANTIC_AGENT_SYSTEM_PROMPT,
                payload=payload,
            )
            try:
                return _parse_plugin_semantic_agent_result(raw, bundle=bundle)
            except Exception as exc:
                repair_raw = _invoke_plugin_semantic_agent(
                    self.agent,
                    system_prompt=_PLUGIN_SEMANTIC_AGENT_REPAIR_PROMPT,
                    payload=_render_repair_prompt(
                        task_name="config extraction",
                        previous_output=raw,
                        error_message=str(exc),
                        expected_schema=(
                            '{"entries":[{"server_id":"[1]大厅服","plugin_name":"MMORPG",'
                            '"memory_type":"plugin_config","relation_type":"located_in",'
                            '"memory_text":"..."}]}'
                        ),
                    ),
                )
                return _parse_plugin_semantic_agent_result(repair_raw, bundle=bundle)


@dataclass(slots=True)
class PluginSemanticAgentService:
    store: LanceSemanticMemoryVectorStore
    embedding_client: JinaEmbeddingClient
    maintenance_runner: PluginSemanticExtractionRunner
    mc_servers_root: str
    manifest_path: Path
    refresh_interval_seconds: int
    max_file_chars: int
    max_files_per_plugin: int
    _executor: ThreadPoolExecutor = field(init=False, repr=False)
    _lock: Lock = field(init=False, repr=False)
    _stop_event: Event = field(init=False, repr=False)
    _refresh_thread: Thread | None = field(default=None, init=False, repr=False)
    _inflight: Future[None] | None = field(default=None, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="plugin_semantic_agent-refresh",
        )
        self._lock = Lock()
        self._stop_event = Event()
        if self.refresh_interval_seconds > 0:
            self._refresh_thread = Thread(
                target=self._run_periodic_refresh,
                name="plugin_semantic_agent-periodic",
                daemon=True,
            )
            self._refresh_thread.start()

    def refresh(self) -> None:
        self._submit_refresh(full=False)

    def refresh_full(self) -> None:
        self._submit_refresh(full=True)

    def _submit_refresh(self, *, full: bool) -> None:
        if self._closed:
            return
        with self._lock:
            if self._inflight is not None and not self._inflight.done():
                return
            future = self._executor.submit(self._run_refresh, full)
            self._inflight = future
            future.add_done_callback(self._on_refresh_done)

    def wait_for_idle(self, timeout: float | None = None) -> None:
        remaining = timeout
        while True:
            with self._lock:
                future = self._inflight
            if future is None:
                return
            try:
                future.result(timeout=remaining)
            except Exception:
                return

    def close(self, *, wait: bool = True) -> None:
        with self._lock:
            self._closed = True
        self._stop_event.set()
        if wait:
            self.wait_for_idle()
        if self._refresh_thread is not None:
            self._refresh_thread.join(timeout=5)
        self._executor.shutdown(wait=wait, cancel_futures=False)

    def _on_refresh_done(self, future: Future[None]) -> None:
        try:
            future.result()
        except Exception as exc:  # pragma: no cover - background failure
            LOGGER.warning("Plugin semantic agent refresh failed: %s", exc, exc_info=True)

        with self._lock:
            if self._inflight is future:
                self._inflight = None

    def _run_refresh(self, full: bool) -> None:
        with trace_operation(
            "plugin_semantic_agent.refresh",
            attributes={
                "component": "plugin_semantic_agent",
                "mode": "full" if full else "incremental",
            },
            metric_name="rag_plugin_semantic_agent_refresh_seconds",
        ):
            try:
                manifest = PluginSemanticManifest.load(self.manifest_path)
                specs = discover_plugin_semantic_bundle_specs(
                    Path(self.mc_servers_root),
                    max_files_per_plugin=self.max_files_per_plugin,
                )
                current_keys = {(spec.server_id, spec.plugin_name) for spec in specs}

                removed_keys = sorted(manifest.keys() - current_keys)
                for server_id, plugin_name in removed_keys:
                    self.store.delete_bundle(server_id=server_id, plugin_name=plugin_name)
                    manifest.remove(server_id, plugin_name)

                spec_map: dict[tuple[str, str], PluginSemanticBundleSpec] = {
                    (spec.server_id, spec.plugin_name): spec for spec in specs
                }
                for server_id, plugin_name in sorted(current_keys):
                    spec = spec_map[(server_id, plugin_name)]
                    previous_state = manifest.get(server_id, plugin_name)
                    if not full and previous_state is not None and previous_state.fingerprint == spec.fingerprint:
                        continue
                    try:
                        self._refresh_bundle(manifest, spec)
                    except Exception as exc:  # pragma: no cover - background failure
                        LOGGER.warning(
                            "Plugin semantic bundle refresh failed: %s / %s: %s",
                            server_id,
                            plugin_name,
                            exc,
                            exc_info=True,
                        )

                try:
                    manifest.save(self.manifest_path)
                except Exception as exc:  # pragma: no cover - background failure
                    LOGGER.warning(
                        "Plugin semantic manifest save failed: %s",
                        exc,
                        exc_info=True,
                    )
            except Exception as exc:  # pragma: no cover - background failure
                LOGGER.warning("Plugin semantic agent refresh failed: %s", exc, exc_info=True)

    def _refresh_bundle(
        self,
        manifest: PluginSemanticManifest,
        spec: PluginSemanticBundleSpec,
    ) -> None:
        bundle = load_plugin_semantic_bundle(
            spec,
            max_file_chars=self.max_file_chars,
        )
        result = self.maintenance_runner.run(bundle)
        entries = _dedupe_entries(_normalize_entries(result.entries, bundle=bundle))

        if entries:
            embeddings = [
                self.embedding_client.embed_query(entry.memory_text)
                for entry in entries
            ]
            self.store.upsert_bundle_entries(
                server_id=bundle.server_id,
                plugin_name=bundle.plugin_name,
                entries=entries,
                embeddings=embeddings,
            )
        else:
            self.store.delete_bundle(
                server_id=bundle.server_id,
                plugin_name=bundle.plugin_name,
            )

        manifest.set(bundle.server_id, bundle.plugin_name, spec.fingerprint)

    def _run_periodic_refresh(self) -> None:
        while not self._stop_event.wait(self.refresh_interval_seconds):
            if self._closed:
                return
            self.refresh()


def build_plugin_semantic_service(
    settings: Settings,
    *,
    maintenance_agent: Any | None = None,
) -> PluginSemanticAgentService | None:
    if not settings.plugin_semantic_agent_enabled:
        return None

    mc_servers_root = Path(settings.plugin_semantic_mc_servers_root)
    if not mc_servers_root.exists():
        return None

    if maintenance_agent is None:
        from agent_for_mc.interfaces.deepagent import build_plugin_semantic_agent

        maintenance_agent = build_plugin_semantic_agent(settings=settings)
    if maintenance_agent is None:
        return None

    store = LanceSemanticMemoryVectorStore(
        settings.semantic_memory_db_dir,
        settings.semantic_memory_table_name,
        expected_embedding_dimension=settings.expected_embedding_dimension,
    )
    return PluginSemanticAgentService(
        store=store,
        embedding_client=JinaEmbeddingClient(settings),
        maintenance_runner=PluginSemanticExtractionRunner(agent=maintenance_agent),
        mc_servers_root=str(mc_servers_root),
        manifest_path=settings.semantic_memory_db_dir / "plugin_semantic_manifest.json",
        refresh_interval_seconds=settings.plugin_semantic_agent_refresh_interval_seconds,
        max_file_chars=settings.plugin_semantic_agent_max_file_chars,
        max_files_per_plugin=settings.plugin_semantic_agent_max_files_per_plugin,
    )


def _normalize_entries(
    entries: list[SemanticMemoryEntry],
    *,
    bundle: PluginSemanticBundle,
) -> list[SemanticMemoryEntry]:
    normalized: list[SemanticMemoryEntry] = []
    for entry in entries:
        server_id = str(entry.server_id or bundle.server_id).strip() or bundle.server_id
        plugin_name = str(entry.plugin_name or bundle.plugin_name).strip() or bundle.plugin_name
        memory_type = str(entry.memory_type).strip().lower()
        relation_type = str(entry.relation_type).strip().lower()
        memory_text = " ".join(str(entry.memory_text).strip().split())
        if (
            not server_id
            or not plugin_name
            or not memory_text
            or memory_type not in ALLOWED_MEMORY_TYPES
            or relation_type not in ALLOWED_RELATION_TYPES
        ):
            continue
        normalized.append(
            SemanticMemoryEntry(
                server_id=server_id,
                plugin_name=plugin_name,
                memory_type=memory_type,
                relation_type=relation_type,
                memory_text=memory_text,
            )
        )
    return normalized


def _dedupe_entries(entries: list[SemanticMemoryEntry]) -> list[SemanticMemoryEntry]:
    unique: list[SemanticMemoryEntry] = []
    seen: set[tuple[str, str, str, str, str]] = set()
    for entry in entries:
        key = (
            entry.server_id,
            entry.plugin_name,
            entry.memory_type,
            entry.relation_type,
            entry.memory_text,
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(entry)
    return unique


def _render_plugin_semantic_agent_prompt(bundle: PluginSemanticBundle) -> str:
    lines: list[str] = [
        "You are plugin_semantic_agent. Produce JSON only.",
        "Extract stable semantic memories from this Minecraft plugin bundle.",
        "",
        f"server_id: {bundle.server_id}",
        f"plugin_name: {bundle.plugin_name}",
        "",
        "Return format:",
        '{"entries":[{"server_id":"...","plugin_name":"...",'
        '"memory_type":"...","relation_type":"...","memory_text":"..."}]}',
        "",
        "Files:",
    ]
    for file in bundle.files:
        lines.append(f"- path: {file.relative_path}")
        lines.append("  content:")
        lines.append(_indent(file.content, prefix="    "))
        lines.append("")
    return "\n".join(lines).strip()


def _render_repair_prompt(
    *,
    task_name: str,
    previous_output: str,
    error_message: str,
    expected_schema: str,
) -> str:
    return (
        f"Previous {task_name} output was invalid.\n"
        f"Error: {error_message}\n"
        f"Original output: {previous_output}\n\n"
        f"Return valid JSON only. Expected schema example: {expected_schema}"
    )


def _invoke_plugin_semantic_agent(
    agent: Any,
    *,
    system_prompt: str,
    payload: str,
) -> str:
    state = agent.invoke(
        {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": payload},
            ]
        }
    )
    text = _extract_agent_text(state)
    if not text:
        raise ValueError("plugin_semantic_agent returned empty output")
    return text


def _extract_agent_text(state: object) -> str:
    if isinstance(state, str):
        return state.strip()

    if isinstance(state, dict):
        messages = state.get("messages")
        if isinstance(messages, list):
            for message in reversed(messages):
                content = getattr(message, "content", None)
                if isinstance(content, str) and content.strip():
                    return content.strip()
                if isinstance(message, dict):
                    content = message.get("content")
                    if isinstance(content, str) and content.strip():
                        return content.strip()

        output = state.get("output")
        if isinstance(output, str):
            return output.strip()

    return ""


def _parse_plugin_semantic_agent_result(
    raw: str,
    *,
    bundle: PluginSemanticBundle,
) -> PluginSemanticExtractionResult:
    data = _load_json_payload(raw)
    entries: list[SemanticMemoryEntry] = []
    for item in _ensure_list(data.get("entries", [])):
        parsed = _parse_entry(item, bundle=bundle)
        if parsed is not None:
            entries.append(parsed)
    return PluginSemanticExtractionResult(entries=_dedupe_entries(entries))


def _parse_entry(
    item: Any,
    *,
    bundle: PluginSemanticBundle,
) -> SemanticMemoryEntry | None:
    if not isinstance(item, dict):
        return None
    server_id = str(item.get("server_id") or bundle.server_id).strip() or bundle.server_id
    plugin_name = str(item.get("plugin_name") or bundle.plugin_name).strip() or bundle.plugin_name
    memory_type = str(item.get("memory_type", "")).strip().lower()
    relation_type = str(item.get("relation_type", "")).strip().lower()
    memory_text = " ".join(str(item.get("memory_text", "")).strip().split())
    if (
        not server_id
        or not plugin_name
        or memory_type not in ALLOWED_MEMORY_TYPES
        or relation_type not in ALLOWED_RELATION_TYPES
        or not memory_text
    ):
        return None
    return SemanticMemoryEntry(
        server_id=server_id,
        plugin_name=plugin_name,
        memory_type=memory_type,
        relation_type=relation_type,
        memory_text=memory_text,
    )


def _load_json_payload(raw: str) -> dict[str, Any]:
    normalized = _extract_json_blob(raw)
    data = json.loads(normalized)
    if not isinstance(data, dict):
        raise ValueError("JSON root must be an object")
    return data


def _extract_json_blob(raw: str) -> str:
    text = str(raw).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"\s*```$", "", text).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start : end + 1]
    return text


def _ensure_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return []


def _indent(text: str, *, prefix: str) -> str:
    return "\n".join(f"{prefix}{line}" for line in str(text).splitlines())


_PLUGIN_SEMANTIC_AGENT_REPAIR_PROMPT = """You are plugin_semantic_agent.
The previous output was invalid.
Return only valid JSON with exactly one top-level key: entries.
"""
