from __future__ import annotations

import json
import re
from dataclasses import dataclass

from langchain_core.tools import tool

from agent_for_mc.infrastructure.clients import DeepSeekChatClient
from agent_for_mc.infrastructure.observability import record_counter, trace_operation
from agent_for_mc.infrastructure.shared_context import SharedContextSlot


WHITESPACE_RE = re.compile(r"\s+")


PLUGIN_CONFIG_ROUTER_SYSTEM_PROMPT = """
You are a routing tool for a Minecraft plugin assistant.

Return JSON only with this schema:
{"route": string, "use_subagent": boolean, "normalized_query": string, "reason": string, "confidence": number}

Rules:
- Use route "plugin_config_agent" when the question is about plugin config files, default values,
  file paths, dependency wiring, configuration differences, or file-level YAML/TOML/JSON settings.
- Use route "main_agent" for all other questions.
- normalized_query should be a concise standalone query suitable for the plugin_config_agent.
- Keep output JSON only. Do not add explanations, markdown, or code fences.
""".strip()


@dataclass(slots=True)
class PluginConfigRoutingToolContext:
    client: DeepSeekChatClient


_TOOL_CONTEXT = SharedContextSlot[PluginConfigRoutingToolContext](
    "plugin_config_routing_tool_context"
)


def configure_plugin_config_routing_tool(context: PluginConfigRoutingToolContext) -> None:
    _TOOL_CONTEXT.set(context)


def get_plugin_config_routing_tool_context() -> PluginConfigRoutingToolContext:
    return _TOOL_CONTEXT.get(
        error_message="plugin config routing tool context has not been configured"
    )


@tool("route_plugin_config_request")
def route_plugin_config_request(
    question: str,
    history: str = "",
) -> str:
    """Route plugin configuration questions to the dedicated subagent or main agent."""
    with trace_operation(
        "tool.route_plugin_config_request",
        attributes={"component": "tool", "question.length": len(question.strip())},
        metric_name="rag_tool_route_plugin_config_request_seconds",
    ):
        record_counter("rag_tool_route_plugin_config_request_requests_total")
        context = get_plugin_config_routing_tool_context()
        messages = [
            {"role": "system", "content": PLUGIN_CONFIG_ROUTER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Recent conversation:\n{history.strip() or 'No history.'}\n\n"
                    f"User question:\n{question.strip()}\n"
                ),
            },
        ]
        content = context.client.chat(messages, temperature=0.0)
        parsed = _parse_routing_result(content, fallback_question=question)
        return json.dumps(parsed, ensure_ascii=False)


def _parse_routing_result(content: str, *, fallback_question: str) -> dict[str, object]:
    normalized = content.strip()
    try:
        data = json.loads(normalized)
    except json.JSONDecodeError:
        return {
            "route": "main_agent",
            "use_subagent": False,
            "normalized_query": _normalize_text(fallback_question),
            "reason": "Invalid routing output; defaulted to main_agent.",
            "confidence": 0.0,
        }

    route = _normalize_text(data.get("route") or data.get("target") or data.get("decision"))
    use_subagent = _parse_bool(data.get("use_subagent"))
    normalized_query = _normalize_text(
        data.get("normalized_query")
        or data.get("search_query")
        or data.get("query")
        or fallback_question
    )
    reason = _normalize_text(data.get("reason") or data.get("explanation"))
    confidence = _parse_float(data.get("confidence"), fallback=0.0)

    if route not in {"plugin_config_agent", "main_agent"}:
        route = "plugin_config_agent" if use_subagent else "main_agent"

    if route == "plugin_config_agent":
        use_subagent = True
        if not normalized_query:
            normalized_query = _normalize_text(fallback_question)
    else:
        use_subagent = False
        normalized_query = _normalize_text(fallback_question)

    if not reason:
        reason = (
            "Routed to plugin_config_agent."
            if route == "plugin_config_agent"
            else "Routed to main_agent."
        )

    return {
        "route": route,
        "use_subagent": use_subagent,
        "normalized_query": normalized_query,
        "reason": reason,
        "confidence": confidence,
    }


def _parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1"}
    return bool(value)


def _parse_float(value: object, *, fallback: float) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return fallback
    return fallback


def _normalize_text(value: object) -> str:
    return WHITESPACE_RE.sub(" ", str(value)).strip()
