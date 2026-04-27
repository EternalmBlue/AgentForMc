from __future__ import annotations

from typing import Any, Protocol

import requests

from agent_for_mc.domain.errors import ConfigurationError, ServiceError
from agent_for_mc.infrastructure.config import Settings
from agent_for_mc.infrastructure.observability import record_counter, trace_operation


SUPPORTED_ZHIPU_EMBEDDING_DIMENSIONS = {256, 512, 1024, 2048}


class EmbeddingClient(Protocol):
    def embed_query(self, text: str) -> list[float]:
        ...


def build_embedding_client(settings: Settings) -> EmbeddingClient:
    return OpenAICompatibleEmbeddingClient(settings)


def validate_embedding_settings(settings: Settings) -> None:
    if not settings.resolved_embedding_api_key:
        raise ConfigurationError(
            "Missing embedding API key. "
            f"Set the env var {settings.resolved_embedding_api_key_env}."
        )
    if not settings.resolved_embedding_url:
        raise ConfigurationError("embedding.url must not be blank.")
    if not settings.resolved_embedding_model:
        raise ConfigurationError("embedding.model must not be blank.")
    if settings.resolved_embedding_dimensions not in SUPPORTED_ZHIPU_EMBEDDING_DIMENSIONS:
        raise ConfigurationError(
            "embedding.dimensions must be one of "
            f"{', '.join(str(value) for value in sorted(SUPPORTED_ZHIPU_EMBEDDING_DIMENSIONS))} "
            "when using Zhipu embedding-3."
        )


class OpenAICompatibleEmbeddingClient:
    def __init__(self, settings: Settings):
        self._settings = settings

    def embed_query(self, text: str) -> list[float]:
        with trace_operation(
            "embedding.embed_query",
            attributes={
                "component": "embedding",
                "provider": "zhipu-openai-compatible",
                "model": self._settings.resolved_embedding_model,
                "dimensions": self._settings.resolved_embedding_dimensions,
            },
            metric_name="rag_embedding_embed_seconds",
        ):
            record_counter("rag_embedding_requests_total")
            api_key = self._settings.resolved_embedding_api_key
            if not api_key:
                raise ConfigurationError(
                    "Missing embedding API key. "
                    f"Set the env var {self._settings.resolved_embedding_api_key_env}."
                )

            response = _post_json(
                self._settings.resolved_embedding_url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                payload={
                    "model": self._settings.resolved_embedding_model,
                    "input": text,
                    "dimensions": self._settings.resolved_embedding_dimensions,
                },
                timeout_seconds=self._settings.request_timeout_seconds,
                service_name="Zhipu embedding",
            )
            data = _read_json(response, "Zhipu embedding")
            return _extract_embedding(data, "Zhipu embedding")


class DeepSeekChatClient:
    def __init__(self, settings: Settings, model_name: str | None = None):
        self._settings = settings
        self._model_name = model_name or settings.deepseek_model

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.1,
    ) -> str:
        with trace_operation(
            "deepseek.chat",
            attributes={"component": "llm", "model": self._model_name},
            metric_name="rag_deepseek_chat_seconds",
        ):
            record_counter("rag_deepseek_chat_requests_total")
            if not self._settings.deepseek_api_key:
                raise ConfigurationError("Missing environment variable RAG_DEEPSEEK_API_KEY.")

            response = _post_json(
                self._settings.deepseek_chat_url,
                headers={
                    "Authorization": f"Bearer {self._settings.deepseek_api_key}",
                    "Content-Type": "application/json",
                },
                payload={
                    "model": self._model_name,
                    "messages": messages,
                    "temperature": temperature,
                },
                timeout_seconds=self._settings.request_timeout_seconds,
                service_name="DeepSeek",
            )
            data = _read_json(response, "DeepSeek")
            try:
                content = data["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError) as exc:
                raise ServiceError("DeepSeek response is missing choices[0].message.content.") from exc

            normalized = str(content).strip()
            if not normalized:
                raise ServiceError("DeepSeek returned an empty response.")
            return normalized


def _post_json(
    url: str,
    *,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout_seconds: int,
    service_name: str,
) -> requests.Response:
    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        return response
    except requests.RequestException as exc:
        detail = _extract_error_detail(exc.response)
        suffix = f": {detail}" if detail else f": {exc}"
        raise ServiceError(f"{service_name} request failed{suffix}") from exc


def _read_json(response: requests.Response, service_name: str) -> dict[str, Any]:
    try:
        data = response.json()
    except ValueError as exc:
        preview = response.text[:200].strip()
        raise ServiceError(f"{service_name} response is not valid JSON: {preview}") from exc
    if not isinstance(data, dict):
        raise ServiceError(f"{service_name} response root must be a JSON object.")
    return data


def _extract_embedding(data: dict[str, Any], service_name: str) -> list[float]:
    try:
        raw_embedding = data["data"][0]["embedding"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ServiceError(f"{service_name} response is missing data[0].embedding.") from exc
    if not isinstance(raw_embedding, list) or not raw_embedding:
        raise ServiceError(f"{service_name} returned an empty embedding.")
    try:
        return [float(value) for value in raw_embedding]
    except (TypeError, ValueError) as exc:
        raise ServiceError(f"{service_name} embedding contains non-numeric values.") from exc


def _extract_error_detail(response: requests.Response | None) -> str:
    if response is None:
        return ""
    try:
        data = response.json()
    except ValueError:
        return response.text[:300].strip()
    if isinstance(data, dict):
        for key in ("message", "error", "detail"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
            if isinstance(value, dict):
                nested_message = value.get("message")
                if isinstance(nested_message, str) and nested_message.strip():
                    return nested_message.strip()
    return response.text[:300].strip()
