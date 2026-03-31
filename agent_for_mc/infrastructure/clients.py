from __future__ import annotations

from typing import Any

import requests

from agent_for_mc.domain.errors import ConfigurationError, ServiceError
from agent_for_mc.infrastructure.config import Settings


class JinaEmbeddingClient:
    def __init__(self, settings: Settings):
        self._settings = settings

    def embed_query(self, text: str) -> list[float]:
        if not self._settings.jina_api_key:
            raise ConfigurationError("缺少环境变量 RAG_JINA_API_KEY。")

        payload = {
            "model": self._settings.jina_embeddings_model,
            "task": self._settings.jina_embeddings_task,
            "normalized": True,
            "input": text,
        }
        headers = {
            "Authorization": f"Bearer {self._settings.jina_api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = requests.post(
                self._settings.jina_embeddings_url,
                headers=headers,
                json=payload,
                timeout=self._settings.request_timeout_seconds,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise ServiceError(f"Jina embedding 请求失败: {exc}") from exc

        data = _read_json(response, "Jina embedding")
        try:
            return data["data"][0]["embedding"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ServiceError("Jina embedding 响应缺少 embedding 字段。") from exc


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
        if not self._settings.deepseek_api_key:
            raise ConfigurationError("缺少环境变量 RAG_DEEPSEEK_API_KEY。")

        payload = {
            "model": self._model_name,
            "messages": messages,
            "temperature": temperature,
        }
        headers = {
            "Authorization": f"Bearer {self._settings.deepseek_api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = requests.post(
                self._settings.deepseek_chat_url,
                headers=headers,
                json=payload,
                timeout=self._settings.request_timeout_seconds,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise ServiceError(f"DeepSeek 请求失败: {exc}") from exc

        data = _read_json(response, "DeepSeek")
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ServiceError("DeepSeek 响应缺少 message.content 字段。") from exc

        normalized = str(content).strip()
        if not normalized:
            raise ServiceError("DeepSeek 返回了空内容。")
        return normalized


def _read_json(response: requests.Response, service_name: str) -> dict[str, Any]:
    try:
        return response.json()
    except ValueError as exc:
        preview = response.text[:200].strip()
        raise ServiceError(f"{service_name} 响应不是合法 JSON: {preview}") from exc
