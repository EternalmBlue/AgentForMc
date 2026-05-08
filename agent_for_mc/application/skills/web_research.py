from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import requests


DEFAULT_ZHIPU_WEB_SEARCH_URL = "https://open.bigmodel.cn/api/paas/v4/web_search"
ZHIPU_SEARCH_QUERY_MAX_CHARS = 70
ZHIPU_SEARCH_RESULT_MAX_COUNT = 10


class WebResearchProvider(Protocol):
    def search(self, search_query: str, *, top_k: int = 5) -> "WebResearchResponse":
        ...


@dataclass(frozen=True, slots=True)
class WebResearchResult:
    title: str
    url: str
    summary: str
    site_name: str = ""
    published_time: str = ""
    icon: str = ""
    raw_rank: int = 0


@dataclass(frozen=True, slots=True)
class WebResearchResponse:
    results: tuple[WebResearchResult, ...] = ()
    diagnostics: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ZhipuWebResearchProvider:
    api_key: str
    url: str = DEFAULT_ZHIPU_WEB_SEARCH_URL
    search_engine: str = "search_std"
    content_size: str = "medium"
    search_intent: bool = False
    search_recency_filter: str = "noLimit"
    search_domain_filter: str = ""
    timeout_seconds: int = 30

    def search(self, search_query: str, *, top_k: int = 5) -> WebResearchResponse:
        query = normalize_zhipu_search_query(search_query)
        if not query:
            return WebResearchResponse(
                diagnostics=("web research query is blank",),
            )
        if not str(self.api_key or "").strip():
            return WebResearchResponse(
                diagnostics=("web research API key is missing",),
            )

        try:
            response = requests.post(
                self.url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=self._payload(query, top_k=top_k),
                timeout=max(1, int(self.timeout_seconds)),
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            return WebResearchResponse(
                diagnostics=(f"web research request failed: {_request_error_detail(exc)}",),
            )

        try:
            data = response.json()
        except ValueError:
            return WebResearchResponse(
                diagnostics=("web research response is not valid JSON",),
            )
        if not isinstance(data, dict):
            return WebResearchResponse(
                diagnostics=("web research response root must be a JSON object",),
            )

        return _parse_zhipu_response(data)

    def _payload(self, query: str, *, top_k: int) -> dict[str, object]:
        payload: dict[str, object] = {
            "search_query": query,
            "search_engine": self.search_engine,
            "count": min(ZHIPU_SEARCH_RESULT_MAX_COUNT, max(1, int(top_k))),
            "content_size": self.content_size,
            "search_intent": bool(self.search_intent),
            "search_recency_filter": self.search_recency_filter,
        }
        domain_filter = str(self.search_domain_filter or "").strip()
        if domain_filter:
            payload["search_domain_filter"] = domain_filter
        return payload


def normalize_zhipu_search_query(search_query: str) -> str:
    query = " ".join(str(search_query or "").strip().split())
    return query[:ZHIPU_SEARCH_QUERY_MAX_CHARS]


def _parse_zhipu_response(data: dict) -> WebResearchResponse:
    raw_results = _extract_result_list(data)
    if raw_results is None:
        return WebResearchResponse(
            diagnostics=("web research response is missing search_result list",),
        )
    if not raw_results:
        return WebResearchResponse(
            diagnostics=("web research returned no matching results",),
        )

    results: list[WebResearchResult] = []
    diagnostics: list[str] = []
    for index, item in enumerate(raw_results, start=1):
        if not isinstance(item, dict):
            diagnostics.append(f"web research result {index} is not an object")
            continue

        title = _first_str(item, "title", "name")
        url = _first_str(item, "link", "url")
        summary = _first_str(item, "content", "summary", "snippet", "description")
        if not title or not url or not summary:
            diagnostics.append(
                f"web research result {index} is missing title, url, or summary"
            )
            continue

        results.append(
            WebResearchResult(
                title=title,
                url=url,
                summary=summary,
                site_name=_first_str(item, "media", "site_name", "source", "refer"),
                published_time=_first_str(item, "publish_date", "published_time", "date"),
                icon=_first_str(item, "icon"),
                raw_rank=index,
            )
        )

    if not results and not diagnostics:
        diagnostics.append("web research returned no usable results")
    return WebResearchResponse(
        results=tuple(results),
        diagnostics=tuple(diagnostics),
    )


def _extract_result_list(data: dict) -> list | None:
    candidates = [
        data.get("search_result"),
        data.get("results"),
    ]
    nested = data.get("data")
    if isinstance(nested, dict):
        candidates.extend(
            [
                nested.get("search_result"),
                nested.get("results"),
            ]
        )
    for candidate in candidates:
        if isinstance(candidate, list):
            return candidate
    return None


def _first_str(data: dict, *keys: str) -> str:
    for key in keys:
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return " ".join(value.strip().split())
    return ""


def _request_error_detail(exc: requests.RequestException) -> str:
    response = exc.response
    if response is None:
        return str(exc)
    try:
        data = response.json()
    except ValueError:
        return response.text[:300].strip() or str(exc)
    if isinstance(data, dict):
        for key in ("message", "error", "detail"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
            if isinstance(value, dict):
                nested = value.get("message")
                if isinstance(nested, str) and nested.strip():
                    return nested.strip()
    return response.text[:300].strip() or str(exc)
