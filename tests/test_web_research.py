from __future__ import annotations

import requests

from agent_for_mc.application.skills.web_research import (
    ZHIPU_SEARCH_QUERY_MAX_CHARS,
    ZHIPU_SEARCH_RESULT_MAX_COUNT,
    WebResearchResponse,
    ZhipuWebResearchProvider,
    normalize_zhipu_search_query,
)


class FakeResponse:
    def __init__(
        self,
        payload,
        *,
        status_error: requests.HTTPError | None = None,
        text: str = "",
    ):
        self._payload = payload
        self._status_error = status_error
        self.text = text

    def raise_for_status(self):
        if self._status_error is not None:
            raise self._status_error

    def json(self):
        if isinstance(self._payload, BaseException):
            raise self._payload
        return self._payload


def test_zhipu_web_research_posts_expected_payload(monkeypatch):
    captured: dict[str, object] = {}

    def fake_post(url, *, headers, json, timeout):
        captured.update(
            {
                "url": url,
                "headers": headers,
                "json": json,
                "timeout": timeout,
            }
        )
        return FakeResponse(
            {
                "search_result": [
                    {
                        "title": "LuckPerms verbose",
                        "link": "https://example.com/luckperms",
                        "content": "Use verbose mode to debug permissions.",
                        "media": "Example Docs",
                        "publish_date": "2026-01-01",
                        "icon": "https://example.com/icon.png",
                    }
                ]
            }
        )

    monkeypatch.setattr(requests, "post", fake_post)
    provider = ZhipuWebResearchProvider(
        api_key="zhipu-key",
        url="https://search.example.com/web_search",
        search_engine="search_pro",
        content_size="high",
        search_intent=True,
        search_recency_filter="oneYear",
        search_domain_filter="example.com",
        timeout_seconds=9,
    )

    response = provider.search("LuckPerms verbose permission debugging", top_k=3)

    assert captured["url"] == "https://search.example.com/web_search"
    assert captured["headers"] == {
        "Authorization": "Bearer zhipu-key",
        "Content-Type": "application/json",
    }
    assert captured["json"] == {
        "search_query": "LuckPerms verbose permission debugging",
        "search_engine": "search_pro",
        "count": 3,
        "content_size": "high",
        "search_intent": True,
        "search_recency_filter": "oneYear",
        "search_domain_filter": "example.com",
    }
    assert captured["timeout"] == 9
    assert response.diagnostics == ()
    assert len(response.results) == 1
    assert response.results[0].title == "LuckPerms verbose"
    assert response.results[0].url == "https://example.com/luckperms"
    assert response.results[0].summary == "Use verbose mode to debug permissions."
    assert response.results[0].site_name == "Example Docs"
    assert response.results[0].published_time == "2026-01-01"
    assert response.results[0].raw_rank == 1


def test_zhipu_web_research_truncates_query(monkeypatch):
    captured: dict[str, object] = {}

    def fake_post(url, *, headers, json, timeout):
        captured["query"] = json["search_query"]
        return FakeResponse({"search_result": []})

    monkeypatch.setattr(requests, "post", fake_post)
    provider = ZhipuWebResearchProvider(api_key="zhipu-key")
    long_query = "a" * (ZHIPU_SEARCH_QUERY_MAX_CHARS + 20)

    provider.search(long_query, top_k=2)

    assert captured["query"] == "a" * ZHIPU_SEARCH_QUERY_MAX_CHARS
    assert normalize_zhipu_search_query(long_query) == captured["query"]


def test_zhipu_web_research_clamps_result_count(monkeypatch):
    captured: dict[str, object] = {}

    def fake_post(url, *, headers, json, timeout):
        captured["count"] = json["count"]
        return FakeResponse({"search_result": []})

    monkeypatch.setattr(requests, "post", fake_post)
    provider = ZhipuWebResearchProvider(api_key="zhipu-key")

    provider.search("LuckPerms", top_k=ZHIPU_SEARCH_RESULT_MAX_COUNT + 50)

    assert captured["count"] == ZHIPU_SEARCH_RESULT_MAX_COUNT


def test_zhipu_web_research_reports_response_failures(monkeypatch):
    def fake_post(url, *, headers, json, timeout):
        return FakeResponse(ValueError("bad json"))

    monkeypatch.setattr(requests, "post", fake_post)
    response = ZhipuWebResearchProvider(api_key="zhipu-key").search("LuckPerms")

    assert response.results == ()
    assert response.diagnostics == ("web research response is not valid JSON",)


def test_zhipu_web_research_reports_http_failures(monkeypatch):
    def fake_post(url, *, headers, json, timeout):
        error_response = FakeResponse({"error": {"message": "quota exceeded"}})
        raise requests.HTTPError("429", response=error_response)

    monkeypatch.setattr(requests, "post", fake_post)
    response = ZhipuWebResearchProvider(api_key="zhipu-key").search("LuckPerms")

    assert response.results == ()
    assert response.diagnostics == ("web research request failed: quota exceeded",)


def test_zhipu_web_research_reports_missing_fields(monkeypatch):
    def fake_post(url, *, headers, json, timeout):
        return FakeResponse(
            {
                "search_result": [
                    {
                        "title": "Incomplete",
                        "link": "https://example.com/incomplete",
                    }
                ]
            }
        )

    monkeypatch.setattr(requests, "post", fake_post)
    response = ZhipuWebResearchProvider(api_key="zhipu-key").search("LuckPerms")

    assert isinstance(response, WebResearchResponse)
    assert response.results == ()
    assert response.diagnostics == (
        "web research result 1 is missing title, url, or summary",
    )
