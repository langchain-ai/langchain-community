"""Unit tests for AnakinAPIWrapper."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from pydantic import SecretStr

from langchain_community.utilities.anakin import AnakinAPIWrapper

# ------------------------------------------------------------------
# API Key Handling
# ------------------------------------------------------------------


class TestAPIKeyHandling:
    def test_explicit_api_key(self) -> None:
        wrapper = AnakinAPIWrapper(anakin_api_key=SecretStr("ak-test-key-123"))
        assert wrapper.anakin_api_key.get_secret_value() == "ak-test-key-123"

    def test_env_var_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANAKIN_API_KEY", "ak-from-env")
        wrapper = AnakinAPIWrapper()
        assert wrapper.anakin_api_key.get_secret_value() == "ak-from-env"

    def test_missing_api_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ANAKIN_API_KEY", raising=False)
        with pytest.raises(ValueError, match="Anakin API key must be provided"):
            AnakinAPIWrapper()


# ------------------------------------------------------------------
# Header & URL Construction
# ------------------------------------------------------------------


class TestHeaders:
    def test_headers_contain_api_key(self) -> None:
        wrapper = AnakinAPIWrapper(anakin_api_key=SecretStr("ak-test"))
        headers = wrapper._headers()
        assert headers["X-API-Key"] == "ak-test"
        assert headers["Content-Type"] == "application/json"

    def test_integration_tag_in_headers(self) -> None:
        wrapper = AnakinAPIWrapper(anakin_api_key=SecretStr("ak-test"))
        headers = wrapper._headers()
        assert headers["X-Integration"] == "langchain"

    def test_url_construction(self) -> None:
        wrapper = AnakinAPIWrapper(anakin_api_key=SecretStr("ak-test"))
        assert wrapper._url("url-scraper") == "https://api.anakin.io/v1/url-scraper"

    def test_url_construction_custom_base(self) -> None:
        wrapper = AnakinAPIWrapper(
            anakin_api_key=SecretStr("ak-test"),
            api_base_url="https://custom.api.com/v2",
        )
        assert wrapper._url("search") == "https://custom.api.com/v2/search"


# ------------------------------------------------------------------
# Scrape
# ------------------------------------------------------------------


class TestScrape:
    @patch("langchain_community.utilities.anakin.requests.get")
    @patch("langchain_community.utilities.anakin.requests.post")
    def test_scrape_success(self, mock_post: MagicMock, mock_get: MagicMock) -> None:
        wrapper = AnakinAPIWrapper(anakin_api_key=SecretStr("ak-test"), poll_interval=0)

        # POST returns jobId
        mock_post.return_value = MagicMock(
            status_code=202,
            json=lambda: {"jobId": "job_123", "status": "pending"},
        )
        mock_post.return_value.raise_for_status = MagicMock()

        # GET returns completed result
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "id": "job_123",
                "status": "completed",
                "url": "https://example.com",
                "markdown": "# Example\n\nHello world",
            },
        )
        mock_get.return_value.raise_for_status = MagicMock()

        result = wrapper.scrape("https://example.com")

        assert result["status"] == "completed"
        assert result["markdown"] == "# Example\n\nHello world"

        # Verify POST payload
        call_kwargs = mock_post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert payload["url"] == "https://example.com"
        assert payload["country"] == "us"

    @patch("langchain_community.utilities.anakin.requests.get")
    @patch("langchain_community.utilities.anakin.requests.post")
    def test_scrape_with_browser(
        self, mock_post: MagicMock, mock_get: MagicMock
    ) -> None:
        wrapper = AnakinAPIWrapper(anakin_api_key=SecretStr("ak-test"), poll_interval=0)
        mock_post.return_value = MagicMock(
            json=lambda: {"jobId": "job_456"},
        )
        mock_post.return_value.raise_for_status = MagicMock()
        mock_get.return_value = MagicMock(
            json=lambda: {"status": "completed", "markdown": "content"},
        )
        mock_get.return_value.raise_for_status = MagicMock()

        wrapper.scrape("https://example.com", use_browser=True)

        call_kwargs = mock_post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert payload["useBrowser"] is True


# ------------------------------------------------------------------
# Batch Scrape
# ------------------------------------------------------------------


class TestBatchScrape:
    def test_batch_scrape_max_urls(self) -> None:
        wrapper = AnakinAPIWrapper(anakin_api_key=SecretStr("ak-test"))
        with pytest.raises(ValueError, match="maximum of 10 URLs"):
            wrapper.batch_scrape([f"https://example.com/{i}" for i in range(11)])

    @patch("langchain_community.utilities.anakin.requests.get")
    @patch("langchain_community.utilities.anakin.requests.post")
    def test_batch_scrape_success(
        self, mock_post: MagicMock, mock_get: MagicMock
    ) -> None:
        wrapper = AnakinAPIWrapper(anakin_api_key=SecretStr("ak-test"), poll_interval=0)
        mock_post.return_value = MagicMock(
            json=lambda: {"jobId": "batch_789"},
        )
        mock_post.return_value.raise_for_status = MagicMock()
        mock_get.return_value = MagicMock(
            json=lambda: {
                "status": "completed",
                "results": [
                    {"url": "https://a.com", "status": "completed", "markdown": "A"},
                    {"url": "https://b.com", "status": "completed", "markdown": "B"},
                ],
            },
        )
        mock_get.return_value.raise_for_status = MagicMock()

        result = wrapper.batch_scrape(["https://a.com", "https://b.com"])
        assert len(result["results"]) == 2


# ------------------------------------------------------------------
# Search
# ------------------------------------------------------------------


class TestSearch:
    @patch("langchain_community.utilities.anakin.requests.post")
    def test_search_success(self, mock_post: MagicMock) -> None:
        wrapper = AnakinAPIWrapper(anakin_api_key=SecretStr("ak-test"))
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "id": "search_123",
                "results": [
                    {
                        "url": "https://example.com/article",
                        "title": "AI News",
                        "snippet": "Latest developments...",
                        "date": "2026-01-15",
                    }
                ],
            },
        )
        mock_post.return_value.raise_for_status = MagicMock()

        results = wrapper.search("AI news")
        assert len(results) == 1
        assert results[0]["title"] == "AI News"
        assert results[0]["snippet"] == "Latest developments..."

        # Verify payload uses 'prompt' field
        call_kwargs = mock_post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert payload["prompt"] == "AI news"
        assert payload["limit"] == 5

    @patch("langchain_community.utilities.anakin.requests.post")
    def test_search_custom_limit(self, mock_post: MagicMock) -> None:
        wrapper = AnakinAPIWrapper(anakin_api_key=SecretStr("ak-test"))
        mock_post.return_value = MagicMock(
            json=lambda: {"results": []},
        )
        mock_post.return_value.raise_for_status = MagicMock()

        wrapper.search("query", limit=10)

        call_kwargs = mock_post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert payload["limit"] == 10


# ------------------------------------------------------------------
# Agentic Search
# ------------------------------------------------------------------


class TestAgenticSearch:
    @patch("langchain_community.utilities.anakin.requests.get")
    @patch("langchain_community.utilities.anakin.requests.post")
    def test_agentic_search_success(
        self, mock_post: MagicMock, mock_get: MagicMock
    ) -> None:
        wrapper = AnakinAPIWrapper(anakin_api_key=SecretStr("ak-test"), poll_interval=0)
        mock_post.return_value = MagicMock(
            json=lambda: {"job_id": "agent_abc", "status": "pending"},
        )
        mock_post.return_value.raise_for_status = MagicMock()
        mock_get.return_value = MagicMock(
            json=lambda: {
                "status": "completed",
                "generatedJson": {
                    "summary": "Research findings...",
                    "structured_data": {"items": []},
                },
                "durationMs": 33500,
            },
        )
        mock_get.return_value.raise_for_status = MagicMock()

        result = wrapper.agentic_search("compare React vs Vue")
        assert result["status"] == "completed"
        assert "summary" in result["generatedJson"]


# ------------------------------------------------------------------
# Poll Logic & Error Handling
# ------------------------------------------------------------------


class TestPollLogic:
    @patch("langchain_community.utilities.anakin.requests.get")
    @patch("langchain_community.utilities.anakin.requests.post")
    def test_poll_retries_on_pending(
        self, mock_post: MagicMock, mock_get: MagicMock
    ) -> None:
        wrapper = AnakinAPIWrapper(
            anakin_api_key=SecretStr("ak-test"), poll_interval=0, timeout=10
        )
        mock_post.return_value = MagicMock(
            json=lambda: {"jobId": "job_poll"},
        )
        mock_post.return_value.raise_for_status = MagicMock()

        # First call: pending, second call: completed
        mock_get.side_effect = [
            MagicMock(
                json=lambda: {"status": "pending"},
                raise_for_status=MagicMock(),
            ),
            MagicMock(
                json=lambda: {"status": "processing"},
                raise_for_status=MagicMock(),
            ),
            MagicMock(
                json=lambda: {"status": "completed", "markdown": "done"},
                raise_for_status=MagicMock(),
            ),
        ]

        result = wrapper.scrape("https://example.com")
        assert result["status"] == "completed"
        assert mock_get.call_count == 3

    @patch("langchain_community.utilities.anakin.requests.get")
    @patch("langchain_community.utilities.anakin.requests.post")
    def test_poll_raises_on_failure(
        self, mock_post: MagicMock, mock_get: MagicMock
    ) -> None:
        wrapper = AnakinAPIWrapper(
            anakin_api_key=SecretStr("ak-test"), poll_interval=0, timeout=10
        )
        mock_post.return_value = MagicMock(
            json=lambda: {"jobId": "job_fail"},
        )
        mock_post.return_value.raise_for_status = MagicMock()
        mock_get.return_value = MagicMock(
            json=lambda: {"status": "failed", "error": "Rate limit exceeded"},
            raise_for_status=MagicMock(),
        )

        with pytest.raises(RuntimeError, match="Rate limit exceeded"):
            wrapper.scrape("https://example.com")

    @patch("langchain_community.utilities.anakin.requests.post")
    def test_submit_no_job_id_raises(self, mock_post: MagicMock) -> None:
        wrapper = AnakinAPIWrapper(anakin_api_key=SecretStr("ak-test"))
        mock_post.return_value = MagicMock(
            json=lambda: {"status": "error"},
        )
        mock_post.return_value.raise_for_status = MagicMock()

        with pytest.raises(ValueError, match="No job ID"):
            wrapper.scrape("https://example.com")

    @patch("langchain_community.utilities.anakin.time.monotonic")
    @patch("langchain_community.utilities.anakin.requests.get")
    @patch("langchain_community.utilities.anakin.requests.post")
    def test_poll_timeout(
        self,
        mock_post: MagicMock,
        mock_get: MagicMock,
        mock_time: MagicMock,
    ) -> None:
        wrapper = AnakinAPIWrapper(
            anakin_api_key=SecretStr("ak-test"), poll_interval=0, timeout=5
        )
        mock_post.return_value = MagicMock(
            json=lambda: {"jobId": "job_timeout"},
        )
        mock_post.return_value.raise_for_status = MagicMock()

        # Simulate time passing beyond the deadline
        mock_time.side_effect = [0, 0, 6]  # start, check, expired
        mock_get.return_value = MagicMock(
            json=lambda: {"status": "pending"},
            raise_for_status=MagicMock(),
        )

        with pytest.raises(TimeoutError, match="did not complete"):
            wrapper.scrape("https://example.com")
