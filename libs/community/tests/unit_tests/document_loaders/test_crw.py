"""Test CrwLoader."""

from typing import Iterator
from unittest.mock import MagicMock, patch

import pytest

from langchain_community.document_loaders import CrwLoader


@pytest.fixture()
def mock_session() -> Iterator[MagicMock]:
    """Mock requests.Session for all tests."""
    with patch("langchain_community.document_loaders.crw.requests.Session") as cls:
        yield cls.return_value


class TestCrwLoader:
    """Test CrwLoader."""

    def test_scrape_mode(self, mock_session: MagicMock) -> None:
        """Test loading in scrape mode."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "success": True,
            "data": {
                "markdown": "# Example\n\nHello world",
                "metadata": {
                    "title": "Example",
                    "sourceURL": "https://example.com",
                },
            },
        }
        mock_response.raise_for_status = MagicMock()
        mock_session.request.return_value = mock_response

        loader = CrwLoader(url="https://example.com", mode="scrape")
        docs = list(loader.lazy_load())

        assert len(docs) == 1
        assert docs[0].page_content == "# Example\n\nHello world"
        assert docs[0].metadata["title"] == "Example"

        mock_session.request.assert_called_once()
        call_args = mock_session.request.call_args
        assert call_args[0] == ("POST", "http://localhost:3000/v1/scrape")

    def test_crawl_mode(self, mock_session: MagicMock) -> None:
        """Test loading in crawl mode with polling."""
        start_response = MagicMock()
        start_response.json.return_value = {"success": True, "id": "job-123"}
        start_response.raise_for_status = MagicMock()

        status_response = MagicMock()
        status_response.json.return_value = {
            "status": "completed",
            "data": [
                {
                    "markdown": "Page 1 content",
                    "metadata": {"sourceURL": "https://example.com/1"},
                },
                {
                    "markdown": "Page 2 content",
                    "metadata": {"sourceURL": "https://example.com/2"},
                },
            ],
        }
        status_response.raise_for_status = MagicMock()

        mock_session.request.side_effect = [start_response, status_response]

        loader = CrwLoader(url="https://example.com", mode="crawl")
        docs = list(loader.lazy_load())

        assert len(docs) == 2
        assert docs[0].page_content == "Page 1 content"
        assert docs[1].page_content == "Page 2 content"

    def test_map_mode(self, mock_session: MagicMock) -> None:
        """Test loading in map mode."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "success": True,
            "links": [
                "https://example.com/a",
                "https://example.com/b",
                "https://example.com/c",
            ],
        }
        mock_response.raise_for_status = MagicMock()
        mock_session.request.return_value = mock_response

        loader = CrwLoader(url="https://example.com", mode="map")
        docs = list(loader.lazy_load())

        assert len(docs) == 3
        assert docs[0].page_content == "https://example.com/a"

    def test_invalid_mode(self) -> None:
        """Test that invalid mode raises ValueError."""
        with pytest.raises(ValueError, match="Invalid mode"):
            CrwLoader(url="https://example.com", mode="invalid")  # type: ignore[arg-type]

    def test_api_key_from_env(
        self, mock_session: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that API key is read from environment variable."""
        monkeypatch.setenv("CRW_API_KEY", "test-key-123")
        loader = CrwLoader(url="https://example.com")
        assert loader.api_key == "test-key-123"

    def test_api_url_from_env(
        self, mock_session: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that API URL is read from environment variable."""
        monkeypatch.setenv("CRW_API_URL", "https://custom.crw.dev")
        loader = CrwLoader(url="https://example.com")
        assert loader.api_url == "https://custom.crw.dev"

    def test_custom_api_url(self, mock_session: MagicMock) -> None:
        """Test that custom API URL overrides default."""
        loader = CrwLoader(
            url="https://example.com",
            api_url="https://fastcrw.com/api",
        )
        assert loader.api_url == "https://fastcrw.com/api"

    def test_params_forwarded(self, mock_session: MagicMock) -> None:
        """Test that params are forwarded to API as camelCase."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "success": True,
            "data": {
                "markdown": "content",
                "metadata": {},
            },
        }
        mock_response.raise_for_status = MagicMock()
        mock_session.request.return_value = mock_response

        loader = CrwLoader(
            url="https://example.com",
            mode="scrape",
            params={"only_main_content": True, "render_js": True},
        )
        list(loader.lazy_load())

        call_kwargs = mock_session.request.call_args
        body = call_kwargs[1]["json"]
        assert body["onlyMainContent"] is True
        assert body["renderJs"] is True

    def test_scrape_empty_data(self, mock_session: MagicMock) -> None:
        """Test that scrape mode handles empty data gracefully."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"success": True, "data": {}}
        mock_response.raise_for_status = MagicMock()
        mock_session.request.return_value = mock_response

        loader = CrwLoader(url="https://example.com", mode="scrape")
        docs = list(loader.lazy_load())

        assert len(docs) == 0

    def test_crawl_missing_job_id(self, mock_session: MagicMock) -> None:
        """Test that crawl raises when job ID is missing from response."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"success": True}
        mock_response.raise_for_status = MagicMock()
        mock_session.request.return_value = mock_response

        loader = CrwLoader(url="https://example.com", mode="crawl")
        with pytest.raises(ValueError, match="did not return a job ID"):
            list(loader.lazy_load())

    def test_crawl_failed_status(self, mock_session: MagicMock) -> None:
        """Test that crawl raises on failed job status."""
        start_response = MagicMock()
        start_response.json.return_value = {"success": True, "id": "job-456"}
        start_response.raise_for_status = MagicMock()

        status_response = MagicMock()
        status_response.json.return_value = {
            "status": "failed",
            "error": "Site unreachable",
        }
        status_response.raise_for_status = MagicMock()

        mock_session.request.side_effect = [start_response, status_response]

        loader = CrwLoader(url="https://example.com", mode="crawl")
        with pytest.raises(RuntimeError, match="crawl job 'job-456' failed"):
            list(loader.lazy_load())

    def test_crawl_timeout(self, mock_session: MagicMock) -> None:
        """Test that crawl raises TimeoutError when polling exceeds timeout."""
        start_response = MagicMock()
        start_response.json.return_value = {"success": True, "id": "job-789"}
        start_response.raise_for_status = MagicMock()

        pending_response = MagicMock()
        pending_response.json.return_value = {"status": "scraping"}
        pending_response.raise_for_status = MagicMock()

        mock_session.request.side_effect = [start_response] + [pending_response] * 10

        loader = CrwLoader(
            url="https://example.com",
            mode="crawl",
            params={"poll_interval": 1, "timeout": 2},
        )
        with pytest.raises(TimeoutError, match="timed out after 2s"):
            list(loader.lazy_load())

    def test_crawl_poll_interval_zero(self, mock_session: MagicMock) -> None:
        """Test that poll_interval=0 raises ValueError."""
        start_response = MagicMock()
        start_response.json.return_value = {"success": True, "id": "job-0"}
        start_response.raise_for_status = MagicMock()
        mock_session.request.return_value = start_response

        loader = CrwLoader(
            url="https://example.com",
            mode="crawl",
            params={"poll_interval": 0},
        )
        with pytest.raises(ValueError, match="poll_interval must be > 0"):
            list(loader.lazy_load())

    def test_crawl_timeout_zero(self, mock_session: MagicMock) -> None:
        """Test that timeout=0 raises ValueError."""
        start_response = MagicMock()
        start_response.json.return_value = {"success": True, "id": "job-t0"}
        start_response.raise_for_status = MagicMock()
        mock_session.request.return_value = start_response

        loader = CrwLoader(
            url="https://example.com",
            mode="crawl",
            params={"timeout": 0},
        )
        with pytest.raises(ValueError, match="timeout must be > 0"):
            list(loader.lazy_load())

    def test_api_success_false(self, mock_session: MagicMock) -> None:
        """Test that API response with success=false raises RuntimeError."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "success": False,
            "error": "Rate limit exceeded",
        }
        mock_response.raise_for_status = MagicMock()
        mock_session.request.return_value = mock_response

        loader = CrwLoader(url="https://example.com", mode="scrape")
        with pytest.raises(RuntimeError, match="Rate limit exceeded"):
            list(loader.lazy_load())
