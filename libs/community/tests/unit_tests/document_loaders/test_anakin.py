"""Unit tests for AnakinLoader."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from langchain_community.document_loaders.anakin import AnakinLoader

# ------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------


class TestValidation:
    def test_invalid_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid mode"):
            AnakinLoader(
                url="https://example.com",
                mode="invalid",  # type: ignore[arg-type]
                api_key="ak-test",
            )

    def test_scrape_requires_url(self) -> None:
        with pytest.raises(ValueError, match="'url' is required"):
            AnakinLoader(mode="scrape", api_key="ak-test")

    def test_batch_scrape_requires_urls(self) -> None:
        with pytest.raises(ValueError, match="'urls' is required"):
            AnakinLoader(mode="batch_scrape", api_key="ak-test")

    def test_search_requires_query(self) -> None:
        with pytest.raises(ValueError, match="'query' is required"):
            AnakinLoader(mode="search", api_key="ak-test")

    def test_agentic_search_requires_query(self) -> None:
        with pytest.raises(ValueError, match="'query' is required"):
            AnakinLoader(mode="agentic_search", api_key="ak-test")


# ------------------------------------------------------------------
# Scrape Mode
# ------------------------------------------------------------------


class TestScrapeMode:
    @patch("langchain_community.utilities.anakin.requests.get")
    @patch("langchain_community.utilities.anakin.requests.post")
    def test_scrape_returns_document(
        self, mock_post: MagicMock, mock_get: MagicMock
    ) -> None:
        mock_post.return_value = MagicMock(
            json=lambda: {"jobId": "job_1"},
        )
        mock_post.return_value.raise_for_status = MagicMock()
        mock_get.return_value = MagicMock(
            json=lambda: {
                "status": "completed",
                "url": "https://example.com",
                "markdown": "# Example\n\nHello world",
                "durationMs": 5000,
            },
        )
        mock_get.return_value.raise_for_status = MagicMock()

        loader = AnakinLoader(
            url="https://example.com", mode="scrape", api_key="ak-test"
        )
        docs = loader.load()

        assert len(docs) == 1
        assert docs[0].page_content == "# Example\n\nHello world"
        assert docs[0].metadata["source"] == "https://example.com"
        assert docs[0].metadata["title"] == "Example"
        assert docs[0].metadata["status"] == "completed"
        assert docs[0].metadata["duration_ms"] == 5000

    @patch("langchain_community.utilities.anakin.requests.get")
    @patch("langchain_community.utilities.anakin.requests.post")
    def test_scrape_lazy_load_yields(
        self, mock_post: MagicMock, mock_get: MagicMock
    ) -> None:
        mock_post.return_value = MagicMock(
            json=lambda: {"jobId": "job_lazy"},
        )
        mock_post.return_value.raise_for_status = MagicMock()
        mock_get.return_value = MagicMock(
            json=lambda: {
                "status": "completed",
                "url": "https://example.com",
                "markdown": "Content here",
            },
        )
        mock_get.return_value.raise_for_status = MagicMock()

        loader = AnakinLoader(
            url="https://example.com", mode="scrape", api_key="ak-test"
        )
        docs = list(loader.lazy_load())
        assert len(docs) == 1
        assert docs[0].page_content == "Content here"


# ------------------------------------------------------------------
# Batch Scrape Mode
# ------------------------------------------------------------------


class TestBatchScrapeMode:
    @patch("langchain_community.utilities.anakin.requests.get")
    @patch("langchain_community.utilities.anakin.requests.post")
    def test_batch_scrape_returns_documents(
        self, mock_post: MagicMock, mock_get: MagicMock
    ) -> None:
        mock_post.return_value = MagicMock(
            json=lambda: {"jobId": "batch_1"},
        )
        mock_post.return_value.raise_for_status = MagicMock()
        mock_get.return_value = MagicMock(
            json=lambda: {
                "status": "completed",
                "results": [
                    {
                        "url": "https://a.com",
                        "status": "completed",
                        "markdown": "# Page A\n\nContent A",
                        "durationMs": 3000,
                    },
                    {
                        "url": "https://b.com",
                        "status": "completed",
                        "markdown": "# Page B\n\nContent B",
                        "durationMs": 4000,
                    },
                ],
            },
        )
        mock_get.return_value.raise_for_status = MagicMock()

        loader = AnakinLoader(
            urls=["https://a.com", "https://b.com"],
            mode="batch_scrape",
            api_key="ak-test",
        )
        docs = loader.load()

        assert len(docs) == 2
        assert docs[0].page_content == "# Page A\n\nContent A"
        assert docs[0].metadata["source"] == "https://a.com"
        assert docs[1].page_content == "# Page B\n\nContent B"
        assert docs[1].metadata["source"] == "https://b.com"

    @patch("langchain_community.utilities.anakin.requests.get")
    @patch("langchain_community.utilities.anakin.requests.post")
    def test_batch_scrape_skips_failed(
        self, mock_post: MagicMock, mock_get: MagicMock
    ) -> None:
        mock_post.return_value = MagicMock(
            json=lambda: {"jobId": "batch_fail"},
        )
        mock_post.return_value.raise_for_status = MagicMock()
        mock_get.return_value = MagicMock(
            json=lambda: {
                "status": "completed",
                "results": [
                    {
                        "url": "https://a.com",
                        "status": "completed",
                        "markdown": "Content A",
                    },
                    {
                        "url": "https://bad.com",
                        "status": "failed",
                        "error": "Timeout",
                    },
                ],
            },
        )
        mock_get.return_value.raise_for_status = MagicMock()

        loader = AnakinLoader(
            urls=["https://a.com", "https://bad.com"],
            mode="batch_scrape",
            api_key="ak-test",
        )
        docs = loader.load()

        assert len(docs) == 1
        assert docs[0].metadata["source"] == "https://a.com"


# ------------------------------------------------------------------
# Search Mode
# ------------------------------------------------------------------


class TestSearchMode:
    @patch("langchain_community.utilities.anakin.requests.post")
    def test_search_returns_documents(self, mock_post: MagicMock) -> None:
        mock_post.return_value = MagicMock(
            json=lambda: {
                "id": "search_1",
                "results": [
                    {
                        "url": "https://article1.com",
                        "title": "Article One",
                        "snippet": "First result snippet",
                        "date": "2026-01-15",
                        "last_updated": "2026-01-20",
                    },
                    {
                        "url": "https://article2.com",
                        "title": "Article Two",
                        "snippet": "Second result snippet",
                        "date": "2026-01-10",
                    },
                ],
            },
        )
        mock_post.return_value.raise_for_status = MagicMock()

        loader = AnakinLoader(query="AI news", mode="search", api_key="ak-test")
        docs = loader.load()

        assert len(docs) == 2
        assert docs[0].page_content == "First result snippet"
        assert docs[0].metadata["source"] == "https://article1.com"
        assert docs[0].metadata["title"] == "Article One"
        assert docs[0].metadata["date"] == "2026-01-15"
        assert docs[1].page_content == "Second result snippet"

    @patch("langchain_community.utilities.anakin.requests.post")
    def test_search_empty_results(self, mock_post: MagicMock) -> None:
        mock_post.return_value = MagicMock(
            json=lambda: {"id": "search_empty", "results": []},
        )
        mock_post.return_value.raise_for_status = MagicMock()

        loader = AnakinLoader(
            query="nonexistent topic", mode="search", api_key="ak-test"
        )
        docs = loader.load()
        assert len(docs) == 0


# ------------------------------------------------------------------
# Agentic Search Mode
# ------------------------------------------------------------------


class TestAgenticSearchMode:
    @patch("langchain_community.utilities.anakin.requests.get")
    @patch("langchain_community.utilities.anakin.requests.post")
    def test_agentic_search_returns_document(
        self, mock_post: MagicMock, mock_get: MagicMock
    ) -> None:
        mock_post.return_value = MagicMock(
            json=lambda: {"job_id": "agent_1", "status": "pending"},
        )
        mock_post.return_value.raise_for_status = MagicMock()
        mock_get.return_value = MagicMock(
            json=lambda: {
                "status": "completed",
                "generatedJson": {
                    "summary": "Comprehensive research report on AI trends...",
                    "structured_data": {
                        "developments": [{"title": "GPT-5 Release", "date": "2026-01"}]
                    },
                },
                "durationMs": 45000,
            },
        )
        mock_get.return_value.raise_for_status = MagicMock()

        loader = AnakinLoader(
            query="AI trends 2026", mode="agentic_search", api_key="ak-test"
        )
        docs = loader.load()

        assert len(docs) == 1
        assert "research report" in docs[0].page_content
        assert docs[0].metadata["source"] == "anakin_agentic_search"
        assert docs[0].metadata["query"] == "AI trends 2026"
        assert docs[0].metadata["duration_ms"] == 45000
        assert "developments" in docs[0].metadata["structured_data"]


# ------------------------------------------------------------------
# Title Extraction
# ------------------------------------------------------------------


class TestTitleExtraction:
    @patch("langchain_community.utilities.anakin.requests.get")
    @patch("langchain_community.utilities.anakin.requests.post")
    def test_title_from_h1(self, mock_post: MagicMock, mock_get: MagicMock) -> None:
        mock_post.return_value = MagicMock(
            json=lambda: {"jobId": "job_title"},
        )
        mock_post.return_value.raise_for_status = MagicMock()
        mock_get.return_value = MagicMock(
            json=lambda: {
                "status": "completed",
                "url": "https://example.com",
                "markdown": "# My Great Page\n\nSome content here.",
            },
        )
        mock_get.return_value.raise_for_status = MagicMock()

        loader = AnakinLoader(
            url="https://example.com", mode="scrape", api_key="ak-test"
        )
        docs = loader.load()
        assert docs[0].metadata["title"] == "My Great Page"

    @patch("langchain_community.utilities.anakin.requests.get")
    @patch("langchain_community.utilities.anakin.requests.post")
    def test_title_empty_when_no_heading(
        self, mock_post: MagicMock, mock_get: MagicMock
    ) -> None:
        mock_post.return_value = MagicMock(
            json=lambda: {"jobId": "job_no_title"},
        )
        mock_post.return_value.raise_for_status = MagicMock()
        mock_get.return_value = MagicMock(
            json=lambda: {
                "status": "completed",
                "url": "https://example.com",
                "markdown": "Just some plain text with no heading.",
            },
        )
        mock_get.return_value.raise_for_status = MagicMock()

        loader = AnakinLoader(
            url="https://example.com", mode="scrape", api_key="ak-test"
        )
        docs = loader.load()
        assert docs[0].metadata["title"] == ""
