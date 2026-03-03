"""Tests for AlterLabLoader document loader."""

import asyncio
import sys
from typing import Any, Dict, Generator, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document


# Mock alterlab module before importing AlterLabLoader
@pytest.fixture(autouse=True)
def mock_alterlab_module() -> (
    Generator[Tuple[MagicMock, MagicMock, MagicMock], None, None]
):
    """Mock alterlab package in sys.modules for all tests."""
    mock_module = MagicMock()
    mock_sync_client = MagicMock()
    mock_async_client = AsyncMock()

    mock_module.AlterLabSync.return_value = mock_sync_client
    mock_module.AlterLab.return_value = mock_async_client

    sys.modules["alterlab"] = mock_module
    yield mock_module, mock_sync_client, mock_async_client

    if "alterlab" in sys.modules:
        del sys.modules["alterlab"]


from langchain_community.document_loaders.alterlab import AlterLabLoader  # noqa: E402

_LOADER_INIT = (
    "langchain_community.document_loaders"
    ".alterlab.AlterLabLoader.__init__"
)

# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

SAMPLE_SCRAPE_RESPONSE: Dict[str, Any] = {
    "job_id": "job_abc123",
    "url": "https://example.com/article",
    "status_code": 200,
    "content": {
        "markdown": "# Example Article\n\nThis is the article body.",
        "text": "Example Article. This is the article body.",
        "html": "<h1>Example Article</h1><p>This is the article body.</p>",
        "json": {
            "title": "Example Article",
            "body": "This is the article body.",
        },
    },
    "title": "Example Article",
    "author": "Jane Doe",
    "published_at": "2026-01-15T10:00:00Z",
    "metadata": {"language": "en", "content_type": "article"},
    "headers": {"content-type": "text/html"},
    "cached": False,
    "response_time_ms": 450,
    "size_bytes": 2048,
    "billing": {
        "total_credits": 2,
        "tier_used": "2",
        "escalations": [],
        "savings": 0,
    },
    "extraction_method": "algorithmic",
    "version": "v1",
}

SAMPLE_LEGACY_RESPONSE: Dict[str, Any] = {
    "url": "https://example.com/page",
    "status_code": 200,
    "content": "This is plain text content from the page.",
    "title": "Plain Page",
    "author": None,
    "published_at": None,
    "metadata": None,
    "headers": {"content-type": "text/html"},
    "cached": True,
    "credits_used": 1,
    "response_time_ms": 120,
    "size_bytes": 512,
}

SAMPLE_EMPTY_RESPONSE: Dict[str, Any] = {
    "url": "https://example.com/empty",
    "status_code": 200,
    "content": {},
    "title": None,
    "metadata": None,
    "headers": {},
    "cached": False,
    "response_time_ms": 50,
    "size_bytes": 0,
    "billing": {"total_credits": 1, "tier_used": "1", "escalations": [], "savings": 0},
}


@pytest.fixture
def mock_alterlab_sync():
    """Create a mock AlterLabSync client."""
    mock_client = MagicMock()
    mock_client.scrape.return_value = SAMPLE_SCRAPE_RESPONSE
    mock_client.close = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    return mock_client


@pytest.fixture
def mock_alterlab_async():
    """Create a mock AlterLab async client."""
    mock_client = AsyncMock()
    mock_client.scrape.return_value = SAMPLE_SCRAPE_RESPONSE
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    return mock_client


# ---------------------------------------------------------------------------
# Constructor tests
# ---------------------------------------------------------------------------


class TestAlterLabLoaderInit:
    def test_init_with_url(self):
        with patch.dict("os.environ", {"ALTERLAB_API_KEY": "sk_test_abc"}):
            loader = AlterLabLoader(url="https://example.com")
            assert loader.urls == ["https://example.com"]
            assert loader.api_key == "sk_test_abc"
            assert loader.api_url == "https://api.alterlab.io"
            assert loader.params == {}

    def test_init_with_urls(self):
        with patch.dict("os.environ", {"ALTERLAB_API_KEY": "sk_test_abc"}):
            urls = ["https://example.com/1", "https://example.com/2"]
            loader = AlterLabLoader(urls=urls)
            assert loader.urls == urls

    def test_init_with_explicit_api_key(self):
        loader = AlterLabLoader(url="https://example.com", api_key="sk_live_xyz")
        assert loader.api_key == "sk_live_xyz"

    def test_init_with_custom_api_url(self):
        loader = AlterLabLoader(
            url="https://example.com",
            api_key="sk_test_abc",
            api_url="https://custom.api.com",
        )
        assert loader.api_url == "https://custom.api.com"

    def test_init_with_params(self):
        loader = AlterLabLoader(
            url="https://example.com",
            api_key="sk_test_abc",
            params={"formats": ["markdown", "json"], "mode": "js"},
        )
        assert loader.params == {"formats": ["markdown", "json"], "mode": "js"}

    def test_init_requires_url_or_urls(self):
        with pytest.raises(ValueError, match="Either 'url' or 'urls' must be provided"):
            AlterLabLoader(api_key="sk_test_abc")

    def test_init_rejects_both_url_and_urls(self):
        with pytest.raises(
            ValueError, match="Provide either 'url' or 'urls', not both"
        ):
            AlterLabLoader(
                url="https://example.com",
                urls=["https://example.com"],
                api_key="sk_test_abc",
            )

    def test_init_requires_api_key(self):
        with patch.dict("os.environ", {}, clear=True):
            # Also clear ALTERLAB_API_KEY if it exists
            import os

            os.environ.pop("ALTERLAB_API_KEY", None)
            with pytest.raises(ValueError, match="API key is required"):
                AlterLabLoader(url="https://example.com")

    def test_init_api_url_from_env(self):
        with patch.dict(
            "os.environ",
            {
                "ALTERLAB_API_KEY": "sk_test_abc",
                "ALTERLAB_API_URL": "https://staging.alterlab.io",
            },
        ):
            loader = AlterLabLoader(url="https://example.com")
            assert loader.api_url == "https://staging.alterlab.io"


# ---------------------------------------------------------------------------
# Content extraction tests
# ---------------------------------------------------------------------------


class TestContentExtraction:
    def test_extract_markdown_priority(self):
        content = AlterLabLoader._extract_page_content(SAMPLE_SCRAPE_RESPONSE)
        assert content == "# Example Article\n\nThis is the article body."

    def test_extract_text_fallback(self):
        result = {
            "content": {
                "text": "Fallback text",
                "html": "<p>HTML</p>",
            }
        }
        content = AlterLabLoader._extract_page_content(result)
        assert content == "Fallback text"

    def test_extract_html_fallback(self):
        result = {"content": {"html": "<p>Only HTML</p>"}}
        content = AlterLabLoader._extract_page_content(result)
        assert content == "<p>Only HTML</p>"

    def test_extract_legacy_string(self):
        content = AlterLabLoader._extract_page_content(SAMPLE_LEGACY_RESPONSE)
        assert content == "This is plain text content from the page."

    def test_extract_empty_content(self):
        content = AlterLabLoader._extract_page_content(SAMPLE_EMPTY_RESPONSE)
        assert content == ""


# ---------------------------------------------------------------------------
# Document conversion tests
# ---------------------------------------------------------------------------


class TestResultToDocument:
    def test_full_metadata(self):
        doc = AlterLabLoader._result_to_document(
            SAMPLE_SCRAPE_RESPONSE, "https://example.com/article"
        )
        assert doc is not None
        assert doc.page_content == "# Example Article\n\nThis is the article body."
        assert doc.metadata["source"] == "https://example.com/article"
        assert doc.metadata["title"] == "Example Article"
        assert doc.metadata["author"] == "Jane Doe"
        assert doc.metadata["published_at"] == "2026-01-15T10:00:00Z"
        assert doc.metadata["status_code"] == 200
        assert doc.metadata["extraction_method"] == "algorithmic"
        assert doc.metadata["response_time_ms"] == 450
        assert doc.metadata["credits_used"] == 2
        assert doc.metadata["tier_used"] == "2"
        # Extra metadata merged
        assert doc.metadata["language"] == "en"
        assert doc.metadata["content_type"] == "article"

    def test_legacy_response_metadata(self):
        doc = AlterLabLoader._result_to_document(
            SAMPLE_LEGACY_RESPONSE, "https://example.com/page"
        )
        assert doc is not None
        assert doc.metadata["source"] == "https://example.com/page"
        assert doc.metadata["title"] == "Plain Page"
        assert doc.metadata["credits_used"] == 1
        assert "author" not in doc.metadata
        assert "tier_used" not in doc.metadata

    def test_empty_content_returns_none(self):
        doc = AlterLabLoader._result_to_document(
            SAMPLE_EMPTY_RESPONSE, "https://example.com/empty"
        )
        assert doc is None

    def test_source_url_fallback(self):
        result = {
            "content": "Some content",
            "status_code": 200,
            "headers": {},
            "cached": False,
            "response_time_ms": 100,
            "size_bytes": 100,
        }
        doc = AlterLabLoader._result_to_document(result, "https://fallback.com")
        assert doc is not None
        assert doc.metadata["source"] == "https://fallback.com"


# ---------------------------------------------------------------------------
# Sync load tests
# ---------------------------------------------------------------------------


class TestLazyLoad:
    def test_single_url(self, mock_alterlab_sync):
        with patch(
            _LOADER_INIT, return_value=None
        ):
            loader = AlterLabLoader.__new__(AlterLabLoader)
            loader.urls = ["https://example.com/article"]
            loader.api_key = "sk_test_abc"
            loader.api_url = "https://api.alterlab.io"
            loader.params = {}

        with patch("alterlab.AlterLabSync", return_value=mock_alterlab_sync):
            docs = list(loader.lazy_load())

        assert len(docs) == 1
        assert isinstance(docs[0], Document)
        assert "Example Article" in docs[0].page_content
        mock_alterlab_sync.scrape.assert_called_once_with("https://example.com/article")
        mock_alterlab_sync.close.assert_called_once()

    def test_multiple_urls(self, mock_alterlab_sync):
        with patch(
            _LOADER_INIT, return_value=None
        ):
            loader = AlterLabLoader.__new__(AlterLabLoader)
            loader.urls = ["https://example.com/1", "https://example.com/2"]
            loader.api_key = "sk_test_abc"
            loader.api_url = "https://api.alterlab.io"
            loader.params = {}

        with patch("alterlab.AlterLabSync", return_value=mock_alterlab_sync):
            docs = list(loader.lazy_load())

        assert len(docs) == 2
        assert mock_alterlab_sync.scrape.call_count == 2

    def test_params_forwarded(self, mock_alterlab_sync):
        with patch(
            _LOADER_INIT, return_value=None
        ):
            loader = AlterLabLoader.__new__(AlterLabLoader)
            loader.urls = ["https://example.com"]
            loader.api_key = "sk_test_abc"
            loader.api_url = "https://api.alterlab.io"
            loader.params = {"formats": ["markdown"], "mode": "js"}

        with patch("alterlab.AlterLabSync", return_value=mock_alterlab_sync):
            list(loader.lazy_load())

        mock_alterlab_sync.scrape.assert_called_once_with(
            "https://example.com", formats=["markdown"], mode="js"
        )

    def test_scrape_error_continues(self, mock_alterlab_sync):
        mock_alterlab_sync.scrape.side_effect = [
            Exception("Network error"),
            SAMPLE_SCRAPE_RESPONSE,
        ]

        with patch(
            _LOADER_INIT, return_value=None
        ):
            loader = AlterLabLoader.__new__(AlterLabLoader)
            loader.urls = ["https://fail.com", "https://example.com"]
            loader.api_key = "sk_test_abc"
            loader.api_url = "https://api.alterlab.io"
            loader.params = {}

        with patch("alterlab.AlterLabSync", return_value=mock_alterlab_sync):
            docs = list(loader.lazy_load())

        assert len(docs) == 1
        assert docs[0].metadata["source"] == "https://example.com/article"


# ---------------------------------------------------------------------------
# Async load tests
# ---------------------------------------------------------------------------


class TestAsyncLoad:
    def test_async_single_url(self, mock_alterlab_async):
        with patch(
            _LOADER_INIT, return_value=None
        ):
            loader = AlterLabLoader.__new__(AlterLabLoader)
            loader.urls = ["https://example.com/article"]
            loader.api_key = "sk_test_abc"
            loader.api_url = "https://api.alterlab.io"
            loader.params = {}

        async def run():
            with patch("alterlab.AlterLab", return_value=mock_alterlab_async):
                docs = []
                async for doc in loader.alazy_load():
                    docs.append(doc)
                return docs

        docs = asyncio.get_event_loop().run_until_complete(run())
        assert len(docs) == 1
        assert isinstance(docs[0], Document)
        assert "Example Article" in docs[0].page_content

    def test_async_error_continues(self, mock_alterlab_async):
        mock_alterlab_async.scrape.side_effect = [
            Exception("Timeout"),
            SAMPLE_SCRAPE_RESPONSE,
        ]

        with patch(
            _LOADER_INIT, return_value=None
        ):
            loader = AlterLabLoader.__new__(AlterLabLoader)
            loader.urls = ["https://fail.com", "https://example.com"]
            loader.api_key = "sk_test_abc"
            loader.api_url = "https://api.alterlab.io"
            loader.params = {}

        async def run():
            with patch("alterlab.AlterLab", return_value=mock_alterlab_async):
                docs = []
                async for doc in loader.alazy_load():
                    docs.append(doc)
                return docs

        docs = asyncio.get_event_loop().run_until_complete(run())
        assert len(docs) == 1


# ---------------------------------------------------------------------------
# Integration-style tests (still mocked, but test the full load() path)
# ---------------------------------------------------------------------------


class TestLoadIntegration:
    def test_load_returns_list(self, mock_alterlab_sync):
        """Test that load() returns a list (provided by BaseLoader)."""
        with patch(
            _LOADER_INIT, return_value=None
        ):
            loader = AlterLabLoader.__new__(AlterLabLoader)
            loader.urls = ["https://example.com"]
            loader.api_key = "sk_test_abc"
            loader.api_url = "https://api.alterlab.io"
            loader.params = {}

        with patch("alterlab.AlterLabSync", return_value=mock_alterlab_sync):
            docs = loader.load()

        assert isinstance(docs, list)
        assert len(docs) == 1
