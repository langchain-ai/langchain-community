"""Integration tests for AnakinLoader.

These tests hit the real Anakin API and require a valid API key.
They are skipped when ANAKIN_API_KEY is not set.

Run with:
    ANAKIN_API_KEY=ak-... pytest tests/integration_tests/ -v
"""

from __future__ import annotations

import os

import pytest

from langchain_community.document_loaders.anakin import AnakinLoader

SKIP_REASON = "ANAKIN_API_KEY not set"
has_api_key = bool(os.environ.get("ANAKIN_API_KEY"))


@pytest.mark.skipif(not has_api_key, reason=SKIP_REASON)
class TestAnakinLoaderIntegration:
    """Integration tests against the live Anakin API."""

    def test_scrape_mode(self) -> None:
        """Test scraping a single URL."""
        loader = AnakinLoader(url="https://example.com", mode="scrape")
        docs = loader.load()

        assert len(docs) == 1
        assert len(docs[0].page_content) > 0
        assert docs[0].metadata["source"] == "https://example.com"
        assert docs[0].metadata["status"] == "completed"

    def test_scrape_with_browser(self) -> None:
        """Test scraping with headless browser rendering."""
        loader = AnakinLoader(
            url="https://example.com",
            mode="scrape",
            params={"use_browser": True},
        )
        docs = loader.load()

        assert len(docs) == 1
        assert len(docs[0].page_content) > 0

    def test_batch_scrape_mode(self) -> None:
        """Test batch scraping multiple URLs."""
        loader = AnakinLoader(
            urls=["https://example.com", "https://httpbin.org/html"],
            mode="batch_scrape",
        )
        docs = loader.load()

        assert len(docs) >= 1  # At least one should succeed
        for doc in docs:
            assert len(doc.page_content) > 0
            assert doc.metadata["source"] is not None

    def test_search_mode(self) -> None:
        """Test AI-powered web search."""
        loader = AnakinLoader(query="Python programming language", mode="search")
        docs = loader.load()

        assert len(docs) > 0
        for doc in docs:
            assert len(doc.page_content) > 0
            assert doc.metadata["source"]
            assert doc.metadata["title"]

    def test_agentic_search_mode(self) -> None:
        """Test deep agentic research.

        This is a long-running test (1-5 minutes).
        """
        loader = AnakinLoader(
            query="What are the latest developments in LLM agents?",
            mode="agentic_search",
        )
        docs = loader.load()

        assert len(docs) == 1
        assert len(docs[0].page_content) > 0
        assert docs[0].metadata["source"] == "anakin_agentic_search"
        assert docs[0].metadata["query"] is not None

    def test_lazy_load(self) -> None:
        """Test that lazy_load yields documents correctly."""
        loader = AnakinLoader(url="https://example.com", mode="scrape")
        docs = list(loader.lazy_load())

        assert len(docs) == 1
        assert len(docs[0].page_content) > 0
