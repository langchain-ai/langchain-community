"""Integration tests for Olostep tools.

These tests require an OLOSTEP_API_KEY environment variable to be set.

The most reliable and cost-effective web search, scraping and crawling API for AI.
Build intelligent agents that can search, scrape, analyze, and structure data
from any website.
"""

import os

import pytest

from langchain_community.tools.olostep import (
    OlostepAnswers,
    OlostepCrawl,
    OlostepMap,
    OlostepScrape,
)


@pytest.fixture
def api_key() -> str:
    """Get API key from environment."""
    key = os.environ.get("OLOSTEP_API_KEY")
    if not key:
        pytest.skip("OLOSTEP_API_KEY not set")
    return key


class TestOlostepScrapeIntegration:
    """Integration tests for OlostepScrape."""

    def test_scrape_website(self, api_key: str) -> None:
        """Test scraping a website."""
        tool = OlostepScrape(olostep_api_key=api_key)  # type: ignore[call-arg]
        content, artifact = tool._run("https://example.com", format="markdown")

        assert content
        assert len(content) > 0
        assert artifact

    def test_scrape_with_html_format(self, api_key: str) -> None:
        """Test scraping with HTML format."""
        tool = OlostepScrape(olostep_api_key=api_key)  # type: ignore[call-arg]
        content, artifact = tool._run("https://example.com", format="html")

        assert content
        assert "<" in content  # Should contain HTML tags


class TestOlostepAnswersIntegration:
    """Integration tests for OlostepAnswers."""

    def test_simple_question(self, api_key: str) -> None:
        """Test answering a simple question."""
        tool = OlostepAnswers(olostep_api_key=api_key)  # type: ignore[call-arg]
        content, artifact = tool._run("What is the capital of France?")

        assert content
        assert "answer" in content.lower() or "paris" in content.lower()

    def test_structured_answer(self, api_key: str) -> None:
        """Test answering with structured output."""
        tool = OlostepAnswers(olostep_api_key=api_key)  # type: ignore[call-arg]
        content, artifact = tool._run(
            "What is the capital of France?",
            json_schema={"capital": "", "country": ""},
        )

        assert content
        assert "sources" in content


class TestOlostepMapIntegration:
    """Integration tests for OlostepMap."""

    def test_map_website(self, api_key: str) -> None:
        """Test mapping a website."""
        tool = OlostepMap(olostep_api_key=api_key)  # type: ignore[call-arg]
        content, artifact = tool._run("https://example.com", top_n=10)

        assert content
        assert "urls" in content
        assert "total_urls" in content


class TestOlostepCrawlIntegration:
    """Integration tests for OlostepCrawl."""

    def test_start_crawl(self, api_key: str) -> None:
        """Test starting a crawl."""
        tool = OlostepCrawl(olostep_api_key=api_key)  # type: ignore[call-arg]
        content, artifact = tool._run("https://example.com", max_pages=5)

        assert content
        assert "crawl_id" in content
        assert "status" in content
