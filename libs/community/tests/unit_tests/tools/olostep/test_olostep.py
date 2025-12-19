"""Unit tests for Olostep tools."""

import os
import unittest
from typing import Any
from unittest.mock import MagicMock, patch

os.environ["OLOSTEP_API_KEY"] = "test_key"

from langchain_community.tools.olostep import (
    OlostepAnswers,
    OlostepCrawl,
    OlostepMap,
    OlostepScrape,
)
from langchain_community.utilities.olostep import OlostepAPIWrapper


class TestOlostepScrape(unittest.TestCase):
    """Test the OlostepScrape tool."""

    @patch.object(OlostepAPIWrapper, "scrape")
    def test_invoke(self, mock_scrape: Any) -> None:
        """Test that the tool can be invoked."""
        mock_scrape.return_value = {
            "retrieve_id": "test_id",
            "result": {
                "markdown_content": "# Test Content\n\nThis is test content.",
                "page_metadata": {"title": "Test Page"},
            },
        }

        wrapper = OlostepAPIWrapper(olostep_api_key="test_key")  # type: ignore[arg-type]
        tool = OlostepScrape(api_wrapper=wrapper)  # type: ignore[call-arg]
        content, artifact = tool._run("https://example.com", format="markdown")

        assert "# Test Content" in content
        mock_scrape.assert_called_once()

    def test_tool_attributes(self) -> None:
        """Test that the tool has correct attributes."""
        tool = OlostepScrape()
        assert tool.name == "olostep_scrape"
        assert "scrape" in tool.description.lower()
        assert "website" in tool.description.lower()


class TestOlostepAnswers(unittest.TestCase):
    """Test the OlostepAnswers tool."""

    @patch.object(OlostepAPIWrapper, "answer")
    def test_invoke(self, mock_answer: Any) -> None:
        """Test that the tool can be invoked."""
        mock_answer.return_value = {
            "id": "answer_test",
            "result": {
                "json_content": '{"answer": "Test answer"}',
                "sources": ["https://source1.com", "https://source2.com"],
            },
        }

        wrapper = OlostepAPIWrapper(olostep_api_key="test_key")  # type: ignore[arg-type]
        tool = OlostepAnswers(api_wrapper=wrapper)  # type: ignore[call-arg]
        content, artifact = tool._run("What is AI?")

        assert "answer" in content.lower()
        mock_answer.assert_called_once()

    @patch.object(OlostepAPIWrapper, "answer")
    def test_invoke_with_schema(self, mock_answer: Any) -> None:
        """Test that the tool can be invoked with JSON schema."""
        mock_answer.return_value = {
            "id": "answer_test",
            "result": {
                "json_content": '{"company": "Stripe", "ceo": "Patrick Collison"}',
                "sources": ["https://source1.com"],
            },
        }

        wrapper = OlostepAPIWrapper(olostep_api_key="test_key")  # type: ignore[arg-type]
        tool = OlostepAnswers(api_wrapper=wrapper)  # type: ignore[call-arg]
        content, artifact = tool._run(
            "Find info about Stripe", json_schema={"company": "", "ceo": ""}
        )

        assert "Stripe" in content
        mock_answer.assert_called_once()

    def test_tool_attributes(self) -> None:
        """Test that the tool has correct attributes."""
        tool = OlostepAnswers()
        assert tool.name == "olostep_answers"
        assert "search" in tool.description.lower() or "answer" in tool.description.lower()


class TestOlostepMap(unittest.TestCase):
    """Test the OlostepMap tool."""

    @patch.object(OlostepAPIWrapper, "map")
    def test_invoke(self, mock_map: Any) -> None:
        """Test that the tool can be invoked."""
        mock_map.return_value = {
            "id": "map_test",
            "urls": [
                "https://example.com/page1",
                "https://example.com/page2",
                "https://example.com/page3",
            ],
        }

        wrapper = OlostepAPIWrapper(olostep_api_key="test_key")  # type: ignore[arg-type]
        tool = OlostepMap(api_wrapper=wrapper)  # type: ignore[call-arg]
        content, artifact = tool._run("https://example.com")

        assert "total_urls" in content
        assert "3" in content
        mock_map.assert_called_once()

    def test_tool_attributes(self) -> None:
        """Test that the tool has correct attributes."""
        tool = OlostepMap()
        assert tool.name == "olostep_map"
        assert "url" in tool.description.lower()


class TestOlostepCrawl(unittest.TestCase):
    """Test the OlostepCrawl tool."""

    @patch.object(OlostepAPIWrapper, "crawl")
    def test_invoke(self, mock_crawl: Any) -> None:
        """Test that the tool can be invoked."""
        mock_crawl.return_value = {
            "id": "crawl_test",
            "status": "in_progress",
            "pages_crawled": 0,
        }

        wrapper = OlostepAPIWrapper(olostep_api_key="test_key")  # type: ignore[arg-type]
        tool = OlostepCrawl(api_wrapper=wrapper)  # type: ignore[call-arg]
        content, artifact = tool._run("https://docs.example.com", max_pages=50)

        assert "crawl_id" in content
        assert "in_progress" in content
        mock_crawl.assert_called_once()

    def test_tool_attributes(self) -> None:
        """Test that the tool has correct attributes."""
        tool = OlostepCrawl()
        assert tool.name == "olostep_crawl"
        assert "crawl" in tool.description.lower()


if __name__ == "__main__":
    unittest.main()
