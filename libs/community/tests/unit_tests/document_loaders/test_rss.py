from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from langchain_community.document_loaders import RSSFeedLoader


@pytest.mark.requires("feedparser", "newspaper")
def test_rss_load_success() -> None:
    """Test successful loading of RSS feed using mocks."""

    # 1. Prepare Mock Data
    # Mock the object returned by feedparser
    mock_feed_data = MagicMock()
    mock_feed_data.bozo = False  # Indicates successful parsing

    # Create a mock entry object
    mock_entry = MagicMock()
    mock_entry.link = "http://example.com/article1"

    # Configure .get() to return a date tuple for 'published_parsed'
    # parsing: (year, month, day, hour, min, sec, wday, yday, isdst)
    def mock_entry_get(key: str, default: Any = None) -> Any:
        if key == "published_parsed":
            return (2023, 12, 1, 10, 0, 0, 0, 0, 0)
        return default

    mock_entry.get.side_effect = mock_entry_get

    mock_feed_data.entries = [mock_entry]

    # Mock the Document returned by NewsURLLoader
    mock_doc = Document(
        page_content="Mock news content",
        metadata={"title": "Mock Title", "language": "en"},
    )

    # 2. Patch dependencies
    # We need to patch two places:
    # (A) feedparser.parse -> To avoid actual network requests for the RSS feed
    # (B) NewsURLLoader -> To avoid scraping the actual article content
    with patch("feedparser.parse", return_value=mock_feed_data):
        with patch(
            "langchain_community.document_loaders.rss.NewsURLLoader"
        ) as MockNewsLoader:
            # Configure the mock NewsURLLoader instance to return our mock document
            mock_loader_instance = MockNewsLoader.return_value
            mock_loader_instance.load.return_value = [mock_doc]

            # 3. Act
            loader = RSSFeedLoader(urls=["http://fake-rss-feed.com/feed"])
            docs = loader.load()

            # 4. Assert
            assert len(docs) == 1
            assert docs[0].page_content == "Mock news content"
            assert docs[0].metadata["feed"] == "http://fake-rss-feed.com/feed"
            # Verify that the publish_date was correctly extracted
            assert "publish_date" in docs[0].metadata
            # Verify the year is correct (based on our mock tuple)
            assert docs[0].metadata["publish_date"].year == 2023


@pytest.mark.requires("feedparser", "newspaper")
def test_continue_on_failure_true() -> None:
    """Test exception is not raised when continue_on_failure=True."""
    # Mock feedparser to simulate a network error
    with patch("feedparser.parse", side_effect=Exception("Network error")):
        loader = RSSFeedLoader(["badurl.foobar"])
        # It should not raise an error, but return an empty list
        docs = loader.load()
        assert docs == []


@pytest.mark.requires("feedparser", "newspaper")
def test_continue_on_failure_false() -> None:
    """Test exception is raised when continue_on_failure=False."""
    # Mock feedparser to simulate a network error
    with patch("feedparser.parse", side_effect=Exception("Network error")):
        loader = RSSFeedLoader(["badurl.foobar"], continue_on_failure=False)
        with pytest.raises(Exception):
            loader.load()
