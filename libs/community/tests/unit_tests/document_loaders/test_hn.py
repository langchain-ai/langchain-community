"""Unit tests for HNLoader."""

from unittest.mock import MagicMock

from langchain_community.document_loaders.hn import HNLoader


class TestHNLoader:
    """Tests for HNLoader."""

    def test_load_comments_with_valid_pagespace(self) -> None:
        """Test load_comments when pagespace element exists."""
        loader = HNLoader("https://news.ycombinator.com/item?id=12345")

        # Create mock soup with valid structure
        mock_soup = MagicMock()
        mock_comment = MagicMock()
        mock_comment.text.strip.return_value = "This is a comment"
        mock_soup.select.return_value = [mock_comment]

        mock_pagespace = MagicMock()
        mock_pagespace.get.return_value = "Test Title"
        mock_soup.select_one.return_value = mock_pagespace

        docs = loader.load_comments(mock_soup)

        assert len(docs) == 1
        assert docs[0].page_content == "This is a comment"
        assert docs[0].metadata["title"] == "Test Title"

    def test_load_comments_with_missing_pagespace(self) -> None:
        """Test load_comments when pagespace element is missing (bug fix for #494)."""
        loader = HNLoader("https://news.ycombinator.com/item?id=12345")

        # Create mock soup where pagespace is None
        mock_soup = MagicMock()
        mock_comment = MagicMock()
        mock_comment.text.strip.return_value = "This is a comment"
        mock_soup.select.return_value = [mock_comment]
        mock_soup.select_one.return_value = None  # pagespace not found

        # This should NOT raise AttributeError anymore
        docs = loader.load_comments(mock_soup)

        assert len(docs) == 1
        assert docs[0].page_content == "This is a comment"
        assert docs[0].metadata["title"] is None

    def test_load_comments_with_empty_comments(self) -> None:
        """Test load_comments when no comments are found."""
        loader = HNLoader("https://news.ycombinator.com/item?id=12345")

        mock_soup = MagicMock()
        mock_soup.select.return_value = []
        mock_soup.select_one.return_value = None

        docs = loader.load_comments(mock_soup)

        assert len(docs) == 0

    def test_load_results_with_valid_items(self) -> None:
        """Test load_results when items have valid structure."""
        loader = HNLoader("https://news.ycombinator.com/")

        mock_soup = MagicMock()

        # Create a mock item with valid structure
        mock_item = MagicMock()
        mock_rank = MagicMock()
        mock_rank.text = "1."
        mock_item.select_one.return_value = mock_rank

        mock_titleline = MagicMock()
        mock_titleline.text.strip.return_value = "Test Article"
        mock_link = MagicMock()
        mock_link.get.return_value = "https://example.com"
        mock_titleline.find.return_value = mock_link
        mock_item.find.return_value = mock_titleline

        mock_soup.select.return_value = [mock_item]

        docs = loader.load_results(mock_soup)

        assert len(docs) == 1
        assert docs[0].page_content == "Test Article"
        assert docs[0].metadata["ranking"] == "1."
        assert docs[0].metadata["link"] == "https://example.com"

    def test_load_results_with_missing_elements(self) -> None:
        """Test load_results when elements are missing."""
        loader = HNLoader("https://news.ycombinator.com/")

        mock_soup = MagicMock()

        # Create a mock item with missing elements
        mock_item = MagicMock()
        mock_item.select_one.return_value = None  # No rank element
        mock_item.find.return_value = None  # No titleline

        mock_soup.select.return_value = [mock_item]

        # This should NOT raise AttributeError
        docs = loader.load_results(mock_soup)

        assert len(docs) == 1
        assert docs[0].metadata["ranking"] is None
        assert docs[0].metadata["link"] is None
        assert docs[0].metadata["title"] is None

    def test_load_results_with_empty_items(self) -> None:
        """Test load_results when no items are found."""
        loader = HNLoader("https://news.ycombinator.com/")

        mock_soup = MagicMock()
        mock_soup.select.return_value = []

        docs = loader.load_results(mock_soup)

        assert len(docs) == 0
