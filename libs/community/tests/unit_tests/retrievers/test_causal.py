"""Tests for CausalRetriever."""

from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.requires("dotcausal")
def test_causal_retriever_import() -> None:
    """Test that CausalRetriever can be imported."""
    from langchain_community.retrievers import CausalRetriever

    assert CausalRetriever is not None


@pytest.mark.requires("dotcausal")
def test_causal_retriever_initialization() -> None:
    """Test CausalRetriever initialization with mocked reader."""
    from langchain_community.retrievers.causal import CausalRetriever

    with patch("langchain_community.retrievers.causal.CausalReader") as mock_reader:
        with patch("pathlib.Path.exists", return_value=True):
            mock_reader_instance = MagicMock()
            mock_reader.return_value = mock_reader_instance

            retriever = CausalRetriever(
                file_path="test.causal",
                top_k=5,
                include_inferred=True,
                min_confidence=0.5,
            )

            assert retriever.file_path == "test.causal"
            assert retriever.top_k == 5
            assert retriever.include_inferred is True
            assert retriever.min_confidence == 0.5


@pytest.mark.requires("dotcausal")
def test_causal_retriever_search() -> None:
    """Test CausalRetriever search functionality."""
    from langchain_community.retrievers.causal import CausalRetriever

    with patch("langchain_community.retrievers.causal.CausalReader") as mock_reader:
        with patch("pathlib.Path.exists", return_value=True):
            mock_reader_instance = MagicMock()
            mock_reader_instance.search.return_value = [
                {
                    "trigger": "COVID",
                    "mechanism": "causes",
                    "outcome": "fatigue",
                    "confidence": 0.9,
                    "is_inferred": False,
                    "source": "paper1.pdf",
                },
                {
                    "trigger": "SARS-CoV-2",
                    "mechanism": "damages",
                    "outcome": "mitochondria",
                    "confidence": 0.85,
                    "is_inferred": True,
                    "provenance": ["triplet_1", "triplet_2"],
                },
            ]
            mock_reader.return_value = mock_reader_instance

            retriever = CausalRetriever(file_path="test.causal", top_k=10)
            docs = retriever.invoke("COVID")

            assert len(docs) == 2
            assert "[EXPLICIT]" in docs[0].page_content
            assert "[INFERRED]" in docs[1].page_content
            assert docs[0].metadata["trigger"] == "COVID"
            assert docs[1].metadata["is_inferred"] is True


@pytest.mark.requires("dotcausal")
def test_causal_retriever_confidence_filter() -> None:
    """Test that min_confidence filter works."""
    from langchain_community.retrievers.causal import CausalRetriever

    with patch("langchain_community.retrievers.causal.CausalReader") as mock_reader:
        with patch("pathlib.Path.exists", return_value=True):
            mock_reader_instance = MagicMock()
            mock_reader_instance.search.return_value = [
                {
                    "trigger": "A",
                    "mechanism": "causes",
                    "outcome": "B",
                    "confidence": 0.9,
                    "is_inferred": False,
                },
                {
                    "trigger": "C",
                    "mechanism": "causes",
                    "outcome": "D",
                    "confidence": 0.3,  # Below threshold
                    "is_inferred": False,
                },
            ]
            mock_reader.return_value = mock_reader_instance

            retriever = CausalRetriever(
                file_path="test.causal", min_confidence=0.5
            )
            docs = retriever.invoke("test")

            # Only high confidence result should be returned
            assert len(docs) == 1
            assert docs[0].metadata["confidence"] == 0.9


@pytest.mark.requires("dotcausal")
def test_causal_retriever_exclude_inferred() -> None:
    """Test that include_inferred=False filters out inferred triplets."""
    from langchain_community.retrievers.causal import CausalRetriever

    with patch("langchain_community.retrievers.causal.CausalReader") as mock_reader:
        with patch("pathlib.Path.exists", return_value=True):
            mock_reader_instance = MagicMock()
            mock_reader_instance.search.return_value = [
                {
                    "trigger": "A",
                    "mechanism": "causes",
                    "outcome": "B",
                    "confidence": 0.9,
                    "is_inferred": False,
                },
                {
                    "trigger": "C",
                    "mechanism": "causes",
                    "outcome": "D",
                    "confidence": 0.85,
                    "is_inferred": True,  # Should be filtered
                },
            ]
            mock_reader.return_value = mock_reader_instance

            retriever = CausalRetriever(
                file_path="test.causal", include_inferred=False
            )
            docs = retriever.invoke("test")

            # Only explicit result should be returned
            assert len(docs) == 1
            assert docs[0].metadata["is_inferred"] is False
