"""Integration tests for LandingAIADEDocumentLoader."""

import os
from pathlib import Path

import pytest


@pytest.mark.skipif(
    not os.environ.get("VISION_AGENT_API_KEY"),
    reason="Landing AI API key not found",
)
def test_landing_ai_ade_loader_local_pdf() -> None:
    """Test LandingAIADEDocumentLoader with local PDF."""
    from langchain_community.document_loaders import LandingAIADEDocumentLoader

    file_path = Path(__file__).parent.parent / "examples/hello.pdf"
    loader = LandingAIADEDocumentLoader(file_path=file_path)
    docs = loader.load()

    assert len(docs) > 0
    assert all(doc.page_content for doc in docs)
    assert all("source" in doc.metadata for doc in docs)
    assert all("chunk" in doc.metadata for doc in docs)


@pytest.mark.skipif(
    not os.environ.get("VISION_AGENT_API_KEY"),
    reason="Landing AI API key not found",
)
def test_landing_ai_ade_loader_url() -> None:
    """Test LandingAIADEDocumentLoader with URL."""
    from langchain_community.document_loaders import LandingAIADEDocumentLoader

    url = "https://people.sc.fsu.edu/~jpeterson/hello_world.pdf"
    loader = LandingAIADEDocumentLoader(file_path=url)
    docs = loader.load()

    assert len(docs) > 0
    assert all(doc.page_content for doc in docs)
    assert all(doc.metadata["source"] == url for doc in docs)


@pytest.mark.skipif(
    not os.environ.get("VISION_AGENT_API_KEY"),
    reason="Landing AI API key not found",
)
def test_landing_ai_ade_loader_lazy_load() -> None:
    """Test LandingAIADEDocumentLoader lazy_load method."""
    from langchain_community.document_loaders import LandingAIADEDocumentLoader

    file_path = Path(__file__).parent.parent / "examples/hello.pdf"
    loader = LandingAIADEDocumentLoader(file_path=file_path)

    docs = []
    for doc in loader.lazy_load():
        docs.append(doc)

    assert len(docs) > 0
    assert all(doc.page_content for doc in docs)
