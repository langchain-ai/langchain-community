import os

import pytest
from langchain_core.documents import Document

from langchain_community.retrievers.you import YouRetriever
from langchain_community.utilities.you import YouSearchAPIWrapper

EXPECTED_METADATA_KEYS = {
    "url",
    "title",
    "description",
    "thumbnail_url",
    "favicon_url",
    "page_age",
}


@pytest.fixture(autouse=True)
def _require_api_key() -> None:
    if not os.getenv("YDC_API_KEY"):
        pytest.skip("YDC_API_KEY not set")


class TestYouSearchAPIWrapper:
    def test_raw_results_structure(self) -> None:
        wrapper = YouSearchAPIWrapper(count=3)
        raw = wrapper.raw_results("test query")

        assert "results" in raw
        assert "web" in raw["results"]
        assert isinstance(raw["results"]["web"], list)
        assert len(raw["results"]["web"]) > 0

        hit = raw["results"]["web"][0]
        assert "url" in hit
        assert "title" in hit
        assert "snippets" in hit
        assert isinstance(hit["snippets"], list)

    def test_results_parsed_metadata(self) -> None:
        wrapper = YouSearchAPIWrapper(count=3)
        docs = wrapper.results("python programming")

        assert len(docs) > 0
        assert all(isinstance(d, Document) for d in docs)

        doc = docs[0]
        assert isinstance(doc.page_content, str)
        assert len(doc.page_content) > 0
        assert doc.metadata.keys() == EXPECTED_METADATA_KEYS
        assert doc.metadata["url"].startswith("http")

    def test_livecrawl_uses_full_content(self) -> None:
        wrapper = YouSearchAPIWrapper(
            count=2, livecrawl="web", livecrawl_formats="markdown"
        )
        docs = wrapper.results("what is python")

        assert len(docs) > 0
        assert docs[0].page_content
        assert len(docs[0].page_content) > 500
        assert docs[0].metadata.keys() == EXPECTED_METADATA_KEYS

    def test_contents_parsed_metadata(self) -> None:
        wrapper = YouSearchAPIWrapper()
        docs = wrapper.contents(["https://example.com"])

        assert len(docs) == 1
        assert docs[0].metadata["url"] == "https://example.com"
        assert len(docs[0].page_content) > 0

    def test_contents_markdown_format(self) -> None:
        wrapper = YouSearchAPIWrapper()
        docs = wrapper.contents(["https://example.com"], formats=["markdown"])

        assert len(docs) == 1
        assert not docs[0].page_content.strip().startswith("<!")

    async def test_results_async(self) -> None:
        wrapper = YouSearchAPIWrapper(count=3)
        docs = await wrapper.results_async("test query")

        assert len(docs) > 0
        assert docs[0].metadata.keys() == EXPECTED_METADATA_KEYS

    async def test_contents_async(self) -> None:
        wrapper = YouSearchAPIWrapper()
        docs = await wrapper.contents_async(["https://example.com"])

        assert len(docs) == 1
        assert docs[0].metadata["url"] == "https://example.com"


class TestYouRetriever:
    def test_invoke(self) -> None:
        retriever = YouRetriever(count=3)
        docs = retriever.invoke("test query")

        assert len(docs) > 0
        assert isinstance(docs[0], Document)
        assert docs[0].metadata.keys() == EXPECTED_METADATA_KEYS

    async def test_ainvoke(self) -> None:
        retriever = YouRetriever(count=3)
        docs = await retriever.ainvoke("test query")

        assert len(docs) > 0
        assert docs[0].metadata.keys() == EXPECTED_METADATA_KEYS

    def test_invoke_with_livecrawl(self) -> None:
        retriever = YouRetriever(count=2, livecrawl="web", livecrawl_formats="markdown")
        docs = retriever.invoke("python tutorial")

        assert len(docs) > 0
        assert len(docs[0].page_content) > 500

    async def test_ainvoke_with_livecrawl(self) -> None:
        retriever = YouRetriever(count=2, livecrawl="web", livecrawl_formats="markdown")
        docs = await retriever.ainvoke("python tutorial")

        assert len(docs) > 0
        assert len(docs[0].page_content) > 500
