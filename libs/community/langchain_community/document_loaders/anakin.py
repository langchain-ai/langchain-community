"""Anakin document loader for web scraping, search, and agentic research.

Provides ``AnakinLoader``, a ``BaseLoader`` that wraps the Anakin API
and returns LangChain ``Document`` objects suitable for RAG pipelines.

API reference: https://anakin.io/llms-full.txt
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Dict, Iterator, List, Literal, Optional

from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseLoader
from langchain_community.utilities.anakin import AnakinAPIWrapper

logger = logging.getLogger(__name__)


class AnakinLoader(BaseLoader):
    """Load documents using the Anakin web scraping and search API.

    Supports four modes:

    - ``scrape``: Scrape a single URL and return its content as markdown.
    - ``batch_scrape``: Scrape up to 10 URLs in one request.
    - ``search``: AI-powered web search returning results with citations.
    - ``agentic_search``: Multi-stage autonomous research (1-5 min).

    Setup:
        Install ``langchain-community`` and set your API key:

        .. code-block:: bash

            pip install -U langchain-community
            export ANAKIN_API_KEY="your-api-key"

    Instantiate:
        .. code-block:: python

            from langchain_community.document_loaders import AnakinLoader

            # Scrape a single URL
            loader = AnakinLoader(url="https://example.com", mode="scrape")

            # Search the web
            loader = AnakinLoader(query="latest AI news", mode="search")

            # Batch scrape multiple URLs
            loader = AnakinLoader(
                urls=["https://a.com", "https://b.com"],
                mode="batch_scrape",
            )

            # Deep agentic research (1-5 min)
            loader = AnakinLoader(
                query="compare React vs Vue",
                mode="agentic_search",
            )

    Lazy load:
        .. code-block:: python

            docs = []
            docs_lazy = loader.lazy_load()

            for doc in docs_lazy:
                docs.append(doc)
            print(docs[0].page_content[:100])
            print(docs[0].metadata)

        .. code-block:: python

            # Example output (scrape mode):
            '# Example Domain\\nThis domain is for use in ...'
            {'source': 'https://example.com', 'title': 'Example Domain', ...}

    Async load:
        .. code-block:: python

            docs = []
            async for doc in loader.alazy_load():
                docs.append(doc)
    """  # noqa: E501

    def __init__(
        self,
        url: Optional[str] = None,
        *,
        urls: Optional[List[str]] = None,
        query: Optional[str] = None,
        api_key: Optional[str] = None,
        mode: Literal["scrape", "batch_scrape", "search", "agentic_search"] = "scrape",
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the Anakin loader.

        Args:
            url: URL to scrape (required for ``scrape`` mode).
            urls: List of URLs (required for ``batch_scrape`` mode, max 10).
            query: Search query (required for ``search`` and
                ``agentic_search`` modes).
            api_key: Anakin API key. Falls back to ``ANAKIN_API_KEY`` env var.
            mode: Operation mode — ``"scrape"``, ``"batch_scrape"``,
                ``"search"``, or ``"agentic_search"``.
            params: Extra parameters forwarded to the Anakin API.
        """
        self.url = url
        self.urls = urls
        self.query = query
        self.mode = mode
        self.params = params or {}

        wrapper_kwargs: Dict[str, Any] = {}
        if api_key:
            wrapper_kwargs["anakin_api_key"] = api_key
        self.api_wrapper = AnakinAPIWrapper(**wrapper_kwargs)

        self._validate_params()

    def _validate_params(self) -> None:
        valid_modes = {"scrape", "batch_scrape", "search", "agentic_search"}
        if self.mode not in valid_modes:
            msg = f"Invalid mode '{self.mode}'. Must be one of: {valid_modes}"
            raise ValueError(msg)
        if self.mode == "scrape" and not self.url:
            msg = "'url' is required for 'scrape' mode."
            raise ValueError(msg)
        if self.mode == "batch_scrape" and not self.urls:
            msg = "'urls' is required for 'batch_scrape' mode."
            raise ValueError(msg)
        if self.mode in ("search", "agentic_search") and not self.query:
            msg = f"'query' is required for '{self.mode}' mode."
            raise ValueError(msg)

    def lazy_load(self) -> Iterator[Document]:
        """Lazily load documents from the Anakin API.

        Yields:
            Document objects with page_content and metadata.
        """
        if self.mode == "scrape":
            yield from self._load_scrape()
        elif self.mode == "batch_scrape":
            yield from self._load_batch_scrape()
        elif self.mode == "search":
            yield from self._load_search()
        elif self.mode == "agentic_search":
            yield from self._load_agentic_search()

    async def alazy_load(self) -> AsyncIterator[Document]:
        """Async variant of :meth:`lazy_load`."""
        if self.mode == "scrape":
            for doc in await self._aload_scrape():
                yield doc
        elif self.mode == "batch_scrape":
            for doc in await self._aload_batch_scrape():
                yield doc
        elif self.mode == "search":
            for doc in await self._aload_search():
                yield doc
        elif self.mode == "agentic_search":
            for doc in await self._aload_agentic_search():
                yield doc

    # ------------------------------------------------------------------
    # Scrape
    # ------------------------------------------------------------------

    def _load_scrape(self) -> List[Document]:
        result = self.api_wrapper.scrape(self.url, **self.params)  # type: ignore[arg-type]
        markdown = result.get("markdown", "")
        metadata = {
            "source": result.get("url", self.url),
            "title": _extract_title(markdown),
            "status": result.get("status"),
            "duration_ms": result.get("durationMs"),
        }
        return [Document(page_content=markdown, metadata=metadata)]

    async def _aload_scrape(self) -> List[Document]:
        result = await self.api_wrapper.ascrape(self.url, **self.params)  # type: ignore[arg-type]
        markdown = result.get("markdown", "")
        metadata = {
            "source": result.get("url", self.url),
            "title": _extract_title(markdown),
            "status": result.get("status"),
            "duration_ms": result.get("durationMs"),
        }
        return [Document(page_content=markdown, metadata=metadata)]

    # ------------------------------------------------------------------
    # Batch Scrape
    # ------------------------------------------------------------------

    def _load_batch_scrape(self) -> List[Document]:
        result = self.api_wrapper.batch_scrape(self.urls, **self.params)  # type: ignore[arg-type]
        docs: List[Document] = []
        for item in result.get("results", []):
            if item.get("status") != "completed":
                logger.warning(
                    "Skipping failed URL %s: %s",
                    item.get("url"),
                    item.get("error", "unknown"),
                )
                continue
            markdown = item.get("markdown", "")
            metadata = {
                "source": item.get("url"),
                "title": _extract_title(markdown),
                "status": item.get("status"),
                "duration_ms": item.get("durationMs"),
            }
            docs.append(Document(page_content=markdown, metadata=metadata))
        return docs

    async def _aload_batch_scrape(self) -> List[Document]:
        result = await self.api_wrapper.abatch_scrape(self.urls, **self.params)  # type: ignore[arg-type]
        docs: List[Document] = []
        for item in result.get("results", []):
            if item.get("status") != "completed":
                logger.warning(
                    "Skipping failed URL %s: %s",
                    item.get("url"),
                    item.get("error", "unknown"),
                )
                continue
            markdown = item.get("markdown", "")
            metadata = {
                "source": item.get("url"),
                "title": _extract_title(markdown),
                "status": item.get("status"),
                "duration_ms": item.get("durationMs"),
            }
            docs.append(Document(page_content=markdown, metadata=metadata))
        return docs

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def _load_search(self) -> List[Document]:
        results = self.api_wrapper.search(self.query, **self.params)  # type: ignore[arg-type]
        docs: List[Document] = []
        for r in results:
            metadata = {
                "source": r.get("url", ""),
                "title": r.get("title", ""),
                "date": r.get("date"),
                "last_updated": r.get("last_updated"),
            }
            docs.append(Document(page_content=r.get("snippet", ""), metadata=metadata))
        return docs

    async def _aload_search(self) -> List[Document]:
        results = await self.api_wrapper.asearch(self.query, **self.params)  # type: ignore[arg-type]
        docs: List[Document] = []
        for r in results:
            metadata = {
                "source": r.get("url", ""),
                "title": r.get("title", ""),
                "date": r.get("date"),
                "last_updated": r.get("last_updated"),
            }
            docs.append(Document(page_content=r.get("snippet", ""), metadata=metadata))
        return docs

    # ------------------------------------------------------------------
    # Agentic Search
    # ------------------------------------------------------------------

    def _load_agentic_search(self) -> List[Document]:
        result = self.api_wrapper.agentic_search(self.query, **self.params)  # type: ignore[arg-type]
        generated = result.get("generatedJson", {})
        summary = generated.get("summary", "")
        structured_data = generated.get("structured_data", {})
        metadata = {
            "source": "anakin_agentic_search",
            "query": self.query,
            "structured_data": structured_data,
            "duration_ms": result.get("durationMs"),
        }
        return [Document(page_content=summary, metadata=metadata)]

    async def _aload_agentic_search(self) -> List[Document]:
        result = await self.api_wrapper.aagentic_search(self.query, **self.params)  # type: ignore[arg-type]
        generated = result.get("generatedJson", {})
        summary = generated.get("summary", "")
        structured_data = generated.get("structured_data", {})
        metadata = {
            "source": "anakin_agentic_search",
            "query": self.query,
            "structured_data": structured_data,
            "duration_ms": result.get("durationMs"),
        }
        return [Document(page_content=summary, metadata=metadata)]


def _extract_title(markdown: str) -> str:
    """Extract the first heading from markdown content as a title."""
    for line in markdown.split("\n"):
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return ""
