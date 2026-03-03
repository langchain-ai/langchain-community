"""AlterLab document loader for LangChain.

This module provides a document loader that uses the AlterLab API to scrape
web pages and return them as LangChain Document objects with structured metadata.

Setup:
    Install ``alterlab`` and ``langchain-community``, then set your API key:

    .. code-block:: bash

        pip install -U alterlab langchain-community
        export ALTERLAB_API_KEY="your-api-key"

Instantiate:
    .. code-block:: python

        from langchain_community.document_loaders import AlterLabLoader

        loader = AlterLabLoader(
            url="https://example.com/article",
        )

Lazy load:
    .. code-block:: python

        docs = []
        for doc in loader.lazy_load():
            docs.append(doc)
        print(docs[0].page_content[:100])
        print(docs[0].metadata)

Async load:
    .. code-block:: python

        docs = await loader.aload()
        print(docs[0].page_content[:100])
        print(docs[0].metadata)
"""

import logging
import os
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class AlterLabLoader(BaseLoader):
    """Load web pages using the AlterLab scraping API.

    AlterLab returns structured data with typed metadata (title, author,
    published date, content type) and supports multiple output formats
    optimized for LLM consumption.

    Setup:
        Install ``alterlab`` and set environment variable ``ALTERLAB_API_KEY``.

        .. code-block:: bash

            pip install -U alterlab langchain-community
            export ALTERLAB_API_KEY="your-api-key"

    Instantiate:
        .. code-block:: python

            from langchain_community.document_loaders import AlterLabLoader

            # Single URL
            loader = AlterLabLoader(
                url="https://example.com/article",
            )

            # Multiple URLs
            loader = AlterLabLoader(
                urls=[
                    "https://example.com/page1",
                    "https://example.com/page2",
                ],
            )

            # With custom parameters
            loader = AlterLabLoader(
                url="https://example.com",
                params={
                    "formats": ["markdown", "json"],
                    "extraction_profile": "article",
                },
            )

    Lazy load:
        .. code-block:: python

            docs = []
            docs_lazy = loader.lazy_load()

            for doc in docs_lazy:
                docs.append(doc)
            print(docs[0].page_content[:100])
            print(docs[0].metadata)

    Async load:
        .. code-block:: python

            docs = await loader.aload()
            print(docs[0].page_content[:100])
            print(docs[0].metadata)

    Metadata fields:
        - ``source``: The URL that was scraped
        - ``title``: Page title (if detected)
        - ``author``: Content author (if detected)
        - ``published_at``: Publication date (if detected)
        - ``status_code``: HTTP response status code
        - ``extraction_method``: How content was extracted (e.g., "algorithmic", "playbook:amazon")
        - ``response_time_ms``: API response time in milliseconds
        - ``credits_used``: AlterLab credits consumed
        - ``tier_used``: Scraping tier used (1=curl, 2=http, 3=stealth, 4=browser)

    See https://docs.alterlab.io for full API documentation.
    """  # noqa: E501

    def __init__(
        self,
        url: Optional[str] = None,
        *,
        urls: Optional[List[str]] = None,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize AlterLabLoader.

        Args:
            url: Single URL to scrape.
            urls: List of URLs to scrape. Cannot be used with ``url``.
            api_key: AlterLab API key. Defaults to ``ALTERLAB_API_KEY``
                environment variable.
            api_url: AlterLab API base URL. Defaults to ``ALTERLAB_API_URL``
                environment variable or ``https://api.alterlab.io``.
            params: Additional parameters passed to the AlterLab scrape API.
                See https://docs.alterlab.io for available options. Common
                parameters include ``formats``, ``mode``, ``extraction_profile``,
                ``extraction_schema``, ``advanced``, and ``cost_controls``.

        Raises:
            ImportError: If the ``alterlab`` package is not installed.
            ValueError: If neither ``url`` nor ``urls`` is provided, or if both
                are provided.
        """
        try:
            import alterlab  # noqa: F401
        except ImportError:
            msg = (
                "Could not import alterlab python package. "
                "Please install it with `pip install alterlab`."
            )
            raise ImportError(msg)

        if url and urls:
            msg = "Provide either 'url' or 'urls', not both."
            raise ValueError(msg)
        if not url and not urls:
            msg = "Either 'url' or 'urls' must be provided."
            raise ValueError(msg)

        self.api_key = api_key or os.getenv("ALTERLAB_API_KEY", "")
        if not self.api_key:
            msg = (
                "AlterLab API key is required. Pass api_key or set the "
                "ALTERLAB_API_KEY environment variable."
            )
            raise ValueError(msg)

        self.api_url = (
            api_url or os.getenv("ALTERLAB_API_URL") or "https://api.alterlab.io"
        )
        self.urls = urls if urls else [url]  # type: ignore[list-item]
        self.params = params or {}

    def lazy_load(self) -> Iterator[Document]:
        """Load documents from AlterLab API synchronously.

        Yields:
            Document objects with page content and structured metadata.
        """
        from alterlab import AlterLabSync

        client = AlterLabSync(api_key=self.api_key, base_url=self.api_url)
        try:
            for url in self.urls:
                try:
                    result = client.scrape(url, **self.params)
                    doc = self._result_to_document(result, url)
                    if doc:
                        yield doc
                except Exception as e:
                    logger.warning("Failed to scrape %s: %s", url, e)
                    continue
        finally:
            client.close()

    async def alazy_load(self) -> AsyncIterator[Document]:
        """Load documents from AlterLab API asynchronously.

        Yields:
            Document objects with page content and structured metadata.
        """
        from alterlab import AlterLab

        async with AlterLab(api_key=self.api_key, base_url=self.api_url) as client:
            for url in self.urls:
                try:
                    result = await client.scrape(url, **self.params)
                    doc = self._result_to_document(result, url)
                    if doc:
                        yield doc
                except Exception as e:
                    logger.warning("Failed to scrape %s: %s", url, e)
                    continue

    @staticmethod
    def _extract_page_content(result: Dict[str, Any]) -> str:
        """Extract page content from API response.

        Priority: markdown > text > html > raw content string.

        Args:
            result: AlterLab API response dictionary.

        Returns:
            Extracted content string, or empty string if no content found.
        """
        content = result.get("content")

        # Multi-format response (dict with format keys)
        if isinstance(content, dict):
            # Prefer markdown for LLM consumption
            if content.get("markdown"):
                return content["markdown"]
            if content.get("text"):
                return content["text"]
            if content.get("html"):
                return content["html"]
            # Empty dict or no usable format
            return ""

        # Legacy single-format response (string)
        if isinstance(content, str):
            return content

        return ""

    @staticmethod
    def _result_to_document(
        result: Dict[str, Any], source_url: str
    ) -> Optional[Document]:
        """Convert an AlterLab API response to a LangChain Document.

        Args:
            result: AlterLab API response dictionary.
            source_url: The URL that was scraped.

        Returns:
            A Document with page_content and metadata, or None if no content.
        """
        page_content = AlterLabLoader._extract_page_content(result)
        if not page_content:
            return None

        # Build metadata from response fields
        metadata: Dict[str, Any] = {"source": result.get("url", source_url)}

        # Content metadata
        if result.get("title"):
            metadata["title"] = result["title"]
        if result.get("author"):
            metadata["author"] = result["author"]
        if result.get("published_at"):
            metadata["published_at"] = result["published_at"]

        # Technical metadata
        if result.get("status_code") is not None:
            metadata["status_code"] = result["status_code"]
        if result.get("extraction_method"):
            metadata["extraction_method"] = result["extraction_method"]
        if result.get("response_time_ms") is not None:
            metadata["response_time_ms"] = result["response_time_ms"]

        # Billing metadata
        billing = result.get("billing")
        if isinstance(billing, dict):
            if billing.get("total_credits") is not None:
                metadata["credits_used"] = billing["total_credits"]
            if billing.get("tier_used"):
                metadata["tier_used"] = billing["tier_used"]
        elif result.get("credits_used") is not None:
            # Legacy response format
            metadata["credits_used"] = result["credits_used"]

        # Include any additional metadata from the API response
        extra_meta = result.get("metadata")
        if isinstance(extra_meta, dict):
            for key, value in extra_meta.items():
                if key not in metadata and value is not None:
                    metadata[key] = value

        return Document(page_content=page_content, metadata=metadata)
