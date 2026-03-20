"""Anybrowse document loader."""
from __future__ import annotations

from typing import Iterator, List, Optional

import requests

from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseLoader


class AnybrowseLoader(BaseLoader):
    """Load web pages using the `Anybrowse` scraping service.

    Anybrowse (https://anybrowse.dev) converts any URL to clean markdown,
    handling JavaScript-rendered pages and Cloudflare-protected sites via
    a multi-tier residential IP fallback pool.

    Setup:
        Optional API key at https://anybrowse.dev (50 requests/day free).
        Anonymous use: 10 calls/day, no API key required.

        .. code-block:: bash

            pip install -U langchain-community requests

    Instantiate:
        .. code-block:: python

            from langchain_community.document_loaders import AnybrowseLoader

            loader = AnybrowseLoader(
                urls=["https://example.com"],
                api_key="ab_your_key_here",  # optional
            )

    Load:
        .. code-block:: python

            docs = loader.load()
            print(docs[0].page_content[:200])

    Attributes:
        urls: List of URLs to scrape.
        api_key: Optional Anybrowse API key.
        timeout: HTTP request timeout in seconds.
    """

    BASE_URL: str = "https://anybrowse.dev/scrape"

    def __init__(
        self,
        urls: List[str],
        api_key: Optional[str] = None,
        timeout: int = 30,
    ) -> None:
        """Initialize AnybrowseLoader.

        Args:
            urls: List of URLs to scrape.
            api_key: Optional API key for higher rate limits.
                     Get one free at https://anybrowse.dev.
            timeout: HTTP request timeout in seconds. Default: 30.
        """
        self.urls = urls
        self.api_key = api_key
        self.timeout = timeout

    def lazy_load(self) -> Iterator[Document]:
        """Lazily load documents from URLs.

        Yields:
            Document objects with markdown content and metadata.

        Raises:
            ValueError: If the HTTP request fails.
        """
        headers: dict = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        for url in self.urls:
            try:
                response = requests.post(
                    self.BASE_URL,
                    json={"url": url},
                    headers=headers,
                    timeout=self.timeout,
                )
                response.raise_for_status()
            except requests.RequestException as e:
                raise ValueError(f"Failed to scrape {url}: {e}") from e

            data = response.json()
            if data.get("status") == "success" and data.get("markdown"):
                yield Document(
                    page_content=data["markdown"],
                    metadata={
                        "source": url,
                        "title": data.get("title", ""),
                        "scraper": "anybrowse",
                    },
                )
