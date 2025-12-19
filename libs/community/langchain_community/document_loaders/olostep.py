"""Document loader for Olostep.

The most reliable and cost-effective web search, scraping and crawling API for AI.
Build intelligent agents that can search, scrape, analyze, and structure data
from any website.
"""

from typing import Any, Iterator, List, Literal, Optional

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from langchain_core.utils import get_from_dict_or_env


class OlostepLoader(BaseLoader):
    """Load web content using Olostep API.

    The most reliable and cost-effective web search, scraping and crawling API for AI.
    Build intelligent agents that can search, scrape, analyze, and structure data
    from any website.

    Features:
        - Scrape: Extract content from any website
        - Crawl: Autonomously discover and scrape entire websites
        - Map: Extract all URLs from a website for site structure analysis
        - Batch: Process up to 10,000 URLs in parallel

    Setup:
        Set environment variable ``OLOSTEP_API_KEY``.

        .. code-block:: bash

            pip install -U langchain-community aiohttp requests
            export OLOSTEP_API_KEY="your-api-key"

    Instantiate:
        .. code-block:: python

            from langchain_community.document_loaders import OlostepLoader

            # Scrape a single URL
            loader = OlostepLoader(
                url="https://example.com",
                mode="scrape"
            )

            # Crawl an entire website
            loader = OlostepLoader(
                url="https://docs.example.com",
                mode="crawl",
                params={"max_pages": 100}
            )

    Lazy load:
        .. code-block:: python

            docs = []
            docs_lazy = loader.lazy_load()

            for doc in docs_lazy:
                docs.append(doc)
            print(docs[0].page_content[:100])
            print(docs[0].metadata)

    .. versionadded:: 0.3.0
    """

    def __init__(
        self,
        url: Optional[str] = None,
        urls: Optional[List[str]] = None,
        *,
        api_key: Optional[str] = None,
        mode: Literal["scrape", "crawl", "map"] = "scrape",
        params: Optional[dict] = None,
    ):
        """Initialize the loader.

        Args:
            url: The URL to scrape or starting URL for crawl/map.
            urls: List of URLs for batch scraping. Used when mode is not specified
                or in batch operations.
            api_key: The Olostep API key. If not specified, reads from env var
                OLOSTEP_API_KEY. Get an API key at https://olostep.com
            mode: The mode to run the loader in. Default is "scrape".
                Options:
                - "scrape": Extract content from a single URL
                - "crawl": Autonomously discover and scrape entire website
                - "map": Extract all URLs from a website
            params: Additional parameters to pass to the Olostep API.
                For scrape mode:
                    - formats: List of formats ["markdown", "html", "json", "text"]
                    - country: ISO country code (e.g., "US", "GB")
                    - wait_before_scraping: Wait time in ms (0-10000)
                For crawl mode:
                    - max_pages: Maximum pages to crawl (default: 100)
                    - include_urls: Glob patterns to include
                    - exclude_urls: Glob patterns to exclude
                    - max_depth: Maximum link depth
                For map mode:
                    - search_query: Filter URLs by relevance
                    - top_n: Maximum URLs to return
                    - include_urls: Glob patterns to include
                    - exclude_urls: Glob patterns to exclude
        """
        self.url = url
        self.urls = urls or []
        self.mode = mode
        self.params = params or {}

        # Get API key from environment if not provided
        import os

        self.api_key = api_key or os.environ.get("OLOSTEP_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OLOSTEP_API_KEY environment variable not set. "
                "Get your API key at https://olostep.com"
            )

        # Validate inputs
        if mode in ("scrape", "crawl", "map") and not url:
            raise ValueError(f"url must be provided for mode '{mode}'")

    def lazy_load(self) -> Iterator[Document]:
        """Load documents lazily.

        Yields:
            Document objects containing scraped content.
        """
        if self.mode == "scrape":
            yield from self._scrape()
        elif self.mode == "crawl":
            yield from self._crawl()
        elif self.mode == "map":
            yield from self._map()
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _scrape(self) -> Iterator[Document]:
        """Scrape a single URL or batch of URLs."""
        import json

        import requests

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Handle single URL
        if self.url:
            urls_to_scrape = [self.url] + self.urls
        else:
            urls_to_scrape = self.urls

        for url in urls_to_scrape:
            payload: dict[str, Any] = {
                "url_to_scrape": url,
                "formats": self.params.get("formats", ["markdown"]),
            }

            if "country" in self.params:
                payload["country"] = self.params["country"]
            if "wait_before_scraping" in self.params:
                payload["wait_before_scraping"] = self.params["wait_before_scraping"]

            try:
                response = requests.post(
                    "https://api.olostep.com/v1/scrapes",
                    headers=headers,
                    json=payload,
                    timeout=60,
                )
                response.raise_for_status()
                result = response.json()

                content_result = result.get("result", {})
                markdown_content = content_result.get("markdown_content", "")
                page_metadata = content_result.get("page_metadata", {})

                yield Document(
                    page_content=markdown_content,
                    metadata={
                        "source": url,
                        "title": page_metadata.get("title", ""),
                        "description": page_metadata.get("description", ""),
                        "scrape_id": result.get("retrieve_id", ""),
                    },
                )
            except Exception as e:
                # Yield error as document so caller can handle
                yield Document(
                    page_content=f"Error scraping {url}: {str(e)}",
                    metadata={"source": url, "error": True},
                )

    def _crawl(self) -> Iterator[Document]:
        """Crawl a website and yield documents.

        Note: Crawling is asynchronous. This method starts the crawl
        and polls for results.
        """
        import json
        import time

        import requests

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload: dict[str, Any] = {
            "start_url": self.url,
            "max_pages": self.params.get("max_pages", 100),
        }

        if "include_urls" in self.params:
            payload["include_urls"] = self.params["include_urls"]
        if "exclude_urls" in self.params:
            payload["exclude_urls"] = self.params["exclude_urls"]
        if "max_depth" in self.params:
            payload["max_depth"] = self.params["max_depth"]
        if "include_external" in self.params:
            payload["include_external"] = self.params["include_external"]

        # Start the crawl
        response = requests.post(
            "https://api.olostep.com/v1/crawls",
            headers=headers,
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        crawl_result = response.json()

        crawl_id = crawl_result.get("id", "")
        if not crawl_id:
            yield Document(
                page_content="Error: No crawl ID returned",
                metadata={"source": self.url, "error": True},
            )
            return

        # Poll for crawl completion
        max_wait = self.params.get("max_wait_seconds", 600)  # 10 minutes default
        poll_interval = self.params.get("poll_interval", 5)  # 5 seconds
        elapsed = 0

        while elapsed < max_wait:
            status_response = requests.get(
                f"https://api.olostep.com/v1/crawls/{crawl_id}",
                headers=headers,
                timeout=30,
            )
            status_response.raise_for_status()
            status_data = status_response.json()

            status = status_data.get("status", "")
            if status == "completed":
                # Get crawl results
                pages = status_data.get("pages", [])
                for page in pages:
                    yield Document(
                        page_content=page.get("markdown_content", ""),
                        metadata={
                            "source": page.get("url", ""),
                            "title": page.get("title", ""),
                            "crawl_id": crawl_id,
                        },
                    )
                return
            elif status == "failed":
                yield Document(
                    page_content=f"Crawl failed: {status_data.get('error', 'Unknown')}",
                    metadata={"source": self.url, "crawl_id": crawl_id, "error": True},
                )
                return

            time.sleep(poll_interval)
            elapsed += poll_interval

        yield Document(
            page_content=f"Crawl timeout after {max_wait} seconds",
            metadata={"source": self.url, "crawl_id": crawl_id, "error": True},
        )

    def _map(self) -> Iterator[Document]:
        """Map a website and yield a document with all URLs."""
        import json

        import requests

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload: dict[str, Any] = {"url": self.url}

        if "search_query" in self.params:
            payload["search_query"] = self.params["search_query"]
        if "top_n" in self.params:
            payload["top_n"] = self.params["top_n"]
        if "include_urls" in self.params:
            payload["include_urls"] = self.params["include_urls"]
        if "exclude_urls" in self.params:
            payload["exclude_urls"] = self.params["exclude_urls"]

        response = requests.post(
            "https://api.olostep.com/v1/maps",
            headers=headers,
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        result = response.json()

        urls = result.get("urls", [])

        yield Document(
            page_content="\n".join(urls),
            metadata={
                "source": self.url,
                "total_urls": len(urls),
                "map_id": result.get("id", ""),
            },
        )
