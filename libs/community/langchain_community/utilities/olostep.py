"""Util that calls Olostep API.

The most reliable and cost-effective web search, scraping and crawling API for AI.
Build intelligent agents that can search, scrape, analyze, and structure data
from any website.

In order to set this up, follow instructions at:
https://docs.olostep.com
"""

import json
from typing import Any, Dict, List, Optional, Union

import aiohttp
import requests
from langchain_core.documents import Document
from langchain_core.utils import get_from_dict_or_env
from pydantic import BaseModel, ConfigDict, SecretStr, model_validator

OLOSTEP_API_URL = "https://api.olostep.com/v1"


class OlostepAPIWrapper(BaseModel):
    """Wrapper for Olostep API.

    The most reliable and cost-effective web search, scraping and crawling API for AI.
    Build intelligent agents that can search, scrape, analyze, and structure data
    from any website.

    Features:
        - Scrape: Extract content from any website with JavaScript rendering support
        - Crawl: Autonomously discover and scrape entire websites
        - Map: Extract all URLs from a website for site structure analysis
        - Batch: Process up to 10,000 URLs in parallel
        - Answers: AI-powered web search with natural language queries

    Setup:
        Get your API key from https://olostep.com and set the environment variable:

        .. code-block:: bash

            export OLOSTEP_API_KEY="your-api-key"

    Example:
        .. code-block:: python

            from langchain_community.utilities import OlostepAPIWrapper

            wrapper = OlostepAPIWrapper()
            result = wrapper.scrape("https://example.com")
    """

    olostep_api_key: SecretStr

    model_config = ConfigDict(
        extra="forbid",
    )

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Any:
        """Validate that api key exists in environment."""
        olostep_api_key = get_from_dict_or_env(
            values, "olostep_api_key", "OLOSTEP_API_KEY"
        )
        values["olostep_api_key"] = olostep_api_key
        return values

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for API requests."""
        return {
            "Authorization": f"Bearer {self.olostep_api_key.get_secret_value()}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    # -------------------------------------------------------------------------
    # Scrape
    # -------------------------------------------------------------------------

    def scrape(
        self,
        url: str,
        formats: Optional[List[str]] = None,
        country: Optional[str] = None,
        wait_before_scraping: int = 0,
    ) -> Dict[str, Any]:
        """Scrape a single URL.

        Args:
            url: Website URL to scrape. Must include protocol (http:// or https://).
            formats: Output formats. Options: "markdown", "html", "json", "text".
                Default: ["markdown"].
            country: ISO country code for location-specific content (e.g., "US", "GB").
            wait_before_scraping: Wait time in milliseconds before scraping.
                Range: 0-10000. Use 2000-5000 for dynamic sites.

        Returns:
            Dict containing scraped content and metadata.

        Example:
            .. code-block:: python

                wrapper = OlostepAPIWrapper()
                result = wrapper.scrape("https://example.com", formats=["markdown"])
        """
        payload: Dict[str, Any] = {
            "url_to_scrape": url,
            "formats": formats or ["markdown"],
        }

        if country:
            payload["country"] = country
        if wait_before_scraping:
            payload["wait_before_scraping"] = wait_before_scraping

        response = requests.post(
            f"{OLOSTEP_API_URL}/scrapes",
            headers=self._get_headers(),
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        return response.json()

    async def scrape_async(
        self,
        url: str,
        formats: Optional[List[str]] = None,
        country: Optional[str] = None,
        wait_before_scraping: int = 0,
    ) -> Dict[str, Any]:
        """Scrape a single URL asynchronously.

        Args:
            url: Website URL to scrape.
            formats: Output formats. Default: ["markdown"].
            country: ISO country code for location-specific content.
            wait_before_scraping: Wait time in milliseconds before scraping.

        Returns:
            Dict containing scraped content and metadata.
        """
        payload: Dict[str, Any] = {
            "url_to_scrape": url,
            "formats": formats or ["markdown"],
        }

        if country:
            payload["country"] = country
        if wait_before_scraping:
            payload["wait_before_scraping"] = wait_before_scraping

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{OLOSTEP_API_URL}/scrapes",
                headers=self._get_headers(),
                json=payload,
            ) as response:
                if response.status != 200:
                    raise Exception(f"HTTP error {response.status}: {response.reason}")
                return await response.json()

    def scrape_as_document(
        self,
        url: str,
        country: Optional[str] = None,
        wait_before_scraping: int = 0,
    ) -> Document:
        """Scrape a URL and return as a LangChain Document.

        Args:
            url: Website URL to scrape.
            country: ISO country code for location-specific content.
            wait_before_scraping: Wait time in milliseconds before scraping.

        Returns:
            Document with scraped content as page_content and metadata.
        """
        result = self.scrape(
            url=url,
            formats=["markdown"],
            country=country,
            wait_before_scraping=wait_before_scraping,
        )

        content_result = result.get("result", {})
        markdown_content = content_result.get("markdown_content", "")
        page_metadata = content_result.get("page_metadata", {})

        return Document(
            page_content=markdown_content,
            metadata={
                "source": url,
                "title": page_metadata.get("title", ""),
                "description": page_metadata.get("description", ""),
                "scrape_id": result.get("retrieve_id", ""),
            },
        )

    # -------------------------------------------------------------------------
    # Batch Scrape
    # -------------------------------------------------------------------------

    def batch_scrape(
        self,
        urls: List[str],
        formats: Optional[List[str]] = None,
        country: Optional[str] = None,
        wait_before_scraping: int = 0,
    ) -> Dict[str, Any]:
        """Scrape multiple URLs in parallel.

        Process up to 10,000 URLs simultaneously. Batch jobs typically complete
        in 5-8 minutes regardless of batch size.

        Args:
            urls: List of website URLs to scrape. Maximum 10,000 URLs per batch.
            formats: Output formats for all URLs. Default: ["markdown"].
            country: ISO country code for location-specific content.
            wait_before_scraping: Wait time in milliseconds before scraping each URL.

        Returns:
            Dict containing batch_id and status information.

        Example:
            .. code-block:: python

                wrapper = OlostepAPIWrapper()
                result = wrapper.batch_scrape([
                    "https://example1.com",
                    "https://example2.com",
                ])
        """
        batch_items = [
            {"url": url, "custom_id": f"url_{i}"} for i, url in enumerate(urls)
        ]

        payload: Dict[str, Any] = {"items": batch_items}

        if formats:
            payload["formats"] = formats
        if country:
            payload["country"] = country
        if wait_before_scraping:
            payload["wait_before_scraping"] = wait_before_scraping

        response = requests.post(
            f"{OLOSTEP_API_URL}/batches",
            headers=self._get_headers(),
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        return response.json()

    async def batch_scrape_async(
        self,
        urls: List[str],
        formats: Optional[List[str]] = None,
        country: Optional[str] = None,
        wait_before_scraping: int = 0,
    ) -> Dict[str, Any]:
        """Scrape multiple URLs in parallel asynchronously.

        Args:
            urls: List of website URLs to scrape.
            formats: Output formats for all URLs. Default: ["markdown"].
            country: ISO country code for location-specific content.
            wait_before_scraping: Wait time in milliseconds before scraping each URL.

        Returns:
            Dict containing batch_id and status information.
        """
        batch_items = [
            {"url": url, "custom_id": f"url_{i}"} for i, url in enumerate(urls)
        ]

        payload: Dict[str, Any] = {"items": batch_items}

        if formats:
            payload["formats"] = formats
        if country:
            payload["country"] = country
        if wait_before_scraping:
            payload["wait_before_scraping"] = wait_before_scraping

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{OLOSTEP_API_URL}/batches",
                headers=self._get_headers(),
                json=payload,
            ) as response:
                if response.status != 200:
                    raise Exception(f"HTTP error {response.status}: {response.reason}")
                return await response.json()

    # -------------------------------------------------------------------------
    # Answer
    # -------------------------------------------------------------------------

    def answer(
        self,
        task: str,
        json_schema: Optional[Union[Dict, str]] = None,
    ) -> Dict[str, Any]:
        """Search the web and get AI-powered answers.

        Ground your AI agents on real-world, up-to-date data. This searches the web,
        analyzes multiple sources, and returns structured answers in your desired
        format.

        Args:
            task: Question or research task to answer. Be specific and clear.
            json_schema: Optional JSON schema defining the desired output structure.
                Can be a dictionary or string. Use empty strings as placeholders.

        Returns:
            Dict containing answer, sources, and metadata.

        Example:
            .. code-block:: python

                wrapper = OlostepAPIWrapper()
                result = wrapper.answer(
                    task="Find information about Stripe",
                    json_schema={"company": "", "ceo": "", "founded_year": ""}
                )
        """
        payload: Dict[str, Any] = {"task": task}

        if json_schema:
            payload["json"] = json_schema

        response = requests.post(
            f"{OLOSTEP_API_URL}/answers",
            headers=self._get_headers(),
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        return response.json()

    async def answer_async(
        self,
        task: str,
        json_schema: Optional[Union[Dict, str]] = None,
    ) -> Dict[str, Any]:
        """Search the web and get AI-powered answers asynchronously.

        Args:
            task: Question or research task to answer.
            json_schema: Optional JSON schema for structured output.

        Returns:
            Dict containing answer, sources, and metadata.
        """
        payload: Dict[str, Any] = {"task": task}

        if json_schema:
            payload["json"] = json_schema

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{OLOSTEP_API_URL}/answers",
                headers=self._get_headers(),
                json=payload,
            ) as response:
                if response.status != 200:
                    raise Exception(f"HTTP error {response.status}: {response.reason}")
                return await response.json()

    # -------------------------------------------------------------------------
    # Map
    # -------------------------------------------------------------------------

    def map(
        self,
        url: str,
        search_query: Optional[str] = None,
        top_n: Optional[int] = None,
        include_urls: Optional[List[str]] = None,
        exclude_urls: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Extract all URLs from a website.

        Discover and map the complete structure of any website. Can discover
        up to ~100,000 URLs in a single call.

        Args:
            url: Base website URL to extract URLs from.
            search_query: Optional search query to filter URLs by relevance.
            top_n: Maximum number of URLs to return.
            include_urls: List of glob patterns to include (e.g., ["/blog/**"]).
            exclude_urls: List of glob patterns to exclude (e.g., ["/admin/**"]).

        Returns:
            Dict containing list of discovered URLs.

        Example:
            .. code-block:: python

                wrapper = OlostepAPIWrapper()
                result = wrapper.map(
                    "https://example.com",
                    include_urls=["/blog/**"],
                    top_n=100
                )
        """
        payload: Dict[str, Any] = {"url": url}

        if search_query:
            payload["search_query"] = search_query
        if top_n:
            payload["top_n"] = top_n
        if include_urls:
            payload["include_urls"] = include_urls
        if exclude_urls:
            payload["exclude_urls"] = exclude_urls

        response = requests.post(
            f"{OLOSTEP_API_URL}/maps",
            headers=self._get_headers(),
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        return response.json()

    async def map_async(
        self,
        url: str,
        search_query: Optional[str] = None,
        top_n: Optional[int] = None,
        include_urls: Optional[List[str]] = None,
        exclude_urls: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Extract all URLs from a website asynchronously.

        Args:
            url: Base website URL to extract URLs from.
            search_query: Optional search query to filter URLs.
            top_n: Maximum number of URLs to return.
            include_urls: List of glob patterns to include.
            exclude_urls: List of glob patterns to exclude.

        Returns:
            Dict containing list of discovered URLs.
        """
        payload: Dict[str, Any] = {"url": url}

        if search_query:
            payload["search_query"] = search_query
        if top_n:
            payload["top_n"] = top_n
        if include_urls:
            payload["include_urls"] = include_urls
        if exclude_urls:
            payload["exclude_urls"] = exclude_urls

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{OLOSTEP_API_URL}/maps",
                headers=self._get_headers(),
                json=payload,
            ) as response:
                if response.status != 200:
                    raise Exception(f"HTTP error {response.status}: {response.reason}")
                return await response.json()

    # -------------------------------------------------------------------------
    # Crawl
    # -------------------------------------------------------------------------

    def crawl(
        self,
        start_url: str,
        max_pages: int = 100,
        include_urls: Optional[List[str]] = None,
        exclude_urls: Optional[List[str]] = None,
        max_depth: Optional[int] = None,
        include_external: bool = False,
    ) -> Dict[str, Any]:
        """Autonomously crawl and scrape entire websites.

        The crawler intelligently follows links, discovers pages, and scrapes
        content automatically. Ideal for documentation sites, blogs, and
        knowledge bases.

        Args:
            start_url: Starting URL for the crawl.
            max_pages: Maximum number of pages to crawl. Default: 100.
            include_urls: List of glob patterns to include.
            exclude_urls: List of glob patterns to exclude.
            max_depth: Maximum link depth to crawl from start_url.
            include_external: Whether to follow external links. Default: False.

        Returns:
            Dict containing crawl_id and status information.

        Example:
            .. code-block:: python

                wrapper = OlostepAPIWrapper()
                result = wrapper.crawl(
                    "https://docs.example.com",
                    max_pages=200,
                    exclude_urls=["/admin/**"]
                )
        """
        payload: Dict[str, Any] = {
            "start_url": start_url,
            "max_pages": max_pages,
        }

        if include_urls:
            payload["include_urls"] = include_urls
        if exclude_urls:
            payload["exclude_urls"] = exclude_urls
        if max_depth is not None:
            payload["max_depth"] = max_depth
        if include_external:
            payload["include_external"] = include_external

        response = requests.post(
            f"{OLOSTEP_API_URL}/crawls",
            headers=self._get_headers(),
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        return response.json()

    async def crawl_async(
        self,
        start_url: str,
        max_pages: int = 100,
        include_urls: Optional[List[str]] = None,
        exclude_urls: Optional[List[str]] = None,
        max_depth: Optional[int] = None,
        include_external: bool = False,
    ) -> Dict[str, Any]:
        """Autonomously crawl and scrape entire websites asynchronously.

        Args:
            start_url: Starting URL for the crawl.
            max_pages: Maximum number of pages to crawl.
            include_urls: List of glob patterns to include.
            exclude_urls: List of glob patterns to exclude.
            max_depth: Maximum link depth to crawl.
            include_external: Whether to follow external links.

        Returns:
            Dict containing crawl_id and status information.
        """
        payload: Dict[str, Any] = {
            "start_url": start_url,
            "max_pages": max_pages,
        }

        if include_urls:
            payload["include_urls"] = include_urls
        if exclude_urls:
            payload["exclude_urls"] = exclude_urls
        if max_depth is not None:
            payload["max_depth"] = max_depth
        if include_external:
            payload["include_external"] = include_external

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{OLOSTEP_API_URL}/crawls",
                headers=self._get_headers(),
                json=payload,
            ) as response:
                if response.status != 200:
                    raise Exception(f"HTTP error {response.status}: {response.reason}")
                return await response.json()

    def get_crawl_status(self, crawl_id: str) -> Dict[str, Any]:
        """Get status of a crawl job.

        Args:
            crawl_id: Unique crawl ID returned from crawl().

        Returns:
            Dict containing crawl status and progress.
        """
        response = requests.get(
            f"{OLOSTEP_API_URL}/crawls/{crawl_id}",
            headers=self._get_headers(),
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    async def get_crawl_status_async(self, crawl_id: str) -> Dict[str, Any]:
        """Get status of a crawl job asynchronously.

        Args:
            crawl_id: Unique crawl ID returned from crawl().

        Returns:
            Dict containing crawl status and progress.
        """
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{OLOSTEP_API_URL}/crawls/{crawl_id}",
                headers=self._get_headers(),
            ) as response:
                if response.status != 200:
                    raise Exception(f"HTTP error {response.status}: {response.reason}")
                return await response.json()

    # -------------------------------------------------------------------------
    # Convenience methods for tool usage
    # -------------------------------------------------------------------------

    def scrape_run(
        self,
        url: str,
        format: str = "markdown",
        country: Optional[str] = None,
        wait_before_scraping: int = 0,
    ) -> str:
        """Scrape a URL and return content as a string.

        Convenience method for use with LangChain tools.

        Args:
            url: Website URL to scrape.
            format: Output format. Default: "markdown".
            country: ISO country code for location-specific content.
            wait_before_scraping: Wait time in milliseconds.

        Returns:
            JSON string containing scraped content.
        """
        result = self.scrape(
            url=url,
            formats=[format],
            country=country,
            wait_before_scraping=wait_before_scraping,
        )

        content_result = result.get("result", {})
        format_key = f"{format}_content"
        content = content_result.get(format_key, "")

        if isinstance(content, dict):
            content = json.dumps(content, indent=2)

        response = {
            "content": content,
            "url": url,
            "format": format,
            "scrape_id": result.get("retrieve_id", ""),
            "metadata": content_result.get("page_metadata", {}),
        }

        return json.dumps(response, indent=2)

    def answer_run(
        self,
        task: str,
        json_schema: Optional[Union[Dict, str]] = None,
    ) -> str:
        """Search the web and get AI-powered answer as a string.

        Convenience method for use with LangChain tools.

        Args:
            task: Question or research task.
            json_schema: Optional JSON schema for structured output.

        Returns:
            JSON string containing answer and sources.
        """
        result = self.answer(task=task, json_schema=json_schema)

        result_data = result.get("result", {})
        json_content = result_data.get("json_content", "")

        if isinstance(json_content, str) and json_content:
            try:
                json_content = json.loads(json_content)
            except json.JSONDecodeError:
                pass

        response = {
            "answer": json_content,
            "task": task,
            "sources": result_data.get("sources", []),
            "answer_id": result.get("id", ""),
        }

        return json.dumps(response, indent=2)

    def map_run(
        self,
        url: str,
        search_query: Optional[str] = None,
        top_n: Optional[int] = None,
        include_urls: Optional[List[str]] = None,
        exclude_urls: Optional[List[str]] = None,
    ) -> str:
        """Extract URLs from a website and return as a string.

        Convenience method for use with LangChain tools.

        Args:
            url: Base website URL.
            search_query: Optional search query to filter URLs.
            top_n: Maximum number of URLs to return.
            include_urls: Glob patterns to include.
            exclude_urls: Glob patterns to exclude.

        Returns:
            JSON string containing discovered URLs.
        """
        result = self.map(
            url=url,
            search_query=search_query,
            top_n=top_n,
            include_urls=include_urls,
            exclude_urls=exclude_urls,
        )

        urls = result.get("urls", [])

        response = {
            "map_id": result.get("id", result.get("map_id", "")),
            "url": url,
            "total_urls": len(urls),
            "urls": urls,
        }

        return json.dumps(response, indent=2)

    def crawl_run(
        self,
        start_url: str,
        max_pages: int = 100,
        include_urls: Optional[List[str]] = None,
        exclude_urls: Optional[List[str]] = None,
        max_depth: Optional[int] = None,
        include_external: bool = False,
    ) -> str:
        """Start a crawl and return status as a string.

        Convenience method for use with LangChain tools.

        Args:
            start_url: Starting URL for the crawl.
            max_pages: Maximum number of pages to crawl.
            include_urls: Glob patterns to include.
            exclude_urls: Glob patterns to exclude.
            max_depth: Maximum link depth.
            include_external: Whether to follow external links.

        Returns:
            JSON string containing crawl ID and status.
        """
        result = self.crawl(
            start_url=start_url,
            max_pages=max_pages,
            include_urls=include_urls,
            exclude_urls=exclude_urls,
            max_depth=max_depth,
            include_external=include_external,
        )

        response = {
            "crawl_id": result.get("id", ""),
            "status": result.get("status", "in_progress"),
            "start_url": start_url,
            "max_pages": max_pages,
            "pages_crawled": result.get("pages_crawled", 0),
        }

        return json.dumps(response, indent=2)

    def batch_scrape_run(
        self,
        urls: List[str],
        format: str = "markdown",
        country: Optional[str] = None,
        wait_before_scraping: int = 0,
    ) -> str:
        """Start a batch scrape and return status as a string.

        Convenience method for use with LangChain tools.

        Args:
            urls: List of URLs to scrape.
            format: Output format. Default: "markdown".
            country: ISO country code for location-specific content.
            wait_before_scraping: Wait time in milliseconds.

        Returns:
            JSON string containing batch ID and status.
        """
        result = self.batch_scrape(
            urls=urls,
            formats=[format],
            country=country,
            wait_before_scraping=wait_before_scraping,
        )

        response = {
            "batch_id": result.get("batch_id", result.get("id", "")),
            "status": result.get("status", "in_progress"),
            "total_urls": len(urls),
            "format": format,
            "urls": urls,
        }

        return json.dumps(response, indent=2)
