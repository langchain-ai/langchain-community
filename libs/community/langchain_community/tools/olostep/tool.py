"""Tools for the Olostep API.

The most reliable and cost-effective web search, scraping and crawling API for AI.
Build intelligent agents that can search, scrape, analyze, and structure data
from any website.
"""

from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_community.utilities.olostep import OlostepAPIWrapper


# -----------------------------------------------------------------------------
# Input Schemas
# -----------------------------------------------------------------------------


class OlostepScrapeInput(BaseModel):
    """Input for the Olostep Scrape tool."""

    url: str = Field(
        description="Website URL to scrape. Must include protocol (http:// or https://)."
    )
    format: Literal["markdown", "html", "json", "text"] = Field(
        default="markdown",
        description="Output format. Options: markdown (default, best for LLMs), html, json, text.",
    )
    country: Optional[str] = Field(
        default=None,
        description="ISO country code for location-specific content (e.g., US, GB, CA).",
    )
    wait_before_scraping: int = Field(
        default=0,
        description="Wait time in milliseconds before scraping. Range: 0-10000. Use 2000-5000 for dynamic sites.",
    )


class OlostepAnswersInput(BaseModel):
    """Input for the Olostep Answers tool."""

    task: str = Field(
        description="Question or research task to answer. Be specific and clear."
    )
    json_schema: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional JSON schema defining desired output structure. Use empty strings as placeholders.",
    )


class OlostepMapInput(BaseModel):
    """Input for the Olostep Map tool."""

    url: str = Field(
        description="Base website URL to extract URLs from. Must include protocol."
    )
    search_query: Optional[str] = Field(
        default=None,
        description="Optional search query to filter URLs by relevance.",
    )
    top_n: Optional[int] = Field(
        default=None,
        description="Maximum number of URLs to return.",
    )
    include_urls: Optional[List[str]] = Field(
        default=None,
        description='Glob patterns to include (e.g., ["/blog/**"]).',
    )
    exclude_urls: Optional[List[str]] = Field(
        default=None,
        description='Glob patterns to exclude (e.g., ["/admin/**"]).',
    )


class OlostepCrawlInput(BaseModel):
    """Input for the Olostep Crawl tool."""

    start_url: str = Field(
        description="Starting URL for the crawl. Must include protocol."
    )
    max_pages: int = Field(
        default=100,
        description="Maximum number of pages to crawl.",
    )
    include_urls: Optional[List[str]] = Field(
        default=None,
        description="Glob patterns to include.",
    )
    exclude_urls: Optional[List[str]] = Field(
        default=None,
        description="Glob patterns to exclude.",
    )
    max_depth: Optional[int] = Field(
        default=None,
        description="Maximum link depth to crawl from start_url.",
    )
    include_external: bool = Field(
        default=False,
        description="Whether to follow external links.",
    )


# -----------------------------------------------------------------------------
# Tools
# -----------------------------------------------------------------------------


class OlostepScrape(BaseTool):
    """Tool for scraping web content using the Olostep API.

    The most reliable and cost-effective web search, scraping and crawling API for AI.
    Build intelligent agents that can search, scrape, analyze, and structure data
    from any website.

    Setup:
        Install ``langchain-community`` and set environment variable ``OLOSTEP_API_KEY``.

        .. code-block:: bash

            pip install -U langchain-community
            export OLOSTEP_API_KEY="your-api-key"

    Instantiate:
        .. code-block:: python

            from langchain_community.tools import OlostepScrape

            tool = OlostepScrape()

    Invoke directly with args:
        .. code-block:: python

            tool.invoke({"url": "https://example.com"})

    Invoke with tool call:
        .. code-block:: python

            tool.invoke({
                "args": {"url": "https://example.com", "format": "markdown"},
                "type": "tool_call",
                "id": "1",
                "name": "olostep_scrape"
            })

    .. versionadded:: 0.3.0
    """

    name: str = "olostep_scrape"
    description: str = (
        "Scrape content from any website using Olostep's web scraping API. "
        "Extract clean, LLM-ready content in multiple formats (markdown, html, json, text). "
        "Handles JavaScript rendering and anti-bot measures. "
        "Perfect for research agents, content analysis, and building RAG systems."
    )
    args_schema: Type[BaseModel] = OlostepScrapeInput

    api_wrapper: OlostepAPIWrapper = Field(default_factory=OlostepAPIWrapper)  # type: ignore[arg-type]
    response_format: Literal["content_and_artifact"] = "content_and_artifact"

    def __init__(self, **kwargs: Any) -> None:
        if "olostep_api_key" in kwargs:
            kwargs["api_wrapper"] = OlostepAPIWrapper(
                olostep_api_key=kwargs["olostep_api_key"]
            )
        super().__init__(**kwargs)

    def _run(
        self,
        url: str,
        format: str = "markdown",
        country: Optional[str] = None,
        wait_before_scraping: int = 0,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Tuple[str, Dict]:
        """Use the tool."""
        try:
            result = self.api_wrapper.scrape(
                url=url,
                formats=[format],
                country=country,
                wait_before_scraping=wait_before_scraping,
            )
            content_result = result.get("result", {})
            format_key = f"{format}_content"
            content = content_result.get(format_key, "")

            if isinstance(content, dict):
                import json

                content = json.dumps(content, indent=2)

            return content, result
        except Exception as e:
            return repr(e), {}

    async def _arun(
        self,
        url: str,
        format: str = "markdown",
        country: Optional[str] = None,
        wait_before_scraping: int = 0,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Tuple[str, Dict]:
        """Use the tool asynchronously."""
        try:
            result = await self.api_wrapper.scrape_async(
                url=url,
                formats=[format],
                country=country,
                wait_before_scraping=wait_before_scraping,
            )
            content_result = result.get("result", {})
            format_key = f"{format}_content"
            content = content_result.get(format_key, "")

            if isinstance(content, dict):
                import json

                content = json.dumps(content, indent=2)

            return content, result
        except Exception as e:
            return repr(e), {}


class OlostepAnswers(BaseTool):
    """Tool for AI-powered web search using the Olostep Answers API.

    Search the web and get AI-powered answers with structured output and sources.
    Ground your AI agents on real-world, up-to-date data.

    Setup:
        Install ``langchain-community`` and set environment variable ``OLOSTEP_API_KEY``.

        .. code-block:: bash

            pip install -U langchain-community
            export OLOSTEP_API_KEY="your-api-key"

    Instantiate:
        .. code-block:: python

            from langchain_community.tools import OlostepAnswers

            tool = OlostepAnswers()

    Invoke directly with args:
        .. code-block:: python

            # Simple question
            tool.invoke({"task": "What is the latest funding round for Anthropic?"})

            # With structured output
            tool.invoke({
                "task": "Find information about Stripe",
                "json_schema": {"company": "", "ceo": "", "founded_year": ""}
            })

    .. versionadded:: 0.3.0
    """

    name: str = "olostep_answers"
    description: str = (
        "Search the web and get AI-powered answers with structured output and sources. "
        "Ground your AI agents on real-world, up-to-date data. "
        "Perfect for research agents, data enrichment, market intelligence, "
        "competitive analysis, and fact-checking. "
        "Optionally provide a JSON schema for structured data extraction."
    )
    args_schema: Type[BaseModel] = OlostepAnswersInput

    api_wrapper: OlostepAPIWrapper = Field(default_factory=OlostepAPIWrapper)  # type: ignore[arg-type]
    response_format: Literal["content_and_artifact"] = "content_and_artifact"

    def __init__(self, **kwargs: Any) -> None:
        if "olostep_api_key" in kwargs:
            kwargs["api_wrapper"] = OlostepAPIWrapper(
                olostep_api_key=kwargs["olostep_api_key"]
            )
        super().__init__(**kwargs)

    def _run(
        self,
        task: str,
        json_schema: Optional[Dict[str, Any]] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Tuple[str, Dict]:
        """Use the tool."""
        import json as json_module

        try:
            result = self.api_wrapper.answer(task=task, json_schema=json_schema)
            result_data = result.get("result", {})
            json_content = result_data.get("json_content", "")

            if isinstance(json_content, str) and json_content:
                try:
                    json_content = json_module.loads(json_content)
                except json_module.JSONDecodeError:
                    pass

            response = {
                "answer": json_content,
                "sources": result_data.get("sources", []),
            }

            return json_module.dumps(response, indent=2), result
        except Exception as e:
            return repr(e), {}

    async def _arun(
        self,
        task: str,
        json_schema: Optional[Dict[str, Any]] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Tuple[str, Dict]:
        """Use the tool asynchronously."""
        import json as json_module

        try:
            result = await self.api_wrapper.answer_async(
                task=task, json_schema=json_schema
            )
            result_data = result.get("result", {})
            json_content = result_data.get("json_content", "")

            if isinstance(json_content, str) and json_content:
                try:
                    json_content = json_module.loads(json_content)
                except json_module.JSONDecodeError:
                    pass

            response = {
                "answer": json_content,
                "sources": result_data.get("sources", []),
            }

            return json_module.dumps(response, indent=2), result
        except Exception as e:
            return repr(e), {}


class OlostepMap(BaseTool):
    """Tool for extracting all URLs from a website using the Olostep Map API.

    Discover and map the complete structure of any website. Can discover
    up to ~100,000 URLs in a single call.

    Setup:
        Install ``langchain-community`` and set environment variable ``OLOSTEP_API_KEY``.

        .. code-block:: bash

            pip install -U langchain-community
            export OLOSTEP_API_KEY="your-api-key"

    Instantiate:
        .. code-block:: python

            from langchain_community.tools import OlostepMap

            tool = OlostepMap()

    Invoke directly with args:
        .. code-block:: python

            # Extract all URLs
            tool.invoke({"url": "https://example.com"})

            # Filter to blog posts only
            tool.invoke({
                "url": "https://example.com",
                "include_urls": ["/blog/**"],
                "top_n": 100
            })

    .. versionadded:: 0.3.0
    """

    name: str = "olostep_map"
    description: str = (
        "Extract all URLs from a website for sitemap generation and content discovery. "
        "Discover and map the complete structure of any website. "
        "Can discover up to ~100,000 URLs in a single call. "
        "Perfect for SEO audits, preparing batch scraping jobs, and website analysis."
    )
    args_schema: Type[BaseModel] = OlostepMapInput

    api_wrapper: OlostepAPIWrapper = Field(default_factory=OlostepAPIWrapper)  # type: ignore[arg-type]
    response_format: Literal["content_and_artifact"] = "content_and_artifact"

    def __init__(self, **kwargs: Any) -> None:
        if "olostep_api_key" in kwargs:
            kwargs["api_wrapper"] = OlostepAPIWrapper(
                olostep_api_key=kwargs["olostep_api_key"]
            )
        super().__init__(**kwargs)

    def _run(
        self,
        url: str,
        search_query: Optional[str] = None,
        top_n: Optional[int] = None,
        include_urls: Optional[List[str]] = None,
        exclude_urls: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Tuple[str, Dict]:
        """Use the tool."""
        import json

        try:
            result = self.api_wrapper.map(
                url=url,
                search_query=search_query,
                top_n=top_n,
                include_urls=include_urls,
                exclude_urls=exclude_urls,
            )
            urls = result.get("urls", [])

            response = {
                "url": url,
                "total_urls": len(urls),
                "urls": urls,
            }

            return json.dumps(response, indent=2), result
        except Exception as e:
            return repr(e), {}

    async def _arun(
        self,
        url: str,
        search_query: Optional[str] = None,
        top_n: Optional[int] = None,
        include_urls: Optional[List[str]] = None,
        exclude_urls: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Tuple[str, Dict]:
        """Use the tool asynchronously."""
        import json

        try:
            result = await self.api_wrapper.map_async(
                url=url,
                search_query=search_query,
                top_n=top_n,
                include_urls=include_urls,
                exclude_urls=exclude_urls,
            )
            urls = result.get("urls", [])

            response = {
                "url": url,
                "total_urls": len(urls),
                "urls": urls,
            }

            return json.dumps(response, indent=2), result
        except Exception as e:
            return repr(e), {}


class OlostepCrawl(BaseTool):
    """Tool for crawling entire websites using the Olostep Crawl API.

    Autonomously crawl and scrape entire websites by following links.
    The crawler intelligently follows links, discovers pages, and scrapes
    content automatically.

    Setup:
        Install ``langchain-community`` and set environment variable ``OLOSTEP_API_KEY``.

        .. code-block:: bash

            pip install -U langchain-community
            export OLOSTEP_API_KEY="your-api-key"

    Instantiate:
        .. code-block:: python

            from langchain_community.tools import OlostepCrawl

            tool = OlostepCrawl()

    Invoke directly with args:
        .. code-block:: python

            # Crawl documentation site
            tool.invoke({
                "start_url": "https://docs.example.com",
                "max_pages": 200
            })

            # Crawl with filters
            tool.invoke({
                "start_url": "https://example.com",
                "max_pages": 100,
                "include_urls": ["/docs/**"],
                "exclude_urls": ["/admin/**"]
            })

    .. versionadded:: 0.3.0
    """

    name: str = "olostep_crawl"
    description: str = (
        "Autonomously crawl and scrape entire websites by following links. "
        "The crawler intelligently discovers pages and scrapes content automatically. "
        "Ideal for documentation sites, blogs, knowledge bases, and building datasets. "
        "Returns a crawl_id to track progress."
    )
    args_schema: Type[BaseModel] = OlostepCrawlInput

    api_wrapper: OlostepAPIWrapper = Field(default_factory=OlostepAPIWrapper)  # type: ignore[arg-type]
    response_format: Literal["content_and_artifact"] = "content_and_artifact"

    def __init__(self, **kwargs: Any) -> None:
        if "olostep_api_key" in kwargs:
            kwargs["api_wrapper"] = OlostepAPIWrapper(
                olostep_api_key=kwargs["olostep_api_key"]
            )
        super().__init__(**kwargs)

    def _run(
        self,
        start_url: str,
        max_pages: int = 100,
        include_urls: Optional[List[str]] = None,
        exclude_urls: Optional[List[str]] = None,
        max_depth: Optional[int] = None,
        include_external: bool = False,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Tuple[str, Dict]:
        """Use the tool."""
        import json

        try:
            result = self.api_wrapper.crawl(
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
            }

            return json.dumps(response, indent=2), result
        except Exception as e:
            return repr(e), {}

    async def _arun(
        self,
        start_url: str,
        max_pages: int = 100,
        include_urls: Optional[List[str]] = None,
        exclude_urls: Optional[List[str]] = None,
        max_depth: Optional[int] = None,
        include_external: bool = False,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Tuple[str, Dict]:
        """Use the tool asynchronously."""
        import json

        try:
            result = await self.api_wrapper.crawl_async(
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
            }

            return json.dumps(response, indent=2), result
        except Exception as e:
            return repr(e), {}
