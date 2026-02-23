"""Anakin tools for LangChain agent workflows.

Provides three tools that wrap the Anakin API:

- ``AnakinScrapeTool``: Scrape a web page and return markdown content.
- ``AnakinSearchTool``: AI-powered web search with citations.
- ``AnakinAgenticSearchTool``: Multi-stage autonomous research.

API reference: https://anakin.io/llms-full.txt
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Type

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, SecretStr

from langchain_community.utilities.anakin import AnakinAPIWrapper

# ------------------------------------------------------------------
# Input schemas
# ------------------------------------------------------------------


class _ScrapeInput(BaseModel):
    url: str = Field(description="The URL of the web page to scrape.")
    use_browser: bool = Field(
        default=False,
        description="Use headless browser for JavaScript-rendered pages.",
    )


class _SearchInput(BaseModel):
    query: str = Field(description="The search query.")
    limit: int = Field(
        default=5,
        description="Maximum number of search results to return.",
    )


class _AgenticSearchInput(BaseModel):
    query: str = Field(
        description="The research question for deep, multi-stage investigation."
    )


# ------------------------------------------------------------------
# Tools
# ------------------------------------------------------------------


class AnakinScrapeTool(BaseTool):
    """Tool that scrapes a web page using the Anakin API.

    Returns the page content as clean markdown.

    Setup:
        Set the ``ANAKIN_API_KEY`` environment variable or pass ``api_key``:

        .. code-block:: bash

            pip install -U langchain-community
            export ANAKIN_API_KEY="your-api-key"

    Instantiate:
        .. code-block:: python

            from langchain_community.tools import AnakinScrapeTool

            tool = AnakinScrapeTool()

    Invoke:
        .. code-block:: python

            tool.invoke({"url": "https://example.com"})

        .. code-block:: python

            '# Example Domain\\n\\nThis domain is for use in illustrative ...'

    Invoke with browser rendering:
        .. code-block:: python

            tool.invoke({"url": "https://example.com", "use_browser": True})
    """  # noqa: E501

    name: str = "anakin_scrape"
    description: str = (
        "Scrape a web page and return its content as clean markdown. "
        "Useful for extracting content from any URL."
    )
    args_schema: Type[BaseModel] = _ScrapeInput
    api_wrapper: AnakinAPIWrapper = Field(default_factory=AnakinAPIWrapper)

    def __init__(self, api_key: Optional[str] = None, **kwargs: Any) -> None:
        if api_key and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = AnakinAPIWrapper(anakin_api_key=SecretStr(api_key))
        super().__init__(**kwargs)

    def _run(
        self,
        url: str,
        use_browser: bool = False,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Scrape the given URL and return markdown content."""
        result = self.api_wrapper.scrape(url, use_browser=use_browser)
        return result.get("markdown", "")

    async def _arun(
        self,
        url: str,
        use_browser: bool = False,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Async scrape the given URL and return markdown content."""
        result = await self.api_wrapper.ascrape(url, use_browser=use_browser)
        return result.get("markdown", "")


class AnakinSearchTool(BaseTool):
    """Tool that performs AI-powered web search using the Anakin API.

    Returns search results with titles, snippets, and source URLs.

    Setup:
        Set the ``ANAKIN_API_KEY`` environment variable or pass ``api_key``:

        .. code-block:: bash

            pip install -U langchain-community
            export ANAKIN_API_KEY="your-api-key"

    Instantiate:
        .. code-block:: python

            from langchain_community.tools import AnakinSearchTool

            tool = AnakinSearchTool()

    Invoke:
        .. code-block:: python

            tool.invoke({"query": "latest AI news"})

        .. code-block:: python

            '[1] AI Advances in 2026\\nNew breakthroughs...\\nSource: https://example.com'

    Invoke with custom limit:
        .. code-block:: python

            tool.invoke({"query": "Python tutorials", "limit": 10})
    """  # noqa: E501

    name: str = "anakin_search"
    description: str = (
        "Search the web using AI-powered search and return results with "
        "titles, snippets, and source URLs. Useful for finding current "
        "information on any topic."
    )
    args_schema: Type[BaseModel] = _SearchInput
    api_wrapper: AnakinAPIWrapper = Field(default_factory=AnakinAPIWrapper)

    def __init__(self, api_key: Optional[str] = None, **kwargs: Any) -> None:
        if api_key and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = AnakinAPIWrapper(anakin_api_key=SecretStr(api_key))
        super().__init__(**kwargs)

    def _run(
        self,
        query: str,
        limit: int = 5,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Search the web and return formatted results."""
        results = self.api_wrapper.search(query, limit=limit)
        return self._format_results(results)

    async def _arun(
        self,
        query: str,
        limit: int = 5,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Async search the web and return formatted results."""
        results = await self.api_wrapper.asearch(query, limit=limit)
        return self._format_results(results)

    @staticmethod
    def _format_results(results: List[Dict[str, Any]]) -> str:
        """Format search results into a readable string."""
        if not results:
            return "No results found."
        parts = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "No title")
            snippet = r.get("snippet", "")
            url = r.get("url", "")
            parts.append(f"[{i}] {title}\n{snippet}\nSource: {url}")
        return "\n\n".join(parts)


class AnakinAgenticSearchTool(BaseTool):
    """Tool that performs deep, multi-stage research using the Anakin API.

    Autonomously explores the web to produce a comprehensive research
    report.  Typically takes 1-5 minutes.

    Setup:
        Set the ``ANAKIN_API_KEY`` environment variable or pass ``api_key``:

        .. code-block:: bash

            pip install -U langchain-community
            export ANAKIN_API_KEY="your-api-key"

    Instantiate:
        .. code-block:: python

            from langchain_community.tools import AnakinAgenticSearchTool

            tool = AnakinAgenticSearchTool()

    Invoke:
        .. code-block:: python

            tool.invoke({"query": "compare React vs Vue in 2026"})

        .. code-block:: python

            'Based on analysis of multiple sources, React and Vue ...'
    """

    name: str = "anakin_agentic_search"
    description: str = (
        "Perform deep, multi-stage autonomous research on a topic. "
        "Returns a comprehensive research report. Best for complex "
        "questions requiring investigation across multiple sources. "
        "Takes 1-5 minutes to complete."
    )
    args_schema: Type[BaseModel] = _AgenticSearchInput
    api_wrapper: AnakinAPIWrapper = Field(default_factory=AnakinAPIWrapper)

    def __init__(self, api_key: Optional[str] = None, **kwargs: Any) -> None:
        if api_key and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = AnakinAPIWrapper(anakin_api_key=SecretStr(api_key))
        super().__init__(**kwargs)

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Run agentic research and return the summary report."""
        result = self.api_wrapper.agentic_search(query)
        generated = result.get("generatedJson", {})
        return generated.get("summary", "No summary available.")

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Async run agentic research and return the summary report."""
        result = await self.api_wrapper.aagentic_search(query)
        generated = result.get("generatedJson", {})
        return generated.get("summary", "No summary available.")
