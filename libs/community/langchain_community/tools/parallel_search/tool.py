"""Tool for the Parallel Search API."""

from typing import Any, Dict, List, Literal, Optional, Tuple, Type

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_community.utilities.parallel_search import ParallelSearchAPIWrapper


class ParallelSearchInput(BaseModel):
    """Input for the Parallel Search tool."""

    objective: Optional[str] = Field(
        default=None,
        description="Natural-language description of the web research goal. "
        "Maximum 5000 characters.",
    )
    search_queries: Optional[List[str]] = Field(
        default=None,
        description="Optional list of search queries to supplement the objective. "
        "Maximum 200 characters per query. "
        "At least one of 'objective' or 'search_queries' must be provided.",
    )


class ParallelSearchRun(BaseTool):
    """Tool that queries the Parallel Search API and gets back text results.

    Setup:
        Install ``langchain-community`` and set environment variable
        ``PARALLEL_API_KEY``.

        .. code-block:: bash

            pip install -U langchain-community
            export PARALLEL_API_KEY="your-api-key"

    Instantiate:

        .. code-block:: python

            from langchain_community.tools import ParallelSearchRun

            tool = ParallelSearchRun(
                processor="base",
                max_results=10,
                max_chars_per_result=6000,
            )

    Invoke directly with args:

        .. code-block:: python

            tool.invoke({
                'objective': 'When was the United Nations established?',
                'search_queries': [
                    'Founding year UN', 'Year of founding United Nations'
                ]
            })
    """

    name: str = "parallel_search"
    description: str = (
        "A web search API optimized for AI agents. "
        "Useful for when you need to answer questions about current events or "
        "find information on the web. "
        "Input should be an objective (natural language description) and/or "
        "search queries (keywords)."
    )
    args_schema: Type[BaseModel] = ParallelSearchInput

    processor: str = "base"
    """The processor to use ("base" or "pro"). Defaults to "base"."""
    max_results: int = 10
    """Maximum number of search results to return (1-20). Defaults to 10."""
    max_chars_per_result: int = 6000
    """Maximum characters per search result (100-30000). Defaults to 6000."""
    source_policy: Optional[Dict[str, Any]] = None
    """Optional source policy to include/exclude domains."""
    api_wrapper: ParallelSearchAPIWrapper = Field(
        default_factory=ParallelSearchAPIWrapper
    )

    def __init__(self, **kwargs: Any) -> None:
        # Create api_wrapper with parallel_api_key if provided
        if "parallel_api_key" in kwargs:
            kwargs["api_wrapper"] = ParallelSearchAPIWrapper(
                parallel_api_key=kwargs["parallel_api_key"]
            )
        super().__init__(**kwargs)

    def _run(
        self,
        objective: Optional[str] = None,
        search_queries: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        try:
            results = self.api_wrapper.results(
                objective=objective,
                search_queries=search_queries,
                processor=self.processor,
                max_results=self.max_results,
                max_chars_per_result=self.max_chars_per_result,
                source_policy=self.source_policy,
            )
            return self._format_results(results)
        except Exception as e:
            return repr(e)

    async def _arun(
        self,
        objective: Optional[str] = None,
        search_queries: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        try:
            results = await self.api_wrapper.results_async(
                objective=objective,
                search_queries=search_queries,
                processor=self.processor,
                max_results=self.max_results,
                max_chars_per_result=self.max_chars_per_result,
                source_policy=self.source_policy,
            )
            return self._format_results(results)
        except Exception as e:
            return repr(e)

    def _format_results(self, results: List[Dict[str, Any]]) -> str:
        """Format search results as a string."""
        if not results:
            return "No results found."

        formatted = []
        for i, result in enumerate(results, 1):
            title = result.get("title", "No title")
            url = result.get("url", "")
            excerpts = result.get("excerpts", [])
            excerpt_text = "\n".join(excerpts) if isinstance(excerpts, list) else ""

            formatted.append(f"Result {i}: {title}\nURL: {url}")
            if excerpt_text:
                formatted.append(f"Content: {excerpt_text}")
            formatted.append("")

        return "\n".join(formatted)


class ParallelSearchResults(BaseTool):
    """Tool that queries the Parallel Search API and gets back structured results.

    Setup:
        Install ``langchain-community`` and set environment variable
        ``PARALLEL_API_KEY``.

        .. code-block:: bash

            pip install -U langchain-community
            export PARALLEL_API_KEY="your-api-key"

    Instantiate:

        .. code-block:: python

            from langchain_community.tools import ParallelSearchResults

            tool = ParallelSearchResults(
                processor="base",
                max_results=10,
                max_chars_per_result=6000,
            )

    Invoke directly with args:

        .. code-block:: python

            tool.invoke({
                'objective': 'When was the United Nations established?',
                'search_queries': ['Founding year UN']
            })
    """

    name: str = "parallel_search_results_json"
    description: str = (
        "A web search API optimized for AI agents. "
        "Useful for when you need to answer questions about current events or "
        "find information on the web. "
        "Input should be an objective (natural language description) and/or "
        "search queries (keywords). "
        "Output is a structured list of search results with URLs, titles, and excerpts."
    )
    args_schema: Type[BaseModel] = ParallelSearchInput

    processor: str = "base"
    """The processor to use ("base" or "pro"). Defaults to "base"."""
    max_results: int = 10
    """Maximum number of search results to return (1-20). Defaults to 10."""
    max_chars_per_result: int = 6000
    """Maximum characters per search result (100-30000). Defaults to 6000."""
    source_policy: Optional[Dict[str, Any]] = None
    """Optional source policy to include/exclude domains."""
    api_wrapper: ParallelSearchAPIWrapper = Field(
        default_factory=ParallelSearchAPIWrapper
    )
    response_format: Literal["content_and_artifact"] = "content_and_artifact"

    def __init__(self, **kwargs: Any) -> None:
        # Create api_wrapper with parallel_api_key if provided
        if "parallel_api_key" in kwargs:
            kwargs["api_wrapper"] = ParallelSearchAPIWrapper(
                parallel_api_key=kwargs["parallel_api_key"]
            )
        super().__init__(**kwargs)

    def _run(
        self,
        objective: Optional[str] = None,
        search_queries: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Use the tool."""
        try:
            results = self.api_wrapper.results(
                objective=objective,
                search_queries=search_queries,
                processor=self.processor,
                max_results=self.max_results,
                max_chars_per_result=self.max_chars_per_result,
                source_policy=self.source_policy,
            )
            return str(results), results
        except Exception as e:
            return repr(e), []

    async def _arun(
        self,
        objective: Optional[str] = None,
        search_queries: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Use the tool asynchronously."""
        try:
            results = await self.api_wrapper.results_async(
                objective=objective,
                search_queries=search_queries,
                processor=self.processor,
                max_results=self.max_results,
                max_chars_per_result=self.max_chars_per_result,
                source_policy=self.source_policy,
            )
            return str(results), results
        except Exception as e:
            return repr(e), []
