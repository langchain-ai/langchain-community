"""Tool for the FeedCoop search API."""

from typing import Any, Dict, List, Literal, Optional, Type, Union

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_community.utilities.feedcoop_search import FeedCoopSearchAPIWrapper


class FeedCoopInput(BaseModel):
    """Input for the FeedCoop tool."""

    query: str = Field(description="search query to look up")


class FeedCoopSearchResults(BaseTool):
    """Tool that queries the FeedCoop Search API and gets back json.

    Instantiate:

        .. code-block:: bash

            pip install -U langchain-community
            export FEEDCOOP_API_KEY="your-api-key"

        .. code-block:: python

            from langchain_community.tools import FeedCoopSearchResults

            tool = FeedCoopSearchResults(
                count=10,
                need_content=False,
                need_url=False,
                include_domains=[],
                need_summary=False,
                time_range=None,
            )
    Invoke directly with args:

        .. code-block:: python

            tool.invoke({'query': 'who won the last french open'})

        .. code-block:: json

            {
                "title": "title",
                "site_name": "火山引擎"
                "url": "https://www.volcengine.com/...",
                "snippet": "snippet",
                "content": "content",
                "summary": "summary",
                "publish_time": "1970-01-01T08:00:00+08:00",
                "logo_url": "https://www.volcengine.com/...",
                "rank_score": 1,
                "auth_info_des": "正常权威",
                "auth_info_level": 2
            }
    """

    name: str = "feedcoop_search_results_json"
    description: str = (
        "A search engine that queries the FeedCoop Search API and gets back json. "
        "Useful for when you need to find information about current events or topics. "
        "Input should be a search query."
    )
    args_schema: Type[BaseModel] = FeedCoopInput

    count: int = 10
    """Max search results to return, default is 10"""
    search_type: Literal["web"] = Field(default="web")
    """
    feedcoop api supports `web` and `web_summary`.
    `web` is only supported for this tool.
    """
    need_content: bool = False
    """Whether to only return results with body text, default to false"""
    need_url: bool = False
    """Whether to only return the result of the original link, default to false"""
    include_domains: List[str] = []
    """A list of domains to specifically include in the search results."""
    need_summary: bool = False
    """Whether to include a summary of the content, default to false"""
    time_range: Optional[Literal["OneDay", "OneWeek", "OneMonth", "OneYear"] | str] = (
        None
    )
    """Specify the publication time for the search.
    The following enumeration values are unrestricted if left blank
    OneDay: Within 1 day
    OneWeek: Within 1 week
    OneMonth: Within 1 month
    OneYear: Within 1 year
    YYYY-MM-DD..YYYY-MM-DD: Content published within the period from date A (inclusive) to date B (inclusive), for example "2024-12-30..2025-12-30"
    """  # noqa: E501
    api_wrapper: FeedCoopSearchAPIWrapper = Field(
        default_factory=FeedCoopSearchAPIWrapper  # type: ignore[arg-type]
    )
    response_format: Literal["content_and_artifact"] = "content_and_artifact"

    def __init__(self, **kwargs: Any):
        """Create api_wrapper with feedcoop_api_key if provided."""
        if "feedcoop_api_key" in kwargs:
            kwargs["api_wrapper"] = FeedCoopSearchAPIWrapper(
                feedcoop_api_key=kwargs["feedcoop_api_key"]
            )
        super().__init__(**kwargs)

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> tuple[Union[str, list[Dict]], Dict]:
        """Use the tool."""
        try:
            raw_results = self.api_wrapper.raw_results(
                query,
                search_type=self.search_type,
                count=self.count,
                need_content=self.need_content,
                need_url=self.need_url,
                include_domains=self.include_domains,
                need_summary=self.need_summary,
                time_range=self.time_range,
            )
        except Exception as e:
            return repr(e), {}
        cleaned_results = self.api_wrapper.clean_results(
            raw_results["Result"]["WebResults"]
        )
        return cleaned_results, raw_results

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> tuple[Union[str, list[Dict]], Dict]:
        """Use the tool asynchronously."""
        try:
            raw_results = await self.api_wrapper.raw_results_async(
                query,
                search_type=self.search_type,
                count=self.count,
                need_content=self.need_content,
                need_url=self.need_url,
                include_domains=self.include_domains,
                need_summary=self.need_summary,
                time_range=self.time_range,
            )
        except Exception as e:
            return repr(e), {}
        cleaned_results = self.api_wrapper.clean_results(
            raw_results["Result"]["WebResults"]
        )
        return cleaned_results, raw_results
