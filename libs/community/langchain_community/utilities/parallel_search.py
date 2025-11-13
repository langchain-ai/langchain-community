"""Util that calls Parallel Search API.

In order to set this up, follow instructions at:
https://docs.parallel.ai/search/search-quickstart
"""

from typing import Any, Dict, List, Optional

import aiohttp
import requests
from langchain_core.utils import get_from_dict_or_env
from pydantic import BaseModel, ConfigDict, SecretStr, model_validator

PARALLEL_API_URL = "https://api.parallel.ai"


class ParallelSearchAPIWrapper(BaseModel):
    """Wrapper for Parallel Search API."""

    parallel_api_key: SecretStr
    """The API key to use for the Parallel search engine."""
    base_url: str = PARALLEL_API_URL
    """The base URL for the Parallel API."""
    api_version: str = "v1beta"
    """The API version to use."""

    model_config = ConfigDict(
        extra="forbid",
    )

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Any:
        """Validate that api key exists in environment."""
        parallel_api_key = get_from_dict_or_env(
            values, "parallel_api_key", "PARALLEL_API_KEY"
        )
        values["parallel_api_key"] = parallel_api_key
        return values

    def raw_results(
        self,
        objective: Optional[str] = None,
        search_queries: Optional[List[str]] = None,
        processor: str = "base",
        max_results: int = 10,
        max_chars_per_result: int = 6000,
        source_policy: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Get raw results from the Parallel Search API.

        Args:
            objective: Natural-language description of the web research goal.
            search_queries: Optional list of search queries to supplement the objective.
            processor: The processor to use ("base" or "pro"). Defaults to "base".
            max_results: Maximum number of search results to return (1-20).
                Defaults to 10.
            max_chars_per_result: Maximum characters per search result (100-30000).
                Defaults to 6000.
            source_policy: Optional source policy to include/exclude domains.

        Returns:
            Raw API response as a dictionary.
        """
        if not objective and not search_queries:
            raise ValueError("Either 'objective' or 'search_queries' must be provided.")

        payload: Dict[str, Any] = {
            "processor": processor,
            "max_results": max_results,
            "excerpts": {
                "max_chars_per_result": max_chars_per_result,
            },
        }

        if objective:
            payload["objective"] = objective
        if search_queries:
            payload["search_queries"] = search_queries
        if source_policy:
            payload["source_policy"] = source_policy

        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.parallel_api_key.get_secret_value(),
            "parallel-beta": "search-extract-2025-10-10",
        }

        url = f"{self.base_url}/{self.api_version}/search"
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()

    def results(
        self,
        objective: Optional[str] = None,
        search_queries: Optional[List[str]] = None,
        processor: str = "base",
        max_results: int = 10,
        max_chars_per_result: int = 6000,
        source_policy: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Run query through Parallel Search and return cleaned results.

        Args:
            objective: Natural-language description of the web research goal.
            search_queries: Optional list of search queries to supplement the objective.
            processor: The processor to use ("base" or "pro"). Defaults to "base".
            max_results: Maximum number of search results to return (1-20).
                Defaults to 10.
            max_chars_per_result: Maximum characters per search result (100-30000).
                Defaults to 6000.
            source_policy: Optional source policy to include/exclude domains.

        Returns:
            List of cleaned search results.
        """
        raw_results = self.raw_results(
            objective=objective,
            search_queries=search_queries,
            processor=processor,
            max_results=max_results,
            max_chars_per_result=max_chars_per_result,
            source_policy=source_policy,
        )
        return self.clean_results(raw_results.get("results", []))

    async def raw_results_async(
        self,
        objective: Optional[str] = None,
        search_queries: Optional[List[str]] = None,
        processor: str = "base",
        max_results: int = 10,
        max_chars_per_result: int = 6000,
        source_policy: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Get raw results from the Parallel Search API asynchronously.

        Args:
            objective: Natural-language description of the web research goal.
            search_queries: Optional list of search queries to supplement the objective.
            processor: The processor to use ("base" or "pro"). Defaults to "base".
            max_results: Maximum number of search results to return (1-20).
                Defaults to 10.
            max_chars_per_result: Maximum characters per search result (100-30000).
                Defaults to 6000.
            source_policy: Optional source policy to include/exclude domains.

        Returns:
            Raw API response as a dictionary.
        """
        if not objective and not search_queries:
            raise ValueError("Either 'objective' or 'search_queries' must be provided.")

        payload: Dict[str, Any] = {
            "processor": processor,
            "max_results": max_results,
            "excerpts": {
                "max_chars_per_result": max_chars_per_result,
            },
        }

        if objective:
            payload["objective"] = objective
        if search_queries:
            payload["search_queries"] = search_queries
        if source_policy:
            payload["source_policy"] = source_policy

        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.parallel_api_key.get_secret_value(),
            "parallel-beta": "search-extract-2025-10-10",
        }

        url = f"{self.base_url}/{self.api_version}/search"
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Error {response.status}: {error_text}")
                return await response.json()

    async def results_async(
        self,
        objective: Optional[str] = None,
        search_queries: Optional[List[str]] = None,
        processor: str = "base",
        max_results: int = 10,
        max_chars_per_result: int = 6000,
        source_policy: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Run query through Parallel Search and return cleaned results asynchronously.

        Args:
            objective: Natural-language description of the web research goal.
            search_queries: Optional list of search queries to supplement the objective.
            processor: The processor to use ("base" or "pro"). Defaults to "base".
            max_results: Maximum number of search results to return (1-20).
                Defaults to 10.
            max_chars_per_result: Maximum characters per search result (100-30000).
                Defaults to 6000.
            source_policy: Optional source policy to include/exclude domains.

        Returns:
            List of cleaned search results.
        """
        raw_results = await self.raw_results_async(
            objective=objective,
            search_queries=search_queries,
            processor=processor,
            max_results=max_results,
            max_chars_per_result=max_chars_per_result,
            source_policy=source_policy,
        )
        return self.clean_results(raw_results.get("results", []))

    def clean_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Clean results from Parallel Search API.

        Args:
            results: Raw results from the API.

        Returns:
            List of cleaned result dictionaries.
        """
        cleaned = []
        for result in results:
            cleaned_result: Dict[str, Any] = {
                "url": result.get("url"),
                "title": result.get("title"),
                "excerpts": result.get("excerpts", []),
            }
            if publish_date := result.get("publish_date"):
                cleaned_result["publish_date"] = publish_date
            cleaned.append(cleaned_result)
        return cleaned
