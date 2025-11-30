from typing import Any, Dict, Optional, List

import aiohttp
import requests
from langchain_core.utils import get_from_dict_or_env
from pydantic import BaseModel, ConfigDict, model_validator


class SearchApiAPIWrapper(BaseModel):
    """
    Wrapper around SearchApi API.

    To use, you should have the environment variable ``SEARCHAPI_API_KEY``
    set with your API key, or pass `searchapi_api_key`
    as a named parameter to the constructor.

    Example:
        .. code-block:: python

            from langchain_community.utilities import SearchApiAPIWrapper
            searchapi = SearchApiAPIWrapper()
    """

    # Use "google" engine by default.
    # Full list of supported ones can be found in https://www.searchapi.io docs
    engine: str = "google"
    searchapi_api_key: Optional[str] = None
    aiosession: Optional[aiohttp.ClientSession] = None

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Any:
        """Validate that API key exists in environment."""
        searchapi_api_key = get_from_dict_or_env(
            values, "searchapi_api_key", "SEARCHAPI_API_KEY"
        )
        values["searchapi_api_key"] = searchapi_api_key
        return values

    def run(self, query: str, **kwargs: Any) -> str:
        results = self.results(query, **kwargs)
        return self._result_as_string(results)

    async def arun(self, query: str, **kwargs: Any) -> str:
        results = await self.aresults(query, **kwargs)
        return self._result_as_string(results)

    def results(self, query: str, **kwargs: Any) -> dict:
        results = self._search_api_results(query, **kwargs)
        return results

    async def aresults(self, query: str, **kwargs: Any) -> dict:
        results = await self._async_search_api_results(query, **kwargs)
        return results

    def _prepare_request(self, query: str, **kwargs: Any) -> dict:
        return {
            "url": "https://www.searchapi.io/api/v1/search",
            "headers": {
                "Authorization": f"Bearer {self.searchapi_api_key}",
            },
            "params": {
                "engine": self.engine,
                "q": query,
                **{key: value for key, value in kwargs.items() if value is not None},
            },
        }

    def _search_api_results(self, query: str, **kwargs: Any) -> dict:
        request_details = self._prepare_request(query, **kwargs)
        response = requests.get(
            url=request_details["url"],
            params=request_details["params"],
            headers=request_details["headers"],
        )
        response.raise_for_status()
        return response.json()

    async def _async_search_api_results(self, query: str, **kwargs: Any) -> dict:
        """Use aiohttp to send request to SearchApi API and return results async."""
        request_details = self._prepare_request(query, **kwargs)
        if not self.aiosession:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url=request_details["url"],
                    headers=request_details["headers"],
                    params=request_details["params"],
                    raise_for_status=True,
                ) as response:
                    results = await response.json()
        else:
            async with self.aiosession.get(
                url=request_details["url"],
                headers=request_details["headers"],
                params=request_details["params"],
                raise_for_status=True,
            ) as response:
                results = await response.json()
        return results

    @staticmethod
    def _result_as_string(result: Dict[str, Any]) -> str:
        """
        Convert a SearchApi API response into a human-readable string.

        This implementation is intentionally defensive:
        - Tolerates missing keys and unexpected types.
        - Skips malformed items instead of raising exceptions.
        - Falls back to a default message when nothing useful is found.
        """
        default_message = "No good search result found"

        # Guard against completely invalid input
        if not isinstance(result, dict) or not result:
            return default_message

        # 1) Answer box: prefer explicit answer, then snippet
        answer_box = result.get("answer_box")
        if isinstance(answer_box, dict):
            answer = answer_box.get("answer")
            if isinstance(answer, str) and answer.strip():
                return answer

            snippet = answer_box.get("snippet")
            if isinstance(snippet, str) and snippet.strip():
                return snippet

        # 2) Knowledge graph description
        knowledge_graph = result.get("knowledge_graph")
        if isinstance(knowledge_graph, dict):
            description = knowledge_graph.get("description")
            if isinstance(description, str) and description.strip():
                return description

        # 3) Organic results snippets
        organic_results = result.get("organic_results")
        snippets: List[str] = []
        if isinstance(organic_results, list):
            for item in organic_results:
                if not isinstance(item, dict):
                    continue
                snippet = item.get("snippet")
                if isinstance(snippet, str) and snippet.strip():
                    snippets.append(snippet.strip())
        if snippets:
            return "\n".join(snippets)

        # 4) Jobs descriptions
        jobs_results = result.get("jobs")
        job_descriptions: List[str] = []
        if isinstance(jobs_results, list):
            for item in jobs_results:
                if not isinstance(item, dict):
                    continue
                description = item.get("description")
                if isinstance(description, str) and description.strip():
                    job_descriptions.append(description.strip())
        if job_descriptions:
            return "\n".join(job_descriptions)

        # 5) Videos (Title + Link)
        video_results = result.get("videos")
        video_lines: List[str] = []
        if isinstance(video_results, list):
            for item in video_results:
                if not isinstance(item, dict):
                    continue
                title = item.get("title")
                link = item.get("link")
                if isinstance(title, str) and isinstance(link, str):
                    video_lines.append(f'Title: "{title}" Link: {link}')
        if video_lines:
            return "\n".join(video_lines)

        # 6) Images (Title + original.link)
        image_results = result.get("images")
        image_lines: List[str] = []
        if isinstance(image_results, list):
            for item in image_results:
                if not isinstance(item, dict):
                    continue
                title = item.get("title")
                original = item.get("original")
                link: Optional[str] = None
                if isinstance(original, dict):
                    raw_link = original.get("link")
                    if isinstance(raw_link, str) and raw_link.strip():
                        link = raw_link.strip()
                if isinstance(title, str) and title.strip() and link:
                    image_lines.append(f'Title: "{title.strip()}" Link: {link}')
        if image_lines:
            return "\n".join(image_lines)

        # Fallback when nothing meaningful found
        return default_message
