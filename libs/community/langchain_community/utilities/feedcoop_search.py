"""Util that calls FeedCoop Search API.

In order to set this up, follow instructions at:
https://www.volcengine.com/docs/85508/1650263
"""

import json
from typing import Any, Dict, List, Literal, Optional

import aiohttp
import requests
from langchain_core.utils import get_from_dict_or_env
from pydantic import BaseModel, ConfigDict, SecretStr, model_validator

FEEDCOOP_API_URL = "https://open.feedcoopapi.com"


class FeedCoopSearchAPIWrapper(BaseModel):
    """Wrapper for FeedCoop Search API."""

    feedcoop_api_key: SecretStr

    model_config = ConfigDict(
        extra="forbid",
    )

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Any:
        """Validate that api key and endpoint exists in environment."""
        feedcoop_api_key = get_from_dict_or_env(
            values, "feedcoop_api_key", "FEEDCOOP_API_KEY"
        )
        values["feedcoop_api_key"] = feedcoop_api_key

        return values

    def raw_results(
        self,
        query: str,
        search_type: Literal["web", "web_summary"] = "web",
        count: int = 10,
        need_content: bool = False,
        need_url: bool = False,
        include_domains: List[str] = [],
        need_summary: bool = False,
        time_range: Optional[
            Literal["OneDay", "OneWeek", "OneMonth", "OneYear"] | str
        ] = None,
    ) -> Dict:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.feedcoop_api_key.get_secret_value()}",
        }
        params = {
            "Query": query,
            "SearchType": search_type,
            "Count": count,
            "Filter": {
                "NeedContent": need_content,
                "NeedUrl": need_url,
                "Sites": "|".join(include_domains),
            },
            "NeedSummary": need_summary,
            "TimeRange": time_range,
        }
        response = requests.post(
            f"{FEEDCOOP_API_URL}/search_api/web_search",
            headers=headers,
            json=params,
        )
        response.raise_for_status()
        response_json = response.json()
        # request failed
        if "Error" in response_json["ResponseMetadata"]:
            error_info = response_json["ResponseMetadata"]["Error"]
            error_code = error_info["CodeN"]
            error_msg = error_info["Message"]
            raise Exception(
                f"FeedCoop API failed, CodeN: {error_code}, Message: {error_msg}"
            )
        return response_json

    def results(
        self,
        query: str,
        search_type: Literal["web", "web_summary"] = "web",
        count: int = 10,
        need_content: bool = False,
        need_url: bool = False,
        include_domains: List[str] = [],
        need_summary: bool = False,
        time_range: Optional[
            Literal["OneDay", "OneWeek", "OneMonth", "OneYear"] | str
        ] = None,
    ) -> List[Dict]:
        """Run query through FeedCoop Search API and return metadata.

        Args:
            query (str): The query to search for.
            search_type (str, optional): Only support `web` in searching. Defaults to "web".
            count (int, optional): Max search results to return. Defaults to 10.
            need_content (bool, optional): Whether to only return results with body text. Defaults to False.
            need_url (bool, optional): Whether to only return the result of the original link. Defaults to False.
            include_domains (List[str], optional): A list of domains to specifically include in the search results. Defaults to [].
            need_summary (bool, optional): Whether to include a summary of the content. Defaults to False.
            time_range (Optional[Literal[&quot;OneDay&quot;, &quot;OneWeek&quot;, &quot;OneMonth&quot;, &quot;OneYear&quot;]  |  str], optional): Specify the publication time for the search. Defaults to None.
        """  # noqa: E501
        raw_results = self.raw_results(
            query,
            search_type=search_type,
            count=count,
            need_content=need_content,
            need_url=need_url,
            include_domains=include_domains,
            need_summary=need_summary,
            time_range=time_range,
        )
        return self.clean_results(raw_results["Result"]["WebResults"])

    async def raw_results_async(
        self,
        query: str,
        search_type: str = "web",
        count: int = 10,
        need_content: bool = False,
        need_url: bool = False,
        include_domains: List[str] = [],
        need_summary: bool = False,
        time_range: Optional[
            Literal["OneDay", "OneWeek", "OneMonth", "OneYear"] | str
        ] = None,
    ) -> Dict:
        """Get raw results from FeedCoop Search API asynchronously."""

        # Function to perform the API call
        async def fetch() -> str:
            authorization = f"Bearer {self.feedcoop_api_key.get_secret_value()}"
            headers = {
                "Content-Type": "application/json",
                "Authorization": authorization,
            }
            params = {
                "Query": query,
                "SearchType": search_type,
                "Count": count,
                "Filter": {
                    "NeedContent": need_content,
                    "NeedUrl": need_url,
                    "Sites": "|".join(include_domains),
                },
                "NeedSummary": need_summary,
                "TimeRange": time_range,
            }
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{FEEDCOOP_API_URL}/search_api/web_search",
                    headers=headers,
                    json=params,
                ) as res:
                    if res.status == 200:
                        data = await res.text()
                        return data
                    else:
                        raise Exception(
                            f"FeedCoop API request failed with status code {res.status}"
                        )  # noqa: E501

        result_json_str = await fetch()
        result_json = json.loads(result_json_str)
        if "Error" in result_json["ResponseMetadata"]:
            error_info = result_json["ResponseMetadata"]["Error"]
            error_code = error_info["CodeN"]
            error_msg = error_info["Message"]
            raise Exception(
                f"FeedCoop API failed, CodeN: {error_code}, Message: {error_msg}"
            )
        return result_json

    async def results_async(
        self,
        query: str,
        search_type: Literal["web", "web_summary"] = "web",
        count: int = 10,
        need_content: bool = False,
        need_url: bool = False,
        include_domains: List[str] = [],
        need_summary: bool = False,
        time_range: Optional[
            Literal["OneDay", "OneWeek", "OneMonth", "OneYear"] | str
        ] = None,
    ) -> List[Dict]:
        results_json = await self.raw_results_async(
            query,
            search_type=search_type,
            count=count,
            need_content=need_content,
            need_url=need_url,
            include_domains=include_domains,
            need_summary=need_summary,
            time_range=time_range,
        )
        return self.clean_results(results_json["Result"]["WebResults"])

    def clean_results(self, results: List[Dict]) -> List[Dict]:
        """Clean results from FeedCoop Search API."""
        clean_results = []
        for result in results:
            clean_result = {
                "title": result["Title"],
                "site_name": result["SiteName"],
                "url": result["Url"],
                "snippet": result["Snippet"],
                "summary": result["Summary"],
                "content": result["Content"],
                "publish_time": result["PublishTime"],
                "logo_url": result["LogoUrl"],
                "rank_score": result["RankScore"],
                "auth_info_des": result["AuthInfoDes"],
                "auth_info_level": result["AuthInfoLevel"],
            }
            clean_results.append(clean_result)
        return clean_results
