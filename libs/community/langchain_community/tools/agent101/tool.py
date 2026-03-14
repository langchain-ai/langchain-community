"""Tool for the Agent101 AI tools directory API."""

import json
from typing import Any, Dict, List, Optional, Type

import requests
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class Agent101SearchInput(BaseModel):
    """Input for the Agent101 search tool."""

    query: str = Field(description="search query to find AI tools")


class Agent101SearchRun(BaseTool):
    """Agent101 AI Tools Directory search tool.

    Agent101 is an open directory of 300+ AI tools across 15 categories
    including search, code, communication, data, AI services, browser,
    payments, files, social, people, realtime, apps, auth, memory, and maps.

    Setup:
        Install requests and langchain-community.

        .. code-block:: bash

            pip install -U requests langchain-community

    Instantiation:
        .. code-block:: python

            from langchain_community.tools import Agent101SearchRun

            tool = Agent101SearchRun()

    Invocation with args:
        .. code-block:: python

            tool.invoke("code generation")

        .. code-block:: python

            '[{"name": "GitHub Copilot", "category": "code", ...}]'

    Invocation with ToolCall:
        .. code-block:: python

            tool.invoke({"args": {"query": "code generation"}, "id": "1", "name": tool.name, "type": "tool_call"})
    """  # noqa: E501

    name: str = "agent101_search"
    description: str = (
        "A wrapper around the Agent101 AI Tools Directory. "
        "Useful for finding AI tools and services across categories like "
        "search, code, communication, data, AI services, browser, payments, "
        "files, social, people, realtime, apps, auth, memory, and maps. "
        "Input should be a search query describing the type of AI tool needed."
    )
    base_url: str = "https://agent101.ventify.ai"
    max_results: int = 5
    args_schema: Type[BaseModel] = Agent101SearchInput

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        try:
            response = requests.get(
                f"{self.base_url}/api/search",
                params={"q": query},
                timeout=10,
            )
            response.raise_for_status()
            results = response.json()

            if isinstance(results, list):
                results = results[: self.max_results]

            return json.dumps(results, indent=2)
        except requests.RequestException as e:
            return f"Error searching Agent101: {e}"


class Agent101SearchResults(BaseTool):
    """Tool that queries the Agent101 AI Tools Directory and returns structured results."""

    name: str = "agent101_results_json"
    description: str = (
        "A wrapper around the Agent101 AI Tools Directory. "
        "Returns structured JSON results of AI tools matching the query. "
        "Input should be a search query describing the type of AI tool needed."
    )
    base_url: str = "https://agent101.ventify.ai"
    max_results: int = 5
    args_schema: Type[BaseModel] = Agent101SearchInput
    response_format: str = "content_and_artifact"

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> tuple:
        """Use the tool."""
        try:
            response = requests.get(
                f"{self.base_url}/api/search",
                params={"q": query},
                timeout=10,
            )
            response.raise_for_status()
            results = response.json()

            if isinstance(results, list):
                truncated = results[: self.max_results]
            else:
                truncated = results

            content = json.dumps(truncated, indent=2)
            return content, results
        except requests.RequestException as e:
            return f"Error searching Agent101: {e}", []
