from __future__ import annotations

from typing import Type

import requests
from pydantic import BaseModel, Field

from langchain_core.tools import BaseTool


class MCPDiscoveryInput(BaseModel):
    """Input for MCP Discovery tool."""

    query: str = Field(
        ...,
        description="Task description used to discover relevant MCP servers.",
    )


class MCPDiscoveryTool(BaseTool):
    """Tool for discovering MCP servers dynamically via MCP Discovery API."""

    name: str = "mcp_discovery"
    description: str = (
        "Discover MCP servers dynamically based on a user query or task requirement."
    )

    api_url: str = "https://mcp-discovery-production.up.railway.app"

    args_schema: Type[BaseModel] = MCPDiscoveryInput

    def _run(self, query: str) -> dict:
        """Run MCP discovery search query."""
        response = requests.get(
            f"{self.api_url}/search",
            params={"q": query},
            timeout=10,
        )
        response.raise_for_status()
        return response.json()
