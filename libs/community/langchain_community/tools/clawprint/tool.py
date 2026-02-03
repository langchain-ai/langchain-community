"""ClawPrint tools — search, inspect, and hire agents via the ClawPrint API.

See ``ClawPrintToolkit`` for the easiest way to get started.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Type

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field

from langchain_community.tools.clawprint._client import (
    ClawPrintAPIError,
    ClawPrintClient,
)

__all__ = [
    "ClawPrintSearchTool",
    "ClawPrintGetAgentTool",
    "ClawPrintTrustTool",
    "ClawPrintDomainsTool",
    "ClawPrintRegisterTool",
    "ClawPrintHireAgentTool",
    "ClawPrintCheckExchangeTool",
    "ClawPrintToolkit",
]


# --- Input schemas ---


class ClawPrintSearchInput(BaseModel):
    """Input for searching the ClawPrint agent registry."""

    query: str = Field(
        description="Free-text search query describing the capability you need."
    )
    domain: Optional[str] = Field(
        default=None,
        description=(
            "Optional domain filter (e.g. 'code-review', 'data-analysis'). "
            "Use the clawprint_domains tool to discover valid domains."
        ),
    )
    min_trust: Optional[int] = Field(
        default=None,
        description="Minimum trust score between 0 and 100.",
        ge=0,
        le=100,
    )


class ClawPrintHandleInput(BaseModel):
    """Input that requires only an agent handle."""

    handle: str = Field(description="The unique handle of the agent (e.g. '@codebot').")


class ClawPrintRegisterInput(BaseModel):
    """Input for registering an agent on ClawPrint."""

    handle: str = Field(
        description=(
            "Unique lowercase handle for the agent (letters, digits, hyphens only). "
            "Example: 'my-code-reviewer'"
        ),
    )
    name: str = Field(description="Human-readable name for the agent.")
    description: str = Field(
        description="What this agent does — be specific about capabilities."
    )
    domains: List[str] = Field(
        description=(
            "Capability domains this agent covers "
            "(e.g. ['code-review', 'data-analysis']). "
            "Use the clawprint_domains tool to discover existing domains."
        ),
    )
    url: Optional[str] = Field(
        default=None,
        description="Optional homepage or API endpoint URL for the agent.",
    )


class ClawPrintHireInput(BaseModel):
    """Input for posting a hire/exchange request."""

    domains: List[str] = Field(
        description="List of capability domains the hired agent should cover."
    )
    task: str = Field(
        description="Plain-language description of the task to be performed."
    )
    requirements: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional structured requirements (budget, deadline, etc.).",
    )


class ClawPrintDomainsInput(BaseModel):
    """Input for listing domains (no parameters required)."""


class ClawPrintExchangeCheckInput(BaseModel):
    """Input for checking an exchange request status."""

    request_id: str = Field(description="The ID of the exchange request to check.")


def _json_result(data: Any) -> str:
    """Serialize API response to a JSON string the LLM can parse."""
    # indent=2 so the model can actually read nested structures
    return json.dumps(data, indent=2, default=str)


class ClawPrintSearchTool(BaseTool):
    """Tool that searches the ClawPrint agent registry.

    Returns a list of agent cards ranked by relevance and trust.
    Use this when you need to *discover* agents with specific capabilities.

    Setup:
        Requires the ``requests`` library. No API key needed.

        .. code-block:: bash

            pip install requests

    Key init args:
        client: ClawPrintClient instance for HTTP communication.

    Instantiate:
        .. code-block:: python

            from langchain_community.tools.clawprint import ClawPrintSearchTool
            from langchain_community.tools.clawprint._client import ClawPrintClient

            client = ClawPrintClient()
            tool = ClawPrintSearchTool(client=client)

    Invoke:
        .. code-block:: python

            tool.invoke({"query": "code review", "min_trust": 80})

    """

    name: str = "clawprint_search"
    description: str = (
        "Search the ClawPrint agent registry. Returns agents matching "
        "a free-text query, optionally filtered by domain and minimum trust score. "
        "Useful for discovering agents with specific capabilities."
    )
    args_schema: Type[BaseModel] = ClawPrintSearchInput
    client: ClawPrintClient = Field(exclude=True)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _run(
        self,
        query: str,
        domain: Optional[str] = None,
        min_trust: Optional[int] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Search the ClawPrint registry for agents.

        Args:
            query: Free-text search query.
            domain: Optional domain filter.
            min_trust: Minimum trust score (0-100).
            run_manager: Optional callback manager.

        Returns:
            JSON string with search results, or an error message string.
        """
        try:
            result = self.client.search_agents(
                query, domain=domain, min_score=min_trust
            )
            return _json_result(result)
        except (ClawPrintAPIError, ValueError) as e:
            return f"Error: {e}"

    async def _arun(
        self,
        query: str,
        domain: Optional[str] = None,
        min_trust: Optional[int] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Async search the ClawPrint registry for agents."""
        try:
            result = await self.client.asearch_agents(
                query, domain=domain, min_score=min_trust
            )
            return _json_result(result)
        except (ClawPrintAPIError, ValueError) as e:
            return f"Error: {e}"


class ClawPrintGetAgentTool(BaseTool):
    """Retrieve the full agent card for a handle.  No auth required."""

    name: str = "clawprint_get_agent"
    description: str = (
        "Get the full agent card for a specific agent by handle. "
        "Returns detailed info including capabilities, domains, trust score, "
        "and metadata. Use after search to inspect a particular agent."
    )
    args_schema: Type[BaseModel] = ClawPrintHandleInput
    client: ClawPrintClient = Field(exclude=True)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _run(
        self,
        handle: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        try:
            return _json_result(self.client.get_agent(handle))
        except (ClawPrintAPIError, ValueError) as e:
            return f"Error: {e}"

    async def _arun(
        self,
        handle: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        try:
            return _json_result(await self.client.aget_agent(handle))
        except (ClawPrintAPIError, ValueError) as e:
            return f"Error: {e}"


class ClawPrintTrustTool(BaseTool):
    """Check an agent's trust score (0-100).  No auth required.

    Scores reflect exchange completion history and verification status.
    """

    name: str = "clawprint_trust"
    description: str = (
        "Check the trust score of an agent by handle. "
        "Returns the score (0-100), breakdown factors, and verification status. "
        "Use to evaluate reliability before hiring."
    )
    args_schema: Type[BaseModel] = ClawPrintHandleInput
    client: ClawPrintClient = Field(exclude=True)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _run(
        self,
        handle: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        try:
            return _json_result(self.client.get_trust(handle))
        except (ClawPrintAPIError, ValueError) as e:
            return f"Error: {e}"

    async def _arun(
        self,
        handle: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        try:
            return _json_result(await self.client.aget_trust(handle))
        except (ClawPrintAPIError, ValueError) as e:
            return f"Error: {e}"


class ClawPrintDomainsTool(BaseTool):
    """List capability domains in the registry (e.g. code-review, data-analysis).

    Handy for discovering valid domain filters before calling search.
    """

    name: str = "clawprint_domains"
    description: str = (
        "List all available capability domains in the ClawPrint registry. "
        "Returns domain names and descriptions. "
        "Useful for discovering valid domain filters before searching."
    )
    args_schema: Type[BaseModel] = ClawPrintDomainsInput
    client: ClawPrintClient = Field(exclude=True)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _run(
        self,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        try:
            return _json_result(self.client.list_domains())
        except ClawPrintAPIError as e:
            return f"Error: {e}"

    async def _arun(
        self,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        try:
            return _json_result(await self.client.alist_domains())
        except ClawPrintAPIError as e:
            return f"Error: {e}"


class ClawPrintRegisterTool(BaseTool):
    """Register an agent so it becomes discoverable on ClawPrint.

    Returns an API key on success — the caller should persist it.
    """

    name: str = "clawprint_register"
    description: str = (
        "Register an agent on the ClawPrint discovery network so other agents "
        "can find and hire it. Provide a handle, name, description, and capability "
        "domains. Returns an API key — store it for future authenticated requests."
    )
    args_schema: Type[BaseModel] = ClawPrintRegisterInput
    client: ClawPrintClient = Field(exclude=True)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _run(
        self,
        handle: str,
        name: str,
        description: str,
        domains: List[str],
        url: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Register the agent.

        Args:
            handle: Unique lowercase handle.
            name: Human-readable name.
            description: Agent capabilities.
            domains: Capability domains.
            url: Optional homepage URL.
            run_manager: Optional callback manager.

        Returns:
            JSON string with agent card and API key, or error message.
        """
        try:
            result = self.client.register_agent(
                handle=handle,
                name=name,
                description=description,
                domains=domains,
                url=url,
            )
            return _json_result(result)
        except (ClawPrintAPIError, ValueError) as e:
            return f"Error: {e}"

    async def _arun(
        self,
        handle: str,
        name: str,
        description: str,
        domains: List[str],
        url: Optional[str] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Async register the agent."""
        try:
            result = await self.client.aregister_agent(
                handle=handle,
                name=name,
                description=description,
                domains=domains,
                url=url,
            )
            return _json_result(result)
        except (ClawPrintAPIError, ValueError) as e:
            return f"Error: {e}"


class ClawPrintHireAgentTool(BaseTool):
    """Post a brokered exchange request to hire an agent.  Requires API key.

    ClawPrint matches the request to available agents and returns a
    ``request_id`` you can poll with :class:`ClawPrintCheckExchangeTool`.

    Example::

        tool.invoke({
            "domains": ["code-review"],
            "task": "Review my FastAPI endpoint for security issues",
        })
    """

    name: str = "clawprint_hire"
    description: str = (
        "Post a brokered exchange request to hire an agent through ClawPrint. "
        "Specify capability domains needed, a task description, and optional "
        "requirements. Returns a request ID to track. Requires API key."
    )
    args_schema: Type[BaseModel] = ClawPrintHireInput
    client: ClawPrintClient = Field(exclude=True)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _run(
        self,
        domains: List[str],
        task: str,
        requirements: Optional[Dict[str, Any]] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Post a hire request.

        Args:
            domains: Required capability domains.
            task: Task description.
            requirements: Optional structured requirements.
            run_manager: Optional callback manager.

        Returns:
            JSON string with request_id and status, or error message.
        """
        try:
            result = self.client.create_exchange_request(
                domains=domains, task=task, requirements=requirements
            )
            return _json_result(result)
        except ClawPrintAPIError as e:
            return f"Error: {e}"

    async def _arun(
        self,
        domains: List[str],
        task: str,
        requirements: Optional[Dict[str, Any]] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Async post a hire request."""
        try:
            result = await self.client.acreate_exchange_request(
                domains=domains, task=task, requirements=requirements
            )
            return _json_result(result)
        except ClawPrintAPIError as e:
            return f"Error: {e}"


class ClawPrintCheckExchangeTool(BaseTool):
    """Poll an exchange request for status updates.  Requires API key."""

    name: str = "clawprint_check_exchange"
    description: str = (
        "Check the status of an exchange request by ID. "
        "Returns current state (pending/matched/in-progress/completed/failed) "
        "and matched agent info. Requires API key."
    )
    args_schema: Type[BaseModel] = ClawPrintExchangeCheckInput
    client: ClawPrintClient = Field(exclude=True)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _run(
        self,
        request_id: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        try:
            return _json_result(self.client.get_exchange_request(request_id))
        except ClawPrintAPIError as e:
            return f"Error: {e}"

    async def _arun(
        self,
        request_id: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        try:
            return _json_result(await self.client.aget_exchange_request(request_id))
        except ClawPrintAPIError as e:
            return f"Error: {e}"


class ClawPrintToolkit:
    """Bundle all seven ClawPrint tools with a shared HTTP client.

    ::

        toolkit = ClawPrintToolkit(api_key="cp_live_...")
        tools = toolkit.get_tools()

    ``api_key`` falls back to ``CLAWPRINT_API_KEY`` env var.
    Only needed for hire/exchange — read-only tools work without it.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://clawprint.io",
        timeout: int = 30,
    ) -> None:
        self._client = ClawPrintClient(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )

    def get_tools(self) -> List[BaseTool]:
        """Return all seven ClawPrint tools wired to a shared HTTP client.

        Tools that require authentication (hire, check exchange) will
        return an error string at call time if no API key was provided.

        Returns:
            List of seven BaseTool instances.
        """
        c = self._client
        return [
            ClawPrintRegisterTool(client=c),
            ClawPrintSearchTool(client=c),
            ClawPrintGetAgentTool(client=c),
            ClawPrintTrustTool(client=c),
            ClawPrintDomainsTool(client=c),
            ClawPrintHireAgentTool(client=c),
            ClawPrintCheckExchangeTool(client=c),
        ]

    def get_tool(self, name: str) -> BaseTool:
        """Look up a specific tool by name.

        Args:
            name: Tool name (e.g. ``"clawprint_search"``).

        Returns:
            The matching BaseTool instance.

        Raises:
            KeyError: If no tool matches the given name.
        """
        tools = self.get_tools()
        for tool in tools:
            if tool.name == name:
                return tool
        available = [t.name for t in tools]
        raise KeyError(f"No tool named '{name}'. Available: {available}")
