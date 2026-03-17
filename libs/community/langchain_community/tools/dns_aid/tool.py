"""Tools for DNS-AID agent discovery and publishing.

DNS-AID (DNS-based Agent Identification and Discovery) enables AI agents
to discover each other via DNS using SVCB and TXT records, per
IETF draft-mozleywilliams-dnsop-dnsaid-01.

These tools wrap the ``dns-aid`` Python package. Install with:

.. code-block:: bash

    pip install dns-aid
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Optional, Type

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


def _import_dns_aid() -> Any:
    """Lazy import dns_aid, raising a clear error if not installed."""
    try:
        import dns_aid

        return dns_aid
    except ImportError:
        raise ImportError(
            "The dns-aid package is required to use DNS-AID tools. "
            "Install it with: pip install dns-aid"
        )


def _import_create_backend() -> Any:
    """Lazy import the dns_aid backend factory."""
    try:
        from dns_aid.backends import create_backend

        return create_backend
    except ImportError:
        raise ImportError(
            "The dns-aid package is required to use DNS-AID tools. "
            "Install it with: pip install dns-aid"
        )


def _run_async(coro: Any) -> Any:
    """Run an async coroutine from a sync context.

    Handles the case where an event loop is already running
    (e.g., Jupyter, FastAPI) by running in a separate thread.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    else:
        return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Input schemas
# ---------------------------------------------------------------------------


class DnsAidPublishInput(BaseModel):
    """Input schema for DnsAidPublishTool."""

    name: str = Field(
        ..., description="Agent identifier in DNS label format (e.g. 'my-agent')"
    )
    domain: str = Field(
        ..., description="Domain to publish under (e.g. 'agents.example.com')"
    )
    protocol: str = Field(
        default="mcp", description="Protocol: 'a2a', 'mcp', or 'https'"
    )
    endpoint: str = Field(
        ..., description="Hostname where the agent is reachable"
    )
    port: int = Field(default=443, description="Port number")
    capabilities: Optional[list[str]] = Field(
        default=None, description="List of agent capabilities"
    )
    version: str = Field(default="1.0.0", description="Agent version")
    description: Optional[str] = Field(
        default=None, description="Human-readable description of the agent"
    )
    ttl: int = Field(default=3600, description="DNS time-to-live in seconds")


class DnsAidDiscoverInput(BaseModel):
    """Input schema for DnsAidDiscoverTool."""

    domain: str = Field(
        ..., description="Domain to search for agents (e.g. 'agents.example.com')"
    )
    protocol: Optional[str] = Field(
        default=None,
        description="Filter by protocol: 'a2a', 'mcp', 'https', or None for all",
    )
    name: Optional[str] = Field(
        default=None, description="Filter by specific agent name"
    )
    require_dnssec: bool = Field(
        default=False, description="Require DNSSEC-validated responses"
    )


class DnsAidUnpublishInput(BaseModel):
    """Input schema for DnsAidUnpublishTool."""

    name: str = Field(..., description="Agent identifier to remove")
    domain: str = Field(..., description="Domain the agent is published under")
    protocol: str = Field(
        default="mcp", description="Protocol: 'a2a', 'mcp', or 'https'"
    )


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


class DnsAidPublishTool(BaseTool):
    """Publish an AI agent to DNS using the DNS-AID protocol.

    Creates SVCB and TXT records so the agent becomes discoverable
    by other agents and orchestrators querying DNS.

    Setup:
        Install the ``dns-aid`` package and configure a DNS backend:

        .. code-block:: bash

            pip install dns-aid
            export DNS_AID_BACKEND=route53  # or cloudflare, ddns, etc.

    Instantiate:
        .. code-block:: python

            from langchain_community.tools.dns_aid import DnsAidPublishTool

            tool = DnsAidPublishTool(backend_name="route53")

    Invoke:
        .. code-block:: python

            tool.invoke({
                "name": "my-agent",
                "domain": "agents.example.com",
                "protocol": "mcp",
                "endpoint": "mcp.example.com",
                "capabilities": ["search", "summarize"],
            })
    """

    name: str = "dns_aid_publish"
    description: str = (
        "Publishes an AI agent to DNS so other agents can discover it via "
        "DNS-AID SVCB records. Requires agent name, domain, protocol, and endpoint."
    )
    args_schema: Type[BaseModel] = DnsAidPublishInput

    backend_name: Optional[str] = None
    """DNS backend name (e.g. 'route53', 'cloudflare', 'ddns', 'mock')."""

    backend: Any = None
    """Pre-configured DNSBackend instance. Takes priority over backend_name."""

    def _get_backend(self) -> Any:
        if self.backend is not None:
            return self.backend
        if self.backend_name:
            create_backend = _import_create_backend()
            return create_backend(self.backend_name)
        return None

    def _run(
        self,
        name: str,
        domain: str,
        protocol: str = "mcp",
        endpoint: str = "",
        port: int = 443,
        capabilities: Optional[list[str]] = None,
        version: str = "1.0.0",
        description: Optional[str] = None,
        ttl: int = 3600,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Publish agent to DNS (sync)."""
        return _run_async(
            self._arun(
                name=name,
                domain=domain,
                protocol=protocol,
                endpoint=endpoint,
                port=port,
                capabilities=capabilities,
                version=version,
                description=description,
                ttl=ttl,
            )
        )

    async def _arun(
        self,
        name: str,
        domain: str,
        protocol: str = "mcp",
        endpoint: str = "",
        port: int = 443,
        capabilities: Optional[list[str]] = None,
        version: str = "1.0.0",
        description: Optional[str] = None,
        ttl: int = 3600,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Publish agent to DNS (async)."""
        dns_aid = _import_dns_aid()
        result = await dns_aid.publish(
            name=name,
            domain=domain,
            protocol=protocol,
            endpoint=endpoint,
            port=port,
            capabilities=capabilities,
            version=version,
            description=description,
            ttl=ttl,
            backend=self._get_backend(),
        )
        return json.dumps(result.model_dump(), default=str)


class DnsAidDiscoverTool(BaseTool):
    """Discover AI agents at a domain via DNS-AID.

    Queries DNS SVCB records to find published agents, optionally
    filtering by protocol or agent name.

    Setup:
        Install the ``dns-aid`` package:

        .. code-block:: bash

            pip install dns-aid

    Instantiate:
        .. code-block:: python

            from langchain_community.tools.dns_aid import DnsAidDiscoverTool

            tool = DnsAidDiscoverTool()

    Invoke:
        .. code-block:: python

            tool.invoke({"domain": "agents.example.com", "protocol": "mcp"})
    """

    name: str = "dns_aid_discover"
    description: str = (
        "Discovers AI agents published at a domain via DNS-AID SVCB records. "
        "Returns agent names, endpoints, capabilities, and protocols. "
        "Input should include the domain to search."
    )
    args_schema: Type[BaseModel] = DnsAidDiscoverInput

    def _run(
        self,
        domain: str,
        protocol: Optional[str] = None,
        name: Optional[str] = None,
        require_dnssec: bool = False,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Discover agents via DNS (sync)."""
        return _run_async(
            self._arun(
                domain=domain,
                protocol=protocol,
                name=name,
                require_dnssec=require_dnssec,
            )
        )

    async def _arun(
        self,
        domain: str,
        protocol: Optional[str] = None,
        name: Optional[str] = None,
        require_dnssec: bool = False,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Discover agents via DNS (async)."""
        dns_aid = _import_dns_aid()
        result = await dns_aid.discover(
            domain=domain,
            protocol=protocol,
            name=name,
            require_dnssec=require_dnssec,
        )
        return json.dumps(result.model_dump(), default=str)


class DnsAidUnpublishTool(BaseTool):
    """Remove an AI agent's DNS-AID records.

    Deletes the SVCB and TXT records for a previously published agent,
    making it no longer discoverable via DNS.

    Setup:
        Install the ``dns-aid`` package and configure a DNS backend:

        .. code-block:: bash

            pip install dns-aid
            export DNS_AID_BACKEND=route53

    Instantiate:
        .. code-block:: python

            from langchain_community.tools.dns_aid import DnsAidUnpublishTool

            tool = DnsAidUnpublishTool(backend_name="route53")

    Invoke:
        .. code-block:: python

            tool.invoke({
                "name": "my-agent",
                "domain": "agents.example.com",
                "protocol": "mcp",
            })
    """

    name: str = "dns_aid_unpublish"
    description: str = (
        "Removes an AI agent's DNS-AID records, making it no longer "
        "discoverable. Requires agent name, domain, and protocol."
    )
    args_schema: Type[BaseModel] = DnsAidUnpublishInput

    backend_name: Optional[str] = None
    """DNS backend name (e.g. 'route53', 'cloudflare', 'ddns', 'mock')."""

    backend: Any = None
    """Pre-configured DNSBackend instance. Takes priority over backend_name."""

    def _get_backend(self) -> Any:
        if self.backend is not None:
            return self.backend
        if self.backend_name:
            create_backend = _import_create_backend()
            return create_backend(self.backend_name)
        return None

    def _run(
        self,
        name: str,
        domain: str,
        protocol: str = "mcp",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Remove agent from DNS (sync)."""
        return _run_async(
            self._arun(name=name, domain=domain, protocol=protocol)
        )

    async def _arun(
        self,
        name: str,
        domain: str,
        protocol: str = "mcp",
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Remove agent from DNS (async)."""
        dns_aid = _import_dns_aid()
        deleted = await dns_aid.unpublish(
            name=name,
            domain=domain,
            protocol=protocol,
            backend=self._get_backend(),
        )
        if deleted:
            return json.dumps(
                {
                    "success": True,
                    "message": f"Agent '{name}' unpublished from {domain}",
                }
            )
        return json.dumps(
            {"success": False, "message": f"Agent '{name}' not found at {domain}"}
        )
