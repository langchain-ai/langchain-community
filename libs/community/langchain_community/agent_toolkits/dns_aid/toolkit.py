"""Toolkit for DNS-AID agent discovery and publishing."""

from __future__ import annotations

from typing import Any, List, Optional

from langchain_core.tools import BaseTool
from langchain_core.tools.base import BaseToolkit
from pydantic import ConfigDict


class DnsAidToolkit(BaseToolkit):
    """Toolkit for DNS-AID agent discovery and publishing.

    Bundles tools for publishing, discovering, and unpublishing
    AI agents via DNS using the DNS-AID protocol (SVCB + TXT records).

    *Security Note*: This toolkit contains tools that can create and
        delete DNS records. Ensure appropriate DNS backend credentials
        are configured.

    Setup:
        Install the ``dns-aid`` package and configure a DNS backend:

        .. code-block:: bash

            pip install dns-aid
            export DNS_AID_BACKEND=route53  # or cloudflare, ddns, etc.

    Key init args:
        backend_name: Optional DNS backend name (e.g. 'route53', 'cloudflare').
        backend: Optional pre-configured DNSBackend instance.

    Instantiate:
        .. code-block:: python

            from langchain_community.agent_toolkits.dns_aid import DnsAidToolkit

            toolkit = DnsAidToolkit(backend_name="route53")

    Tools:
        .. code-block:: python

            toolkit.get_tools()

        .. code-block:: none

            [DnsAidPublishTool, DnsAidDiscoverTool, DnsAidUnpublishTool]

    Use within an agent:
        .. code-block:: python

            from langchain_openai import ChatOpenAI
            from langgraph.prebuilt import create_react_agent

            llm = ChatOpenAI(model="gpt-4o-mini")
            tools = toolkit.get_tools()
            agent = create_react_agent(llm, tools)

            events = agent.stream(
                {"messages": [("user", "Discover MCP agents")]},
                stream_mode="values",
            )
            for event in events:
                event["messages"][-1].pretty_print()

    Parameters:
        backend_name: DNS backend name (e.g. 'route53', 'cloudflare', 'ddns', 'mock').
        backend: Pre-configured DNSBackend instance. Takes priority over backend_name.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    backend_name: Optional[str] = None
    """DNS backend name (e.g. 'route53', 'cloudflare', 'ddns', 'mock')."""

    backend: Any = None
    """Pre-configured DNSBackend instance. Takes priority over backend_name."""

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        from langchain_community.tools.dns_aid import (
            DnsAidDiscoverTool,
            DnsAidPublishTool,
            DnsAidUnpublishTool,
        )

        kwargs: dict[str, Any] = {}
        if self.backend is not None:
            kwargs["backend"] = self.backend
        if self.backend_name is not None:
            kwargs["backend_name"] = self.backend_name

        return [
            DnsAidPublishTool(**kwargs),
            DnsAidDiscoverTool(),  # Discover doesn't need a backend
            DnsAidUnpublishTool(**kwargs),
        ]
