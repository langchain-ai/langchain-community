"""Tools for interacting with the ClawPrint agent registry.

`ClawPrint <https://clawprint.io>`_ is a REST API where AI agents register
capabilities, discover each other, and broker task exchanges.

Seven tools + a toolkit wrapper.  Read-only tools (search, get, trust,
domains) work without auth.  Write tools (register, hire, check exchange)
need a ``CLAWPRINT_API_KEY``.
"""

from langchain_community.tools.clawprint.tool import (
    ClawPrintCheckExchangeTool,
    ClawPrintDomainsTool,
    ClawPrintGetAgentTool,
    ClawPrintHireAgentTool,
    ClawPrintRegisterTool,
    ClawPrintSearchTool,
    ClawPrintToolkit,
    ClawPrintTrustTool,
)

__all__ = [
    "ClawPrintRegisterTool",
    "ClawPrintSearchTool",
    "ClawPrintGetAgentTool",
    "ClawPrintTrustTool",
    "ClawPrintDomainsTool",
    "ClawPrintHireAgentTool",
    "ClawPrintCheckExchangeTool",
    "ClawPrintToolkit",
]
