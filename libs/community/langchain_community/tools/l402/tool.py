"""Tools for accessing L402-protected APIs with automatic Lightning payments.

L402 is a protocol that uses HTTP 402 responses with Lightning Network invoices
to enable machine-to-machine micropayments. These tools allow LangChain agents
to automatically pay Lightning invoices when accessing L402-protected APIs.

Setup:
    Install ``l402-requests``:

    .. code-block:: bash

        pip install l402-requests

    Set a wallet environment variable:

    .. code-block:: bash

        export STRIKE_API_KEY="your-key"  # or NWC_CONNECTION_STRING, LND_REST_HOST

Usage:
    .. code-block:: python

        from langchain_community.tools.l402 import L402FetchTool, L402SpendingTool

        tools = [L402FetchTool(), L402SpendingTool()]
        agent = create_react_agent(llm, tools)
"""

from __future__ import annotations

import json
from typing import Any, Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class _L402FetchInput(BaseModel):
    """Input for L402FetchTool."""

    url: str = Field(description="The full URL to request.")
    method: str = Field(
        default="GET",
        description="HTTP method: GET or POST.",
    )
    body: Optional[str] = Field(
        default=None,
        description="JSON string body for POST requests.",
    )


class L402FetchTool(BaseTool):
    """Fetch a URL that may require an L402 Lightning micropayment.

    Automatically handles HTTP 402 responses by paying the Lightning invoice
    and retrying with L402 credentials. Supports GET and POST.

    Setup:
        Install ``l402-requests``:

        .. code-block:: bash

            pip install l402-requests

        Set a wallet environment variable:

        .. code-block:: bash

            export STRIKE_API_KEY="your-key"

    Instantiate:
        .. code-block:: python

            from langchain_community.tools.l402 import L402FetchTool

            tool = L402FetchTool()

        With custom budget:

        .. code-block:: python

            from l402_requests import L402Client, BudgetController

            client = L402Client(
                budget=BudgetController(max_sats_per_request=500),
            )
            tool = L402FetchTool(l402_client=client)

    Invoke:
        .. code-block:: python

            tool.invoke({"url": "https://l402.services/geoip/8.8.8.8"})
    """

    name: str = "l402_fetch"
    description: str = (
        "Fetch a URL that may be behind an L402 Lightning paywall. "
        "If the server returns HTTP 402 (Payment Required), the Lightning "
        "invoice is paid automatically and the request is retried. "
        "Use this for any API that requires Lightning micropayments. "
        "Supports GET and POST methods."
    )
    args_schema: Type[BaseModel] = _L402FetchInput

    l402_client: Any = None
    """Optional l402_requests.L402Client instance. If not provided,
    a default client will be created using auto-detected wallet."""

    def _get_client(self) -> Any:
        try:
            from l402_requests import L402Client
        except ImportError:
            raise ImportError(
                "l402-requests is required for L402FetchTool. "
                "Install it with: pip install l402-requests"
            )
        if self.l402_client is None:
            self.l402_client = L402Client()
        return self.l402_client

    def _run(
        self,
        url: str,
        method: str = "GET",
        body: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Fetch a URL with automatic L402 payment handling."""
        client = self._get_client()
        try:
            if method.upper() == "POST":
                json_body = json.loads(body) if body else None
                response = client.post(url, json=json_body)
            else:
                response = client.get(url)

            try:
                data = response.json()
                return json.dumps(data, indent=2)
            except Exception:
                return response.text[:4000]

        except json.JSONDecodeError as e:
            return f"Error: Invalid JSON body — {e}"
        except Exception as e:
            error_type = type(e).__name__
            return f"Error: {error_type} — {e}"


class L402SpendingTool(BaseTool):
    """Check how many sats have been spent in this session.

    Returns a summary of Lightning payments made so far including total sats
    spent, per-domain breakdown, and hourly spending.

    Setup:
        Share the same client with ``L402FetchTool`` for accurate tracking:

        .. code-block:: python

            from l402_requests import L402Client
            from langchain_community.tools.l402 import L402FetchTool, L402SpendingTool

            client = L402Client()
            tools = [L402FetchTool(l402_client=client), L402SpendingTool(l402_client=client)]
    """

    name: str = "l402_spending"
    description: str = (
        "Returns a summary of Lightning payments made so far: "
        "total sats spent, payment count, and per-domain breakdown."
    )

    l402_client: Any = None
    """Optional l402_requests.L402Client instance. Must be the same client
    used by L402FetchTool for accurate spending tracking."""

    def _run(
        self,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Return spending summary."""
        if self.l402_client is None:
            return "No L402 client configured — pass the same client used by L402FetchTool."

        log = self.l402_client.spending_log
        total = log.total_spent()
        if total == 0:
            return "No L402 payments made yet."
        return json.dumps(
            {
                "total_sats": total,
                "spent_last_hour": log.spent_last_hour(),
                "by_domain": log.by_domain(),
            },
            indent=2,
        )
