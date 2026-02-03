"""Private HTTP client for the ClawPrint API.

Handles authentication, base URL routing, and error mapping so the
tool layer never touches ``requests`` directly.

This module is internal to the ``langchain_community.tools.clawprint``
package.  Import tools or the toolkit instead.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any, Dict, List, Optional

import requests

CLAWPRINT_DEFAULT_BASE_URL = "https://clawprint.io"

_HANDLE_RE = re.compile(r"^@?[a-z0-9][a-z0-9\-]{0,62}$")


def validate_handle(handle: str) -> str:
    """Validate and normalize an agent handle.

    Args:
        handle: Agent handle to validate.

    Returns:
        Normalized handle string.

    Raises:
        ValueError: If the handle is invalid.
    """
    handle = handle.strip().lower().lstrip("@")
    if not handle:
        raise ValueError("Handle cannot be empty.")
    if not _HANDLE_RE.match(handle):
        raise ValueError(
            f"Invalid handle '{handle}'. "
            "Use lowercase letters, digits, and hyphens only."
        )
    return handle


class ClawPrintAPIError(Exception):
    """Raised when the ClawPrint API returns a non-2xx response."""

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"ClawPrint API error {status_code}: {detail}")


class ClawPrintClient:
    """Minimal, synchronous HTTP client for the ClawPrint REST API.

    Supports both sync and async usage.  Async methods use
    ``asyncio.get_running_loop().run_in_executor`` to bridge to sync I/O,
    following the pattern used by other langchain-community tools.

    Args:
        api_key: Bearer token for authenticated endpoints.
            Falls back to ``CLAWPRINT_API_KEY`` env var.
        base_url: API root.  Defaults to ``https://clawprint.io``.
        timeout: Request timeout in seconds.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = CLAWPRINT_DEFAULT_BASE_URL,
        timeout: int = 30,
    ) -> None:
        self.api_key = api_key or os.environ.get("CLAWPRINT_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Accept": "application/json",
                "User-Agent": "langchain-community/clawprint",
            }
        )

    # ------------------------------------------------------------------
    # Resource management
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying HTTP session."""
        self._session.close()

    def __del__(self) -> None:
        try:
            self._session.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _headers(self, auth_required: bool = False) -> Dict[str, str]:
        """Build per-request headers, injecting auth when needed."""
        headers: Dict[str, str] = {}
        if auth_required:
            if not self.api_key:
                raise ClawPrintAPIError(
                    401,
                    "This endpoint requires an API key. "
                    "Pass api_key= or set CLAWPRINT_API_KEY.",
                )
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _handle_response(self, resp: requests.Response) -> Dict[str, Any]:
        """Parse API response, raising on errors.

        Args:
            resp: The ``requests.Response`` to process.

        Returns:
            Parsed JSON body as a dictionary.

        Raises:
            ClawPrintAPIError: On non-2xx responses.
        """
        if resp.ok:
            if resp.status_code == 204:
                return {}
            return resp.json()  # type: ignore[no-any-return]
        # ClawPrint error bodies aren't totally consistent yet, so we
        # try a few keys before falling back to raw text.
        try:
            body = resp.json()
            detail = (
                body.get("detail")
                or body.get("message")
                or body.get("error", {}).get("message")
                or json.dumps(body)
            )
        except (ValueError, KeyError):
            detail = resp.text[:1000] if resp.text else f"HTTP {resp.status_code}"
        raise ClawPrintAPIError(resp.status_code, str(detail))

    # ------------------------------------------------------------------
    # Public sync API methods
    # ------------------------------------------------------------------

    def search_agents(
        self,
        query: str,
        domain: Optional[str] = None,
        min_score: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Search the agent registry.

        Args:
            query: Free-text search query.
            domain: Optional domain filter.
            min_score: Minimum trust score (0-100 scale).

        Returns:
            Search results dict containing matched agents.
        """
        if not query or not query.strip():
            raise ValueError("Search query cannot be empty.")
        params: Dict[str, Any] = {"q": query.strip()}
        if domain:
            params["domain"] = domain
        if min_score is not None:
            params["min_score"] = min_score
        resp = self._session.get(
            f"{self.base_url}/v1/agents/search",
            params=params,
            headers=self._headers(),
            timeout=self.timeout,
        )
        return self._handle_response(resp)

    def get_agent(self, handle: str) -> Dict[str, Any]:
        """Retrieve the full agent card for *handle*.

        Args:
            handle: Agent handle to look up.

        Returns:
            Full agent card dict.
        """
        handle = validate_handle(handle)
        resp = self._session.get(
            f"{self.base_url}/v1/agents/{handle}",
            headers=self._headers(),
            timeout=self.timeout,
        )
        return self._handle_response(resp)

    def get_trust(self, handle: str) -> Dict[str, Any]:
        """Fetch the trust score for *handle*.

        Args:
            handle: Agent handle to look up.

        Returns:
            Trust score dict with score, factors, and verification status.
        """
        handle = validate_handle(handle)
        # NOTE: trust lives under /v1/trust, not /v1/agents/:handle/trust
        resp = self._session.get(
            f"{self.base_url}/v1/trust/{handle}",
            headers=self._headers(),
            timeout=self.timeout,
        )
        return self._handle_response(resp)

    def list_domains(self) -> Dict[str, Any]:
        """List all available capability domains.

        Returns:
            Dict containing domain names and descriptions.
        """
        resp = self._session.get(
            f"{self.base_url}/v1/domains",
            headers=self._headers(),
            timeout=self.timeout,
        )
        return self._handle_response(resp)

    def register_agent(
        self,
        handle: str,
        name: str,
        description: str,
        domains: List[str],
        *,
        url: Optional[str] = None,
        pricing_model: str = "free",
        response_time: str = "async",
    ) -> Dict[str, Any]:
        """Register a new agent on ClawPrint.

        Args:
            handle: Unique lowercase handle.
            name: Human-readable agent name.
            description: What this agent does.
            domains: Capability domains.
            url: Optional homepage or endpoint URL.
            pricing_model: One of free, per_request, subscription, negotiable.
            response_time: One of sync, async, batch, streaming.

        Returns:
            Created agent card and API key.
        """
        handle = validate_handle(handle)
        identity: Dict[str, Any] = {
            "name": name,
            "handle": handle,
            "description": description,
        }
        if url:
            identity["url"] = url

        payload: Dict[str, Any] = {
            "agent_card": "0.2",
            "identity": identity,
            "services": [
                {
                    "id": "main",
                    "description": description,
                    "domains": domains,
                    "pricing": {"model": pricing_model},
                    "sla": {"response_time": response_time},
                }
            ],
        }
        resp = self._session.post(
            f"{self.base_url}/v1/agents",
            json=payload,
            headers=self._headers(),
            timeout=self.timeout,
        )
        return self._handle_response(resp)

    def create_exchange_request(
        self,
        domains: List[str],
        task: str,
        requirements: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Post a new exchange (hire) request.  Requires authentication.

        Args:
            domains: Capability domains the hired agent should cover.
            task: Plain-language task description.
            requirements: Optional structured requirements.

        Returns:
            Exchange request with request_id and status.
        """
        payload: Dict[str, Any] = {"domains": domains, "task": task}
        if requirements:
            payload["requirements"] = requirements
        resp = self._session.post(
            f"{self.base_url}/v1/exchange/requests",
            json=payload,
            headers=self._headers(auth_required=True),
            timeout=self.timeout,
        )
        return self._handle_response(resp)

    def get_exchange_request(self, request_id: str) -> Dict[str, Any]:
        """Check the status of an existing exchange request.

        Args:
            request_id: The exchange request ID.

        Returns:
            Exchange status dict.
        """
        resp = self._session.get(
            f"{self.base_url}/v1/exchange/requests/{request_id}",
            headers=self._headers(auth_required=True),
            timeout=self.timeout,
        )
        return self._handle_response(resp)

    # ------------------------------------------------------------------
    # Async wrappers (run_in_executor bridge)
    # TODO: native aiohttp client if this sees enough async usage
    # ------------------------------------------------------------------

    async def asearch_agents(
        self,
        query: str,
        domain: Optional[str] = None,
        min_score: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Async version of :meth:`search_agents`."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, lambda: self.search_agents(query, domain=domain, min_score=min_score)
        )

    async def aget_agent(self, handle: str) -> Dict[str, Any]:
        """Async version of :meth:`get_agent`."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.get_agent(handle))

    async def aget_trust(self, handle: str) -> Dict[str, Any]:
        """Async version of :meth:`get_trust`."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.get_trust(handle))

    async def alist_domains(self) -> Dict[str, Any]:
        """Async version of :meth:`list_domains`."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.list_domains)

    async def aregister_agent(
        self,
        handle: str,
        name: str,
        description: str,
        domains: List[str],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Async version of :meth:`register_agent`."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.register_agent(handle, name, description, domains, **kwargs),
        )

    async def acreate_exchange_request(
        self,
        domains: List[str],
        task: str,
        requirements: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Async version of :meth:`create_exchange_request`."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.create_exchange_request(domains, task, requirements),
        )

    async def aget_exchange_request(self, request_id: str) -> Dict[str, Any]:
        """Async version of :meth:`get_exchange_request`."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, lambda: self.get_exchange_request(request_id)
        )
