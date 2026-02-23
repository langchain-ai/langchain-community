"""Anakin API wrapper for web scraping, search, and agentic research.

Provides a shared API client used by ``AnakinLoader`` and the Anakin
tool classes.  All HTTP logic and polling for async jobs lives here.

Setup:
    Install ``langchain-community`` and set your API key:

    .. code-block:: bash

        pip install -U langchain-community
        export ANAKIN_API_KEY="your-api-key"

API reference: https://anakin.io/llms-full.txt
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List

import requests
from langchain_core.utils import secret_from_env
from pydantic import BaseModel, Field, SecretStr

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://api.anakin.io/v1"


class AnakinAPIWrapper(BaseModel):
    """Wrapper around the Anakin REST API.

    Handles authentication, request construction, and the async
    job polling pattern used by scrape / batch-scrape / agentic-search
    endpoints (POST -> jobId -> poll GET until completed).

    Setup:
        Set the ``ANAKIN_API_KEY`` environment variable or pass
        ``anakin_api_key`` explicitly:

        .. code-block:: bash

            export ANAKIN_API_KEY="your-api-key"

    Instantiate:
        .. code-block:: python

            from langchain_community.utilities import AnakinAPIWrapper

            # Uses ANAKIN_API_KEY env var
            wrapper = AnakinAPIWrapper()

            # Or pass explicitly
            wrapper = AnakinAPIWrapper(anakin_api_key="ak-...")

    Usage:
        .. code-block:: python

            # Scrape a URL
            result = wrapper.scrape("https://example.com")
            print(result["markdown"][:200])

            # Web search
            results = wrapper.search("latest AI news", limit=5)
            for r in results:
                print(r["title"], r["url"])

            # Deep research (1-5 min)
            report = wrapper.agentic_search("compare React vs Vue")
            print(report["generatedJson"]["summary"])
    """

    anakin_api_key: SecretStr = Field(
        default_factory=secret_from_env(
            "ANAKIN_API_KEY",
            error_message=(
                "Anakin API key must be provided via the 'anakin_api_key' "
                "parameter or the ANAKIN_API_KEY environment variable. "
                "Get your key at https://anakin.io/dashboard"
            ),
        ),
    )
    """Anakin API key.  Falls back to the ``ANAKIN_API_KEY`` env var."""

    api_base_url: str = _DEFAULT_BASE_URL
    """Base URL of the Anakin REST API."""

    timeout: int = 300
    """Maximum seconds to wait for an async job to complete."""

    poll_interval: int = 3
    """Seconds between poll requests for async jobs."""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _headers(self) -> Dict[str, str]:
        return {
            "X-API-Key": self.anakin_api_key.get_secret_value(),
            "Content-Type": "application/json",
            "X-Integration": "langchain",
        }

    def _url(self, path: str) -> str:
        base = self.api_base_url.rstrip("/")
        return f"{base}/{path.lstrip('/')}"

    def _submit_job(self, path: str, payload: Dict[str, Any]) -> str:
        """POST a job and return the job ID."""
        resp = requests.post(
            self._url(path),
            json=payload,
            headers=self._headers(),
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        job_id = data.get("jobId") or data.get("job_id")
        if not job_id:
            msg = f"No job ID in response: {data}"
            raise ValueError(msg)
        return job_id

    def _poll_result(self, path: str, job_id: str) -> Dict[str, Any]:
        """Poll GET ``path/{job_id}`` until terminal status or timeout."""
        url = self._url(f"{path}/{job_id}")
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            resp = requests.get(url, headers=self._headers(), timeout=30)
            resp.raise_for_status()
            data = resp.json()
            status = data.get("status", "")
            if status == "completed":
                return data
            if status == "failed":
                error = data.get("error", "Unknown error")
                msg = f"Anakin job {job_id} failed: {error}"
                raise RuntimeError(msg)
            time.sleep(self.poll_interval)
        msg = f"Anakin job {job_id} did not complete within {self.timeout}s"
        raise TimeoutError(msg)

    async def _asubmit_job(self, path: str, payload: Dict[str, Any]) -> str:
        """Async variant of :meth:`_submit_job`."""
        try:
            import aiohttp
        except ImportError:
            msg = (
                "aiohttp is required for async operations. "
                "Install it with: pip install aiohttp"
            )
            raise ImportError(msg)
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self._url(path),
                json=payload,
                headers=self._headers(),
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
                job_id = data.get("jobId") or data.get("job_id")
                if not job_id:
                    msg = f"No job ID in response: {data}"
                    raise ValueError(msg)
                return job_id

    async def _apoll_result(self, path: str, job_id: str) -> Dict[str, Any]:
        """Async variant of :meth:`_poll_result`."""
        try:
            import aiohttp
        except ImportError:
            msg = (
                "aiohttp is required for async operations. "
                "Install it with: pip install aiohttp"
            )
            raise ImportError(msg)
        url = self._url(f"{path}/{job_id}")
        deadline = time.monotonic() + self.timeout
        async with aiohttp.ClientSession() as session:
            while time.monotonic() < deadline:
                async with session.get(
                    url,
                    headers=self._headers(),
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    status = data.get("status", "")
                    if status == "completed":
                        return data
                    if status == "failed":
                        error = data.get("error", "Unknown error")
                        msg = f"Anakin job {job_id} failed: {error}"
                        raise RuntimeError(msg)
                await asyncio.sleep(self.poll_interval)
        msg = f"Anakin job {job_id} did not complete within {self.timeout}s"
        raise TimeoutError(msg)

    # ------------------------------------------------------------------
    # Public API – Scrape
    # ------------------------------------------------------------------

    def scrape(
        self,
        url: str,
        *,
        country: str = "us",
        use_browser: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Scrape a single URL and return the result dict.

        Args:
            url: The URL to scrape.
            country: ISO 3166-1 alpha-2 country code for proxy routing.
            use_browser: Whether to use headless browser rendering.
            **kwargs: Additional parameters passed to the API.

        Returns:
            Dict with keys: markdown, html, cleanedHtml, url, status, etc.
        """
        payload: Dict[str, Any] = {
            "url": url,
            "country": country,
            "useBrowser": use_browser,
            **kwargs,
        }
        job_id = self._submit_job("url-scraper", payload)
        return self._poll_result("url-scraper", job_id)

    async def ascrape(
        self,
        url: str,
        *,
        country: str = "us",
        use_browser: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Async variant of :meth:`scrape`."""
        payload: Dict[str, Any] = {
            "url": url,
            "country": country,
            "useBrowser": use_browser,
            **kwargs,
        }
        job_id = await self._asubmit_job("url-scraper", payload)
        return await self._apoll_result("url-scraper", job_id)

    def batch_scrape(
        self,
        urls: List[str],
        *,
        country: str = "us",
        use_browser: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Scrape up to 10 URLs in a single batch request.

        Args:
            urls: List of URLs (max 10).
            country: ISO 3166-1 alpha-2 country code for proxy routing.
            use_browser: Whether to use headless browser rendering.
            **kwargs: Additional parameters passed to the API.

        Returns:
            Dict with a ``results`` list, one entry per URL.
        """
        if len(urls) > 10:
            msg = "Batch scrape supports a maximum of 10 URLs."
            raise ValueError(msg)
        payload: Dict[str, Any] = {
            "urls": urls,
            "country": country,
            "useBrowser": use_browser,
            **kwargs,
        }
        job_id = self._submit_job("url-scraper/batch", payload)
        return self._poll_result("url-scraper", job_id)

    async def abatch_scrape(
        self,
        urls: List[str],
        *,
        country: str = "us",
        use_browser: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Async variant of :meth:`batch_scrape`."""
        if len(urls) > 10:
            msg = "Batch scrape supports a maximum of 10 URLs."
            raise ValueError(msg)
        payload: Dict[str, Any] = {
            "urls": urls,
            "country": country,
            "useBrowser": use_browser,
            **kwargs,
        }
        job_id = await self._asubmit_job("url-scraper/batch", payload)
        return await self._apoll_result("url-scraper", job_id)

    # ------------------------------------------------------------------
    # Public API – Search (synchronous endpoint)
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Run an AI-powered web search. Returns results immediately.

        Args:
            query: The search query.
            limit: Maximum number of results.
            **kwargs: Additional parameters passed to the API.

        Returns:
            List of result dicts with: url, title, snippet, date.
        """
        payload: Dict[str, Any] = {
            "prompt": query,
            "limit": limit,
            **kwargs,
        }
        resp = requests.post(
            self._url("search"),
            json=payload,
            headers=self._headers(),
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("results", [])

    async def asearch(
        self,
        query: str,
        *,
        limit: int = 5,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Async variant of :meth:`search`."""
        try:
            import aiohttp
        except ImportError:
            msg = (
                "aiohttp is required for async operations. "
                "Install it with: pip install aiohttp"
            )
            raise ImportError(msg)
        payload: Dict[str, Any] = {
            "prompt": query,
            "limit": limit,
            **kwargs,
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self._url("search"),
                json=payload,
                headers=self._headers(),
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
                return data.get("results", [])

    # ------------------------------------------------------------------
    # Public API – Agentic Search (async job)
    # ------------------------------------------------------------------

    def agentic_search(self, query: str, **kwargs: Any) -> Dict[str, Any]:
        """Run a multi-stage autonomous research job.

        This is a long-running operation (typically 1-5 minutes).

        Args:
            query: The research question.
            **kwargs: Additional parameters passed to the API.

        Returns:
            Dict with generatedJson containing summary and structured_data.
        """
        payload: Dict[str, Any] = {"prompt": query, **kwargs}
        job_id = self._submit_job("agentic-search", payload)
        # Agentic search is long-running; use a longer poll interval
        original_interval = self.poll_interval
        try:
            self.poll_interval = max(self.poll_interval, 10)
            return self._poll_result("agentic-search", job_id)
        finally:
            self.poll_interval = original_interval

    async def aagentic_search(self, query: str, **kwargs: Any) -> Dict[str, Any]:
        """Async variant of :meth:`agentic_search`."""
        payload: Dict[str, Any] = {"prompt": query, **kwargs}
        job_id = await self._asubmit_job("agentic-search", payload)
        original_interval = self.poll_interval
        try:
            self.poll_interval = max(self.poll_interval, 10)
            return await self._apoll_result("agentic-search", job_id)
        finally:
            self.poll_interval = original_interval
