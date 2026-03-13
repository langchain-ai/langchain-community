"""Browser tool for isolated browser automation via Chrome DevTools Protocol."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Literal, Optional, Type

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr

if TYPE_CHECKING:
    from boxlite import BrowserBox


class BrowserInput(BaseModel):
    """Input schema for BrowserTool."""

    action: Literal["start", "stop", "endpoint"] = Field(
        description="Action: start (launch browser), stop (close browser), endpoint (get CDP URL)"
    )


class BrowserTool(BaseTool):
    """Tool for browser automation in an isolated VM environment.

    BoxLite BrowserBox provides isolated browsers (Chromium, Firefox, WebKit)
    running inside a VM. The browsers expose Chrome DevTools Protocol (CDP)
    endpoints that can be used with Puppeteer or Playwright for automation.

    Setup:
        Install ``boxlite``:

        .. code-block:: bash

            pip install boxlite

    Instantiation:
        .. code-block:: python

            from langchain_community.tools.boxlite import BrowserTool

            # Default Chromium browser
            tool = BrowserTool()

            # Firefox with more resources
            tool = BrowserTool(browser="firefox", memory=4096)

    Invocation with args:
        .. code-block:: python

            # Start browser and get CDP endpoint
            result = await tool.ainvoke({"action": "start"})
            # Returns: {"status": "started", "endpoint": "http://localhost:9222"}

            # Get endpoint for existing browser
            result = await tool.ainvoke({"action": "endpoint"})
            # Returns: {"endpoint": "http://localhost:9222"}

            # Stop browser
            result = await tool.ainvoke({"action": "stop"})

    Invocation with ToolCall:
        .. code-block:: python

            tool.invoke({
                "args": {"action": "start"},
                "id": "1",
                "name": tool.name,
                "type": "tool_call",
            })

    Usage with Playwright:
        .. code-block:: python

            from playwright.async_api import async_playwright

            # Start browser
            result = await tool.ainvoke({"action": "start"})
            endpoint = result["endpoint"]

            # Connect with Playwright
            async with async_playwright() as p:
                browser = await p.chromium.connect_over_cdp(endpoint)
                page = await browser.new_page()
                await page.goto("https://example.com")
                # ... automate
    """

    name: str = "browser"
    description: str = (
        "Control an isolated web browser for automation. "
        "Use this for web scraping, testing, or browser-based tasks. "
        "Actions: start (launches browser, returns CDP endpoint), "
        "endpoint (returns CDP URL for Puppeteer/Playwright connection), "
        "stop (closes browser). "
        "The browser runs in an isolated VM for security. "
        "Use the returned endpoint URL to connect with Playwright or Puppeteer."
    )
    args_schema: Type[BaseModel] = BrowserInput

    browser: Literal["chromium", "firefox", "webkit"] = Field(
        default="chromium",
        description="Browser type: chromium, firefox, or webkit",
    )
    memory: int = Field(default=2048, description="Memory in MiB")
    cpus: int = Field(default=2, description="Number of CPU cores")

    _browser_box: Optional[BrowserBox] = PrivateAttr(default=None)

    def _get_default_port(self) -> int:
        """Get default CDP port for browser type."""
        ports = {
            "chromium": 9222,
            "firefox": 9223,
            "webkit": 9224,
        }
        return ports.get(self.browser, 9222)

    async def _ensure_browser(self) -> "BrowserBox":
        """Ensure browser is started."""
        try:
            from boxlite import BrowserBox, BrowserBoxOptions
        except ImportError as e:
            raise ImportError(
                "Unable to import boxlite, please install with `pip install boxlite`."
            ) from e

        if self._browser_box is None:
            opts = BrowserBoxOptions(
                browser=self.browser,
                memory=self.memory,
                cpu=self.cpus,
            )
            self._browser_box = BrowserBox(opts)
            await self._browser_box.__aenter__()
        return self._browser_box

    async def _cleanup(self) -> None:
        """Cleanup browser resources."""
        if self._browser_box is not None:
            await self._browser_box.__aexit__(None, None, None)
            self._browser_box = None

    async def _arun(
        self,
        action: str,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> dict[str, Any]:
        """Execute browser action asynchronously.

        Args:
            action: The action to perform (start, stop, endpoint).
            run_manager: Callback manager.

        Returns:
            Dict with action result.
        """
        try:
            if action == "start":
                browser_box = await self._ensure_browser()
                endpoint = browser_box.endpoint()
                return {
                    "status": "started",
                    "browser": self.browser,
                    "endpoint": endpoint,
                }

            elif action == "endpoint":
                if self._browser_box is None:
                    return {
                        "status": "not_started",
                        "error": "Browser not started. Use action='start' first.",
                    }
                endpoint = self._browser_box.endpoint()
                return {"endpoint": endpoint}

            elif action == "stop":
                if self._browser_box is not None:
                    await self._cleanup()
                    return {"status": "stopped"}
                return {"status": "not_running"}

            else:
                return {"error": f"Unknown action: {action}"}

        except Exception as e:
            return {"error": f"Browser error: {e!r}"}

    def _run(
        self,
        action: str,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> dict[str, Any]:
        """Execute browser action synchronously."""
        return asyncio.get_event_loop().run_until_complete(
            self._arun(action, run_manager=None)
        )
