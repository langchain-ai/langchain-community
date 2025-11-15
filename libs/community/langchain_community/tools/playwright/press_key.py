from __future__ import annotations

from typing import Optional, Type

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from pydantic import BaseModel, Field

from langchain_community.tools.playwright.base import BaseBrowserTool
from langchain_community.tools.playwright.utils import (
    aget_current_page,
    get_current_page,
)


class PressKeyToolInput(BaseModel):
    """Input for PressKeyTool."""

    key: str = Field(
        ..., description="Key to press (e.g., 'Enter', 'Tab', 'ArrowDown', 'a', etc.)"
    )
    selector: Optional[str] = Field(
        None,
        description=(
            "Optional CSS selector for the element to focus before pressing key"
        ),
    )


class PressKeyTool(BaseBrowserTool):
    """Tool for pressing a key or keyboard shortcut."""

    name: str = "press_key"
    description: str = (
        "Press a keyboard key or shortcut. Optionally focus an element first by "
        "providing a selector."
    )
    args_schema: Type[BaseModel] = PressKeyToolInput

    visible_only: bool = True
    """Whether to consider only visible elements when selector is provided."""
    playwright_strict: bool = False
    """Whether to employ Playwright's strict mode when selecting elements."""
    playwright_timeout: float = 1_000
    """Timeout (in ms) for Playwright to wait for element to be ready."""

    def _selector_effective(self, selector: str) -> str:
        if not self.visible_only:
            return selector
        return f"{selector} >> visible=1"

    def _run(
        self,
        key: str,
        selector: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        if self.sync_browser is None:
            raise ValueError(f"Synchronous browser not provided to {self.name}")
        page = get_current_page(self.sync_browser)

        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

        try:
            if selector:
                selector_effective = self._selector_effective(selector=selector)
                # Focus the element first if selector is provided
                try:
                    page.focus(
                        selector_effective,
                        strict=self.playwright_strict,
                        timeout=self.playwright_timeout,
                    )
                except PlaywrightTimeoutError:
                    return f"Unable to focus element '{selector}'"

                # Press the key on the focused element
                page.press(
                    selector_effective,
                    key,
                    strict=self.playwright_strict,
                    timeout=self.playwright_timeout,
                )
                return f"Pressed '{key}' on element '{selector}'"
            else:
                # Press the key globally
                page.keyboard.press(key)
                return f"Pressed '{key}'"
        except Exception as e:
            return f"Error pressing key '{key}': {str(e)}"

    async def _arun(
        self,
        key: str,
        selector: Optional[str] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        if self.async_browser is None:
            raise ValueError(f"Asynchronous browser not provided to {self.name}")
        page = await aget_current_page(self.async_browser)

        from playwright.async_api import TimeoutError as PlaywrightTimeoutError

        try:
            if selector:
                selector_effective = self._selector_effective(selector=selector)
                # Focus the element first if selector is provided
                try:
                    await page.focus(
                        selector_effective,
                        strict=self.playwright_strict,
                        timeout=self.playwright_timeout,
                    )
                except PlaywrightTimeoutError:
                    return f"Unable to focus element '{selector}'"

                # Press the key on the focused element
                await page.press(
                    selector_effective,
                    key,
                    strict=self.playwright_strict,
                    timeout=self.playwright_timeout,
                )
                return f"Pressed '{key}' on element '{selector}'"
            else:
                # Press the key globally
                await page.keyboard.press(key)
                return f"Pressed '{key}'"
        except Exception as e:
            return f"Error pressing key '{key}': {str(e)}"
