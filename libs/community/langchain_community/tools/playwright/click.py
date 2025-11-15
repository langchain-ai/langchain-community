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


class ClickToolInput(BaseModel):
    """Input for ClickTool."""

    css_selector: Optional[str] = Field(None, description="CSS selector to click")
    text: Optional[str] = Field(None, description="Text content to match and click")
    xpath: Optional[str] = Field(None, description="XPath to locate and click")
    data_attribute: Optional[str] = Field(
        None, description="Name of data-* attribute (e.g., data-testid)"
    )
    data_value: Optional[str] = Field(None, description="Value of the data-* attribute")
    nth: Optional[int] = Field(
        0,
        description=(
            "Index of the element to click when multiple are found (default is 0)"
        ),
    )
    force: bool = Field(
        False, description="Whether to force the click (bypass visibility)"
    )
    wait_for_navigation: bool = Field(
        False, description="Wait for navigation after the click"
    )
    wait_for_timeout: int = Field(0, description="Milliseconds to wait after clicking")


class ClickTool(BaseBrowserTool):
    name: str = "click_element"
    description: str = (
        "Click on an element using one of: css_selector, text content (text=...), "
        "XPath, or data-* attributes. You may specify nth to choose a specific "
        "element when multiple match."
    )
    args_schema: Type[BaseModel] = ClickToolInput

    visible_only: bool = False
    playwright_strict: bool = False
    playwright_timeout: float = 10_000

    def _build_selector(
        self,
        css_selector: Optional[str],
        text: Optional[str],
        xpath: Optional[str],
        data_attribute: Optional[str],
        data_value: Optional[str],
    ) -> str:
        if css_selector:
            return css_selector
        elif text:
            return f"text={text}"
        elif xpath:
            return f"xpath={xpath}"
        elif data_attribute and data_value:
            return f'[{data_attribute}="{data_value}"]'
        else:
            raise ValueError(
                "Provide css_selector, text, xpath, or data_attribute + data_value"
            )

    def _run(
        self,
        css_selector: Optional[str] = None,
        text: Optional[str] = None,
        xpath: Optional[str] = None,
        data_attribute: Optional[str] = None,
        data_value: Optional[str] = None,
        nth: Optional[int] = 0,
        force: bool = False,
        wait_for_navigation: bool = False,
        wait_for_timeout: int = 0,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        from playwright.sync_api import (
            Error as PlaywrightError,
        )
        from playwright.sync_api import (
            TimeoutError as PlaywrightTimeoutError,
        )

        if self.sync_browser is None:
            raise ValueError("Synchronous browser not initialized.")

        page = get_current_page(self.sync_browser)
        page.wait_for_load_state("load")
        try:
            selector = self._build_selector(
                css_selector, text, xpath, data_attribute, data_value
            )
            page.wait_for_selector(selector, timeout=self.playwright_timeout)
            locator = page.locator(selector)
            if locator.count() == 0:
                return f"No element found for selector: {selector}"

            element = locator.nth(nth or 0)
            element.scroll_into_view_if_needed(timeout=self.playwright_timeout)

            if (
                not element.is_visible(timeout=self.playwright_timeout / 2)
                and not force
                and self.visible_only
            ):
                return "Element found but not visible. Use force=True to click it."

            try:
                if wait_for_navigation:
                    with page.expect_navigation(timeout=self.playwright_timeout * 2):
                        element.click(force=force, timeout=self.playwright_timeout)
                else:
                    element.click(force=force, timeout=self.playwright_timeout)
            except (PlaywrightTimeoutError, PlaywrightError):
                handle = element.element_handle()
                if handle and page.evaluate("el => document.body.contains(el)", handle):
                    page.evaluate("el => el.click()", handle)
                else:
                    return "Fallback click failed. Element not attached to DOM."

            if wait_for_timeout > 0:
                page.wait_for_timeout(wait_for_timeout)

            return f"Clicked element (nth={nth}). Current URL: {page.url}"

        except Exception as e:
            return f"ClickTool error: {str(e)}"

    async def _arun(
        self,
        css_selector: Optional[str] = None,
        text: Optional[str] = None,
        xpath: Optional[str] = None,
        data_attribute: Optional[str] = None,
        data_value: Optional[str] = None,
        nth: Optional[int] = 0,
        force: bool = False,
        wait_for_navigation: bool = False,
        wait_for_timeout: int = 0,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        from playwright.async_api import (
            Error as PlaywrightError,
        )
        from playwright.async_api import (
            TimeoutError as PlaywrightTimeoutError,
        )

        if self.async_browser is None:
            raise ValueError("Asynchronous browser not initialized.")

        page = await aget_current_page(self.async_browser)
        await page.wait_for_load_state("load")
        try:
            selector = self._build_selector(
                css_selector, text, xpath, data_attribute, data_value
            )
            await page.wait_for_selector(selector, timeout=self.playwright_timeout)
            locator = page.locator(selector)
            count = await locator.count()
            if count == 0:
                return f"No element found for selector: {selector}"

            element = locator.nth(nth or 0)
            await element.scroll_into_view_if_needed(timeout=self.playwright_timeout)

            if (
                not await element.is_visible(timeout=self.playwright_timeout / 2)
                and not force
                and self.visible_only
            ):
                return "Element found but not visible. Use force=True to click it."

            try:
                if wait_for_navigation:
                    async with page.expect_navigation(
                        timeout=self.playwright_timeout * 2
                    ):
                        await element.click(
                            force=force, timeout=self.playwright_timeout
                        )
                else:
                    await element.click(force=force, timeout=self.playwright_timeout)
            except (PlaywrightTimeoutError, PlaywrightError):
                handle = await element.element_handle()
                if handle and await page.evaluate(
                    "el => document.body.contains(el)", handle
                ):
                    await page.evaluate("el => el.click()", handle)
                else:
                    return "Fallback click failed. Element not attached to DOM."

            if wait_for_timeout > 0:
                await page.wait_for_timeout(wait_for_timeout)

            return f"Clicked element (nth={nth}). Current URL: {page.url}"

        except Exception as e:
            return f"ClickTool async error: {str(e)}"
