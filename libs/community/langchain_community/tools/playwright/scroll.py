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


class ScrollToolInput(BaseModel):
    """Input for ScrollTool."""

    selector: Optional[str] = Field(
        None, description="CSS selector to scroll into view"
    )
    x: Optional[int] = Field(None, description="Pixels to scroll horizontally (x-axis)")
    y: Optional[int] = Field(None, description="Pixels to scroll vertically (y-axis)")


class ScrollTool(BaseBrowserTool):
    """Tool to scroll an element into view or scroll the page by x/y pixels."""

    name: str = "scroll_page"
    description: str = (
        "Scroll an element into view if 'selector' is provided, "
        "or scroll the page by a number of pixels using x and y."
    )
    args_schema: Type[BaseModel] = ScrollToolInput

    visible_only: bool = False
    playwright_strict: bool = False
    playwright_timeout: float = 10_000

    def _run(
        self,
        selector: Optional[str] = None,
        x: Optional[int] = None,
        y: Optional[int] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        if self.sync_browser is None:
            raise ValueError("Synchronous browser not initialized.")
        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

        page = get_current_page(self.sync_browser)

        try:
            if selector:
                page.wait_for_selector(selector, timeout=self.playwright_timeout)
                locator = page.locator(selector)

                if locator.count() == 0:
                    return f"No element found for selector: {selector}"

                element = locator.nth(0)
                if self.visible_only and not element.is_visible(
                    timeout=self.playwright_timeout / 2
                ):
                    return f"Element found but not visible: {selector}"

                handle = element.element_handle()
                if handle:
                    if x is not None or y is not None:
                        scroll_x = x or 0
                        scroll_y = y or 0
                        handle.evaluate(f"(el) => el.scrollBy({scroll_x}, {scroll_y})")
                        return (
                            f"Scrolled element '{selector}' by x={scroll_x}, "
                            f"y={scroll_y}"
                        )
                    else:
                        handle.scroll_into_view_if_needed(
                            timeout=self.playwright_timeout
                        )
                        return f"Scrolled to element '{selector}'"
                else:
                    return f"Element handle not usable for '{selector}'"

            elif x is not None or y is not None:
                scroll_x = x or 0
                scroll_y = y or 0
                page.evaluate(f"window.scrollBy({scroll_x}, {scroll_y})")
                return f"Scrolled page by x={scroll_x}, y={scroll_y}"

            return "No selector or scroll amount provided."

        except PlaywrightTimeoutError:
            return f"Timeout waiting for element '{selector}'"
        except Exception as e:
            return f"ScrollTool error: {str(e)}"

    async def _arun(
        self,
        selector: Optional[str] = None,
        x: Optional[int] = None,
        y: Optional[int] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        if self.async_browser is None:
            raise ValueError("Asynchronous browser not initialized.")
        from playwright.async_api import TimeoutError as PlaywrightTimeoutError

        page = await aget_current_page(self.async_browser)

        try:
            if selector:
                await page.wait_for_selector(selector, timeout=self.playwright_timeout)
                locator = page.locator(selector)
                count = await locator.count()

                if count == 0:
                    return f"No element found for selector: {selector}"

                element = locator.nth(0)
                if self.visible_only and not await element.is_visible(
                    timeout=self.playwright_timeout / 2
                ):
                    return f"Element found but not visible: {selector}"

                handle = await element.element_handle()
                if handle:
                    if x is not None or y is not None:
                        scroll_x = x or 0
                        scroll_y = y or 0
                        await handle.evaluate(
                            f"(el) => el.scrollBy({scroll_x}, {scroll_y})"
                        )
                        return (
                            f"Scrolled element '{selector}' by x={scroll_x}, "
                            f"y={scroll_y}"
                        )
                    else:
                        await handle.scroll_into_view_if_needed(
                            timeout=self.playwright_timeout
                        )
                        return f"Scrolled to element '{selector}'"
                else:
                    return f"Element handle not usable for '{selector}'"

            elif x is not None or y is not None:
                scroll_x = x or 0
                scroll_y = y or 0
                await page.evaluate(f"window.scrollBy({scroll_x}, {scroll_y})")
                return f"Scrolled page by x={scroll_x}, y={scroll_y}"

            return "No selector or scroll amount provided."

        except PlaywrightTimeoutError:
            return f"Timeout waiting for element '{selector}'"
        except Exception as e:
            return f"ScrollTool async error: {str(e)}"
