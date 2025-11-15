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


class InputTextToolInput(BaseModel):
    """Input for InputTextTool."""

    selector: str = Field(..., description="CSS selector for the input element")
    text: str = Field(..., description="Text to type into the input element")
    nth: Optional[int] = Field(
        0,
        description=(
            "Index of the element to use when multiple are found (default is 0)"
        ),
    )


class InputTextTool(BaseBrowserTool):
    """Tool for typing text into an input element with the given CSS selector."""

    name: str = "input_text"
    description: str = (
        "Type text into an input element using a CSS selector. "
        "Use nth to pick among multiple matches."
    )
    args_schema: Type[BaseModel] = InputTextToolInput

    visible_only: bool = True
    playwright_strict: bool = False
    playwright_timeout: float = 5_000

    def _run(
        self,
        selector: str,
        text: str,
        nth: Optional[int] = 0,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

        if self.sync_browser is None:
            raise ValueError("Synchronous browser not initialized.")

        page = get_current_page(self.sync_browser)
        page.wait_for_load_state("load")

        try:
            page.wait_for_selector(selector, timeout=self.playwright_timeout)
            locator = page.locator(selector)

            if locator.count() == 0:
                return f"No element found for selector: {selector}"

            element = locator.nth(nth or 0)
            element.scroll_into_view_if_needed(timeout=self.playwright_timeout)

            if (
                not element.is_visible(timeout=self.playwright_timeout / 2)
                and self.visible_only
            ):
                return "Element found but not visible. Try disabling visible_only."

            element.click()
            element.type(text, delay=50)
            return f"Entered text into element '{selector}' (nth={nth})"

        except PlaywrightTimeoutError:
            return f"Timeout waiting for element '{selector}'"
        except Exception as e:
            return f"InputTextTool error: {str(e)}"

    async def _arun(
        self,
        selector: str,
        text: str,
        nth: Optional[int] = 0,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        from playwright.async_api import TimeoutError as PlaywrightTimeoutError

        if self.async_browser is None:
            raise ValueError("Asynchronous browser not initialized.")

        page = await aget_current_page(self.async_browser)

        try:
            await page.wait_for_selector(selector, timeout=self.playwright_timeout)
            locator = page.locator(selector)
            count = await locator.count()

            if count == 0:
                return f"No element found for selector: {selector}"

            element = locator.nth(nth or 0)
            await element.scroll_into_view_if_needed(timeout=self.playwright_timeout)

            if (
                not await element.is_visible(timeout=self.playwright_timeout / 2)
                and self.visible_only
            ):
                return "Element found but not visible. Try disabling visible_only."

            await element.click()
            await element.type(text, delay=50)
            return f"Entered text into element '{selector}' (nth={nth})"

        except PlaywrightTimeoutError:
            return f"Timeout waiting for element '{selector}'"
        except Exception as e:
            return f"InputTextTool async error: {str(e)}"
