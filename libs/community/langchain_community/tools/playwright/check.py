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


class CheckToolInput(BaseModel):
    """Input for CheckTool."""

    selector: str = Field(
        ..., description="CSS selector for the checkbox or radio element"
    )
    check: bool = Field(
        True, description="Whether to check (True) or uncheck (False) the element"
    )


class CheckTool(BaseBrowserTool):
    """Tool for checking or unchecking checkboxes and radio buttons."""

    name: str = "check_element"
    description: str = "Check or uncheck a checkbox or radio button element"
    args_schema: Type[BaseModel] = CheckToolInput

    visible_only: bool = True
    """Whether to consider only visible elements."""
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
        selector: str,
        check: bool = True,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        if self.sync_browser is None:
            raise ValueError(f"Synchronous browser not provided to {self.name}")
        page = get_current_page(self.sync_browser)

        # Navigate to the desired webpage before using this tool
        selector_effective = self._selector_effective(selector=selector)
        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

        try:
            if check:
                page.check(
                    selector_effective,
                    strict=self.playwright_strict,
                    timeout=self.playwright_timeout,
                )
                action = "Checked"
            else:
                page.uncheck(
                    selector_effective,
                    strict=self.playwright_strict,
                    timeout=self.playwright_timeout,
                )
                action = "Unchecked"
        except PlaywrightTimeoutError:
            return f"Unable to {'check' if check else 'uncheck'} element '{selector}'"
        except Exception as e:
            return (
                f"Error {'checking' if check else 'unchecking'} "
                f"element '{selector}': {str(e)}"
            )

        return f"{action} element '{selector}'"

    async def _arun(
        self,
        selector: str,
        check: bool = True,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        if self.async_browser is None:
            raise ValueError(f"Asynchronous browser not provided to {self.name}")
        page = await aget_current_page(self.async_browser)

        # Navigate to the desired webpage before using this tool
        selector_effective = self._selector_effective(selector=selector)
        from playwright.async_api import TimeoutError as PlaywrightTimeoutError

        try:
            if check:
                await page.check(
                    selector_effective,
                    strict=self.playwright_strict,
                    timeout=self.playwright_timeout,
                )
                action = "Checked"
            else:
                await page.uncheck(
                    selector_effective,
                    strict=self.playwright_strict,
                    timeout=self.playwright_timeout,
                )
                action = "Unchecked"
        except PlaywrightTimeoutError:
            return f"Unable to {'check' if check else 'uncheck'} element '{selector}'"
        except Exception as e:
            return (
                f"Error {'checking' if check else 'unchecking'} "
                f"element '{selector}': {str(e)}"
            )

        return f"{action} element '{selector}'"
