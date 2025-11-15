from __future__ import annotations

from typing import List, Optional, Type, Union

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


class SelectOptionToolInput(BaseModel):
    """Input for SelectOptionTool."""

    selector: str = Field(..., description="CSS selector for the select element")
    values: Union[List[str], str] = Field(
        ...,
        description=(
            "Value(s) to select. Can be a single string or a list of strings. "
            "These are the option values, not necessarily the display text."
        ),
    )


class SelectOptionTool(BaseBrowserTool):
    """Tool for selecting option(s) from a dropdown/select element."""

    name: str = "select_option"
    description: str = "Select option(s) from a dropdown or select element"
    args_schema: Type[BaseModel] = SelectOptionToolInput

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
        values: Union[List[str], str],
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        if self.sync_browser is None:
            raise ValueError(f"Synchronous browser not provided to {self.name}")
        page = get_current_page(self.sync_browser)

        # Standardize values to list format
        values_list = values if isinstance(values, list) else [values]

        # Navigate to the desired webpage before using this tool
        selector_effective = self._selector_effective(selector=selector)
        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

        try:
            page.select_option(
                selector_effective,
                values_list,
                strict=self.playwright_strict,
                timeout=self.playwright_timeout,
            )
        except PlaywrightTimeoutError:
            return f"Unable to select option(s) from element '{selector}'"
        except Exception as e:
            return f"Error selecting option(s) from '{selector}': {str(e)}"

        values_display = ", ".join(f"'{v}'" for v in values_list)
        return f"Selected {values_display} from dropdown '{selector}'"

    async def _arun(
        self,
        selector: str,
        values: Union[List[str], str],
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        if self.async_browser is None:
            raise ValueError(f"Asynchronous browser not provided to {self.name}")
        page = await aget_current_page(self.async_browser)

        # Standardize values to list format
        values_list = values if isinstance(values, list) else [values]

        # Navigate to the desired webpage before using this tool
        selector_effective = self._selector_effective(selector=selector)
        from playwright.async_api import TimeoutError as PlaywrightTimeoutError

        try:
            await page.select_option(
                selector_effective,
                values_list,
                strict=self.playwright_strict,
                timeout=self.playwright_timeout,
            )
        except PlaywrightTimeoutError:
            return f"Unable to select option(s) from element '{selector}'"
        except Exception as e:
            return f"Error selecting option(s) from '{selector}': {str(e)}"

        values_display = ", ".join(f"'{v}'" for v in values_list)
        return f"Selected {values_display} from dropdown '{selector}'"
