from __future__ import annotations

import asyncio
from typing import Optional, Type

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from playwright.async_api import (
    ElementHandle as AsyncElementHandle,
)
from playwright.async_api import (
    Frame as AsyncFrame,
)
from playwright.async_api import (
    Page as AsyncPage,
)
from playwright.sync_api import ElementHandle as SyncElementHandle
from playwright.sync_api import Frame, Page
from pydantic import BaseModel, Field

from langchain_community.tools.playwright.base import BaseBrowserTool
from langchain_community.tools.playwright.utils import (
    aget_current_page as lc_aget_current_page,
)
from langchain_community.tools.playwright.utils import (
    get_current_page as lc_get_current_page,
)

# Define a shared attribute name for the active frame on the page object
ACTIVE_FRAME_ATTR = "_lc_active_frame"


class SwitchFrameInput(BaseModel):
    name_or_selector: str = Field(
        ...,
        description=(
            "The name (name attribute of the iframe), ID (e.g., '#frameId'), or a "
            "CSS selector for the iframe element (e.g., 'iframe.my-class', "
            "'iframe[title=\"some title\"]'). Use special value 'PARENT_FRAME' or "
            "'DEFAULT_CONTENT' to switch back to the main page."
        ),
    )


class SwitchFrameTool(BaseBrowserTool):
    name: str = "switch_to_frame"
    description: str = (
        "Switches the browser context to a specified iframe or back to the main "
        "document. All subsequent element interactions will target within the "
        "switched frame until switched again."
    )
    args_schema: Type[BaseModel] = SwitchFrameInput

    def _handle_switch_sync(self, page: Page, name_or_selector: str) -> str:
        """Sync logic for switching frames."""
        if name_or_selector.upper() in ["PARENT_FRAME", "DEFAULT_CONTENT"]:
            if hasattr(page, ACTIVE_FRAME_ATTR):
                delattr(page, ACTIVE_FRAME_ATTR)
            return "Switched back to the main page content."

        resolved_frame: Optional[Frame] = None
        selector_detail = ""

        try:
            resolved_frame = page.frame(name=name_or_selector)
            if resolved_frame:
                selector_detail = f"by name attribute '{name_or_selector}'"
        except Exception:
            pass

        if not resolved_frame:
            potential_selectors = [
                name_or_selector,
                f"iframe[name='{name_or_selector}']",
                f"iframe#{name_or_selector.replace('#', '')}",
            ]
            if name_or_selector.startswith(".") and not name_or_selector.startswith(
                "iframe."
            ):
                potential_selectors.append(f"iframe{name_or_selector}")

            iframe_element: Optional[SyncElementHandle] = None
            for sel in potential_selectors:
                try:
                    element = page.query_selector(sel, timeout=2000)
                    if element and element.evaluate(
                        "el => el.tagName.toLowerCase() === 'iframe'"
                    ):
                        iframe_element = element
                        selector_detail = f"by selector '{sel}'"
                        break
                    elif element:
                        element.dispose()
                except Exception:
                    continue

            if iframe_element:
                try:
                    resolved_frame = iframe_element.content_frame()
                    iframe_element.dispose()
                except Exception as e:
                    return f"Error getting content frame ({selector_detail}): {e}"

        if not resolved_frame:
            return (
                f"Could not find an iframe matching '{name_or_selector}'."
            )

        setattr(page, ACTIVE_FRAME_ATTR, resolved_frame)
        return f"Successfully switched to frame '{name_or_selector}' ({selector_detail})."

    async def _handle_switch_async(self, page: AsyncPage, name_or_selector: str) -> str:
        """Async logic for switching frames."""
        if name_or_selector.upper() in ["PARENT_FRAME", "DEFAULT_CONTENT"]:
            if hasattr(page, ACTIVE_FRAME_ATTR):
                delattr(page, ACTIVE_FRAME_ATTR)
            return "Switched back to the main page content."

        resolved_frame: Optional[AsyncFrame] = None
        selector_detail = ""

        try:
            resolved_frame = page.frame(name=name_or_selector)
            if resolved_frame:
                selector_detail = f"by name attribute '{name_or_selector}'"
        except Exception:
            pass

        if not resolved_frame:
            potential_selectors = [
                name_or_selector,
                f"iframe[name='{name_or_selector}']",
                f"iframe#{name_or_selector.replace('#', '')}",
            ]
            if name_or_selector.startswith(".") and not name_or_selector.startswith(
                "iframe."
            ):
                potential_selectors.append(f"iframe{name_or_selector}")

            iframe_element: Optional[AsyncElementHandle] = None
            for sel in potential_selectors:
                try:
                    element = await asyncio.wait_for(
                        page.query_selector(sel), timeout=2.0
                    )
                    if element and await element.evaluate(
                        "el => el.tagName.toLowerCase() === 'iframe'"
                    ):
                        iframe_element = element
                        selector_detail = f"by selector '{sel}'"
                        break
                    elif element:
                        await element.dispose()
                except (asyncio.TimeoutError, Exception):
                    continue

            if iframe_element:
                try:
                    resolved_frame = await iframe_element.content_frame()
                    await iframe_element.dispose()
                except Exception as e:
                    return f"Error getting content frame ({selector_detail}): {e}"

        if not resolved_frame:
            return (
                f"Could not find an iframe matching '{name_or_selector}'."
            )

        setattr(page, ACTIVE_FRAME_ATTR, resolved_frame)
        return f"Successfully switched to frame '{name_or_selector}' ({selector_detail})."

    def _run(
        self,
        name_or_selector: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        page = lc_get_current_page(self.sync_browser)
        if not page:
            return "No active page found. Ensure a page is navigated to first."
        return self._handle_switch_sync(page, name_or_selector)

    async def _arun(
        self,
        name_or_selector: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        page = await lc_aget_current_page(self.async_browser)
        if not page:
            return "No active page found. Ensure a page is navigated to first."
        return await self._handle_switch_async(page, name_or_selector)

