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


class DragAndDropToolInput(BaseModel):
    source_selector: str = Field(
        ..., description="CSS selector for the source element to drag"
    )
    target_selector: str = Field(
        ..., description="CSS selector for the target element to drop onto"
    )
    source_nth: Optional[int] = Field(
        0, description="Index of source element when multiple matches"
    )
    target_nth: Optional[int] = Field(
        0, description="Index of target element when multiple matches"
    )


class DragAndDropTool(BaseBrowserTool):
    name: str = Field("drag_and_drop", description="Unique tool name.")
    description: str = Field(
        "Drag an element matched by source_selector and drop it onto target_selector.",
        description="Tool description",
    )
    args_schema: Type[BaseModel] = DragAndDropToolInput

    visible_only: bool = True
    playwright_strict: bool = False
    playwright_timeout: float = 5_000

    def _run(
        self,
        source_selector: str,
        target_selector: str,
        source_nth: Optional[int] = 0,
        target_nth: Optional[int] = 0,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

        if self.sync_browser is None:
            raise ValueError("Synchronous browser not initialized.")
        page = get_current_page(self.sync_browser)
        page.wait_for_load_state("load")

        try:
            page.wait_for_selector(source_selector, timeout=self.playwright_timeout)
            page.wait_for_selector(target_selector, timeout=self.playwright_timeout)
            src_locator = page.locator(source_selector)
            tgt_locator = page.locator(target_selector)

            if src_locator.count() == 0:
                return f"No source element found for selector: {source_selector}"
            if tgt_locator.count() == 0:
                return f"No target element found for selector: {target_selector}"

            src = src_locator.nth(source_nth or 0)
            tgt = tgt_locator.nth(target_nth or 0)
            src.scroll_into_view_if_needed(timeout=self.playwright_timeout)
            tgt.scroll_into_view_if_needed(timeout=self.playwright_timeout)

            if self.visible_only and (
                not src.is_visible(timeout=self.playwright_timeout / 2)
                or not tgt.is_visible(timeout=self.playwright_timeout / 2)
            ):
                return "Element found but not visible. Try disabling visible_only."

            src.drag_to(tgt, timeout=self.playwright_timeout)
            return (
                f"Dragged element '{source_selector}' (nth={source_nth}) to "
                f"'{target_selector}' (nth={target_nth})"
            )

        except PlaywrightTimeoutError:
            return (
                f"Timeout waiting for element(s) '{source_selector}' or "
                f"'{target_selector}'"
            )
        except Exception as e:
            return f"DragAndDropTool error: {e}"

    async def _arun(
        self,
        source_selector: str,
        target_selector: str,
        source_nth: Optional[int] = 0,
        target_nth: Optional[int] = 0,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        from playwright.async_api import TimeoutError as PlaywrightTimeoutError

        if self.async_browser is None:
            raise ValueError("Asynchronous browser not initialized.")

        page = await aget_current_page(self.async_browser)

        try:
            await page.wait_for_selector(
                source_selector, timeout=self.playwright_timeout
            )
            await page.wait_for_selector(
                target_selector, timeout=self.playwright_timeout
            )
            src_locator = page.locator(source_selector)
            tgt_locator = page.locator(target_selector)
            count_src = await src_locator.count()
            count_tgt = await tgt_locator.count()

            if count_src == 0:
                return f"No source element found for selector: {source_selector}"
            if count_tgt == 0:
                return f"No target element found for selector: {target_selector}"

            src = src_locator.nth(source_nth or 0)
            tgt = tgt_locator.nth(target_nth or 0)
            await src.scroll_into_view_if_needed(timeout=self.playwright_timeout)
            await tgt.scroll_into_view_if_needed(timeout=self.playwright_timeout)

            if self.visible_only and (
                not await src.is_visible(timeout=self.playwright_timeout / 2)
                or not await tgt.is_visible(timeout=self.playwright_timeout / 2)
            ):
                return "Element found but not visible. Try disabling visible_only."

            await src.drag_to(tgt, timeout=self.playwright_timeout)
            return (
                f"Dragged element '{source_selector}' (nth={source_nth}) to "
                f"'{target_selector}' (nth={target_nth})"
            )

        except PlaywrightTimeoutError:
            return (
                f"Timeout waiting for element(s) '{source_selector}' or "
                f"'{target_selector}'"
            )
        except Exception as e:
            return f"DragAndDropTool async error: {e}"
