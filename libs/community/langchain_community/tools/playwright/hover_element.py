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


class HoverElementInput(BaseModel):
    selector: str = Field(..., description="CSS selector of the element to hover")


class HoverElementTool(BaseBrowserTool):
    name: str = "hover_element"
    description: str = "Hover over an element using a CSS selector"
    args_schema: Type[BaseModel] = HoverElementInput

    def _run(
        self, selector: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        page = get_current_page(self.sync_browser)
        try:
            page.hover(selector)
            return f"Hovered over element '{selector}'"
        except Exception as e:
            return f"HoverElement error: {str(e)}"

    async def _arun(
        self,
        selector: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        page = await aget_current_page(self.async_browser)
        try:
            await page.hover(selector)
            return f"Hovered over element '{selector}'"
        except Exception as e:
            return f"HoverElement async error: {str(e)}"
