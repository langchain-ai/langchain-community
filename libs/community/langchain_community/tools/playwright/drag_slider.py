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


class DragSliderInput(BaseModel):
    selector: str = Field(..., description="CSS selector for the slider input element")
    x_offset: int = Field(..., description="Horizontal pixels to drag the slider knob")


class DragSliderTool(BaseBrowserTool):
    name: str = "drag_slider"
    description: str = "Drag a slider knob horizontally by an offset"
    args_schema: Type[BaseModel] = DragSliderInput

    def _run(
        self,
        selector: str,
        x_offset: int,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        page = get_current_page(self.sync_browser)
        try:
            box = page.locator(selector).bounding_box()
            page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
            page.mouse.down()
            page.mouse.move(
                box["x"] + box["width"] / 2 + x_offset, box["y"] + box["height"] / 2
            )
            page.mouse.up()
            return f"Dragged slider '{selector}' by {x_offset}px"
        except Exception as e:
            return f"DragSlider error: {str(e)}"

    async def _arun(
        self,
        selector: str,
        x_offset: int,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        page = await aget_current_page(self.async_browser)
        try:
            box = await page.locator(selector).bounding_box()
            await page.mouse.move(
                box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
            )
            await page.mouse.down()
            await page.mouse.move(
                box["x"] + box["width"] / 2 + x_offset, box["y"] + box["height"] / 2
            )
            await page.mouse.up()
            return f"Dragged slider '{selector}' by {x_offset}px"
        except Exception as e:
            return f"DragSlider async error: {str(e)}"
