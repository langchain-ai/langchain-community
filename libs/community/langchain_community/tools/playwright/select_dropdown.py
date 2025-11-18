from __future__ import annotations

from typing import List, Literal, Optional, Type

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


class SelectDropdownInput(BaseModel):
    """Input for SelectDropdownTool."""
    selector: str = Field(description="CSS selector for the <select> element.")
    value: List[str] = Field(
        description="The value(s), label(s), or index/indices to select. Must be a list."
    )
    select_by: Literal["value", "label", "index"] = Field(
        "value",
        description="Whether to select by 'value', 'label', or 'index'.",
    )


class SelectDropdownTool(BaseBrowserTool):
    name: str = "select_dropdown"
    description: str = (
        "Select one or more options from a <select> dropdown element "
        "by 'value', 'label', or 'index'."
    )
    args_schema: Type[BaseModel] = SelectDropdownInput

    def _run(
        self,
        selector: str,
        value: List[str],
        select_by: Literal["value", "label", "index"] = "value",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        page = get_current_page(self.sync_browser)
        try:
            options = {select_by: value}
            page.select_option(selector, **options)
            return f"Selected {value} in dropdown '{selector}' by {select_by}"
        except Exception as e:
            return f"SelectDropdown error: {str(e)}"

    async def _arun(
        self,
        selector: str,
        value: List[str],
        select_by: Literal["value", "label", "index"] = "value",
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        page = await aget_current_page(self.async_browser)
        try:
            options = {select_by: value}
            await page.select_option(selector, **options)
            return f"Selected {value} in dropdown '{selector}' by {select_by}"
        except Exception as e:
            return f"SelectDropdown async error: {str(e)}"
