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


class SelectDropdownInput(BaseModel):
    selector: str = Field(..., description="CSS selector for the <select> element")
    value: Union[str, List[str]] = Field(
        ...,
        description=(
            "Value or list of values to select. These should match the 'value' "
            "attributes of <option> elements."
        ),
    )


class SelectDropdownTool(BaseBrowserTool):
    name: str = "select_dropdown"
    description: str = (
        "Select one or more options from a <select> dropdown element by value."
    )
    args_schema: Type[BaseModel] = SelectDropdownInput

    def _run(
        self,
        selector: str,
        value: Union[str, List[str]],
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        page = get_current_page(self.sync_browser)
        try:
            values = value if isinstance(value, list) else [value]
            page.select_option(selector, values)
            return f"Selected {values} in dropdown '{selector}'"
        except Exception as e:
            return f"SelectDropdown error: {str(e)}"

    async def _arun(
        self,
        selector: str,
        value: Union[str, List[str]],
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        page = await aget_current_page(self.async_browser)
        try:
            values = value if isinstance(value, list) else [value]
            await page.select_option(selector, values)
            return f"Selected {values} in dropdown '{selector}'"
        except Exception as e:
            return f"SelectDropdown async error: {str(e)}"
