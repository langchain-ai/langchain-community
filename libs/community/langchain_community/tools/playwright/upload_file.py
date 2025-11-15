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


class UploadFileInput(BaseModel):
    selector: str = Field(..., description="CSS selector for file input element")
    file_path: str = Field(..., description="Path to the file to upload")


class UploadFileTool(BaseBrowserTool):
    name: str = "upload_file"
    description: str = "Upload a file to an <input type='file'> element"
    args_schema: Type[BaseModel] = UploadFileInput

    def _run(
        self,
        selector: str,
        file_path: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        page = get_current_page(self.sync_browser)
        try:
            page.set_input_files(selector, file_path)
            return f"Uploaded file '{file_path}' to '{selector}'"
        except Exception as e:
            return f"UploadFile error: {str(e)}"

    async def _arun(
        self,
        selector: str,
        file_path: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        page = await aget_current_page(self.async_browser)
        try:
            await page.set_input_files(selector, file_path)
            return f"Uploaded file '{file_path}' to '{selector}'"
        except Exception as e:
            return f"UploadFile async error: {str(e)}"
