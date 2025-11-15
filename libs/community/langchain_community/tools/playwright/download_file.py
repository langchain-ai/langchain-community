from __future__ import annotations

from typing import Optional, Type

import httpx
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools.base import BaseTool
from pydantic import BaseModel, Field


class DownloadFileToolInput(BaseModel):
    url: str = Field(..., description="URL of the file to download")
    dest_path: str = Field(..., description="Local path where the file will be saved")
    timeout: Optional[float] = Field(10.0, description="Download timeout in seconds")


class DownloadFileTool(BaseTool):
    """
    Download a file from a given URL and save it to a local path.
    """

    name: str = "download_file"
    description: str = "Download a file from the specified URL and save to dest_path."
    args_schema: Type[BaseModel] = DownloadFileToolInput

    def _run(
        self,
        url: str,
        dest_path: str,
        timeout: Optional[float] = 10.0,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        try:
            response = httpx.get(url, timeout=timeout)
            response.raise_for_status()
            with open(dest_path, "wb") as f:
                f.write(response.content)
            return f"Downloaded file from {url} to {dest_path}"
        except Exception as e:
            return f"DownloadFileTool error: {str(e)}"

    async def _arun(
        self,
        url: str,
        dest_path: str,
        timeout: Optional[float] = 10.0,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=timeout)
                response.raise_for_status()
                with open(dest_path, "wb") as f:
                    f.write(response.content)
            return f"Downloaded file from {url} to {dest_path}"
        except Exception as e:
            return f"DownloadFileTool async error: {str(e)}"
