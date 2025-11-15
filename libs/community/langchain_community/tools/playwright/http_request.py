from __future__ import annotations

from typing import Any, Dict, Optional, Type

import httpx
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools.base import BaseTool
from pydantic import BaseModel


class HttpRequestToolInput(BaseModel):
    method: str
    url: str
    headers: Optional[Dict[str, Any]] = None
    params: Optional[Dict[str, Any]] = None
    json_body: Optional[Any] = None  # ← renamed
    data: Optional[Any] = None


class HttpRequestTool(BaseTool):
    name: str = "http_request"
    description: str = "Make HTTP requests using httpx..."
    args_schema: Type[BaseModel] = HttpRequestToolInput

    def _run(
        self,
        method: str,
        url: str,
        headers: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Any] = None,  # ← renamed
        data: Optional[Any] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.request(
                    method=method,
                    url=url,
                    headers=headers,
                    params=params,
                    json=json_body,
                    data=data,
                )
            return f"Status {resp.status_code}: {resp.text}"
        except Exception as e:
            return f"HttpRequestTool error: {e}"

    async def _arun(
        self,
        method: str,
        url: str,
        headers: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Any] = None,
        data: Optional[Any] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.request(
                    method=method,
                    url=url,
                    headers=headers,
                    params=params,
                    json=json_body,
                    data=data,
                )
            return f"Status {resp.status_code}: {resp.text}"
        except Exception as e:
            return f"HttpRequestTool async error: {e}"
