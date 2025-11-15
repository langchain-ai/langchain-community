from __future__ import annotations

from typing import Any, Dict, List, Optional, Type, Union

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools.base import BaseTool
from pydantic import BaseModel, Field


class OutputToFileToolInput(BaseModel):
    file_path: str = Field(..., description="Path to the output file")
    data: Union[List[Dict[str, Any]], Dict[str, Any], str] = Field(
        ..., description="Data to write (list of dicts for CSV/JSON, or string for TXT)"
    )
    format: str = Field(..., description="Format: 'csv', 'json', or 'txt'")


class OutputToFileTool(BaseTool):
    name: str = "output_to_file"
    description: str = "Write data to a file in csv, json, or txt format."
    args_schema: Type[BaseModel] = OutputToFileToolInput

    def _run(
        self,
        file_path: str,
        data: Union[List[Dict[str, Any]], Dict[str, Any], str],
        format: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        fmt = format.lower()
        try:
            # CSV
            if fmt == "csv":
                import csv

                if (
                    not isinstance(data, list)
                    or not data
                    or not all(isinstance(item, dict) for item in data)
                ):
                    return "CSV format requires a non-empty list of dictionaries."
                with open(file_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=list(data[0].keys()))
                    writer.writeheader()
                    writer.writerows(data)

            # JSON
            elif fmt == "json":
                import json

                obj: Any = data
                if isinstance(data, str):
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        return "Invalid JSON string provided."
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(obj, f, ensure_ascii=False, indent=2)

            # TXT
            elif fmt == "txt":
                text = data if isinstance(data, str) else repr(data)
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(text)

            else:
                return f"Unsupported format: {format}"

            return f"Data written to {file_path}"

        except Exception as e:
            return f"OutputToFileTool error: {e}"

    async def _arun(
        self,
        file_path: str,
        data: Union[List[Dict[str, Any]], Dict[str, Any], str],
        format: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        # Delegate to synchronous implementation
        return self._run(file_path, data, format)
