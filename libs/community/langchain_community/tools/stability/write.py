"""Stability Write Tool."""

import json
from typing import Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_community.utilities.stability import StabilityAPIWrapper


class StabilityWriteToolSchema(BaseModel):
    """Input for StabilityWriteTool."""
    
    arguments: str = Field(description="The message to send to the blockchain")


class StabilityWriteTool(BaseTool):
    """Tool for sending messages to the Stability blockchain using ZKT v1."""
    
    name: str = "StabilityWriteTool"
    description: str = (
        "Send a plain text message to the Stability blockchain using ZKT v1. "
        "This creates a permanent record on the blockchain."
    )
    args_schema: Type[BaseModel] = StabilityWriteToolSchema
    
    api_wrapper: StabilityAPIWrapper = Field(default_factory=StabilityAPIWrapper)
    
    def _run(
        self,
        arguments: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Execute the tool."""
        return self.api_wrapper.post_zkt_v1(arguments)
