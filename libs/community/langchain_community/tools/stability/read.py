"""Stability Read Tool."""

import json
from typing import Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_community.utilities.stability import StabilityAPIWrapper


class StabilityReadToolSchema(BaseModel):
    """Input for StabilityReadTool."""
    
    arguments: str = Field(
        description="JSON string containing to, abi, method, arguments fields for contract read"
    )


class StabilityReadTool(BaseTool):
    """Tool for reading data from Stability smart contracts."""
    
    name: str = "StabilityReadTool"
    description: str = (
        "Read data from a Stability smart contract using ZKT v2 read request. "
        "Input must be JSON string with: to, abi, method, arguments"
    )
    args_schema: Type[BaseModel] = StabilityReadToolSchema
    
    api_wrapper: StabilityAPIWrapper = Field(default_factory=StabilityAPIWrapper)
    
    def _run(
        self,
        arguments: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Execute the tool."""
        try:
            params = json.loads(arguments)
            return self.api_wrapper.call_contract_read(**params)
        except json.JSONDecodeError as e:
            return f"Error: Invalid JSON input: {e}"
        except Exception as e:
            return f"Error: {e}"
