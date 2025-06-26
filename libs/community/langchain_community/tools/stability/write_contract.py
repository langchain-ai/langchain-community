"""Stability Write Contract Tool."""

import json
from typing import Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_community.utilities.stability import StabilityAPIWrapper


class StabilityWriteContractToolSchema(BaseModel):
    """Input for StabilityWriteContractTool."""
    
    arguments: str = Field(
        description="JSON string containing to, abi, method, arguments, id, wait fields for contract write"
    )


class StabilityWriteContractTool(BaseTool):
    """Tool for writing data to Stability smart contracts."""
    
    name: str = "StabilityWriteContractTool"
    description: str = (
        "Write data to a Stability smart contract using ZKT v2 write request. "
        "Input must be JSON string with: to, abi, method, arguments, id, wait"
    )
    args_schema: Type[BaseModel] = StabilityWriteContractToolSchema
    
    api_wrapper: StabilityAPIWrapper = Field(default_factory=StabilityAPIWrapper)
    
    def _run(
        self,
        arguments: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Execute the tool."""
        try:
            params = json.loads(arguments)
            return self.api_wrapper.call_contract_write(**params)
        except json.JSONDecodeError as e:
            return f"Error: Invalid JSON input: {e}"
        except Exception as e:
            return f"Error: {e}"
