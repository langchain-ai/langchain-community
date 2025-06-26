"""Stability Deploy Tool."""

import json
from typing import Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_community.utilities.stability import StabilityAPIWrapper


class StabilityDeployToolSchema(BaseModel):
    """Input for StabilityDeployTool."""
    
    arguments: str = Field(
        description="JSON string containing code, arguments for contract deployment"
    )


class StabilityDeployTool(BaseTool):
    """Tool for deploying Solidity contracts to the Stability blockchain."""
    
    name: str = "StabilityDeployTool"
    description: str = (
        "Deploy a Solidity smart contract to the Stability blockchain. "
        "Input must be JSON string with: code, arguments"
    )
    args_schema: Type[BaseModel] = StabilityDeployToolSchema
    
    api_wrapper: StabilityAPIWrapper = Field(default_factory=StabilityAPIWrapper)
    
    def _run(
        self,
        arguments: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Execute the tool."""
        try:
            params = json.loads(arguments)
            return self.api_wrapper.deploy_contract(**params)
        except json.JSONDecodeError as e:
            return f"Error: Invalid JSON input: {e}"
        except Exception as e:
            return f"Error: {e}"
