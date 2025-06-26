"""Stability Toolkit implementation."""

from typing import List

from langchain_core.tools import BaseToolkit, BaseTool
from pydantic import Field

from langchain_community.tools.stability import (
    StabilityDeployTool,
    StabilityReadTool,
    StabilityWriteContractTool,
    StabilityWriteTool,
)
from langchain_community.utilities.stability import StabilityAPIWrapper


class StabilityToolkit(BaseToolkit):
    """Stability Blockchain Toolkit for LangChain.
    
    This toolkit provides AI agents with access to the Stability blockchain
    through Zero Gas Transaction (ZKT) API endpoints.
    
    Args:
        api_key: Stability API key. If not provided, will use STABILITY_API_KEY 
                environment variable or default to "try-it-out"
    
    Environment Variables:
        STABILITY_API_KEY: Your Stability API key (recommended for production)
    
    Getting Your FREE API Key:
        Visit https://portal.stabilityprotocol.com/ to get your free API key.
        Free tier includes:
        - Up to 3 API keys per account
        - 1,000 write transactions per month  
        - 200 read operations per minute
        - Completely free access
    
    Support:
        Email: contact@stabilityprotocol.com
        Portal: https://portal.stabilityprotocol.com/
    
    Example:
        # Using environment variable (recommended)
        export STABILITY_API_KEY="your-api-key-from-portal"
        toolkit = StabilityToolkit()
        
        # Or passing directly
        toolkit = StabilityToolkit(api_key="your-api-key-from-portal")
        
        # Development/testing (limited functionality)
        toolkit = StabilityToolkit()  # Uses "try-it-out" key
    """
    
    api_wrapper: StabilityAPIWrapper = Field(default_factory=StabilityAPIWrapper)
    
    def __init__(self, api_key: str | None = None, **kwargs):
        """Initialize the Stability toolkit.
        
        Args:
            api_key: Stability API key. If None, uses environment variable
                    STABILITY_API_KEY or defaults to "try-it-out"
        """
        api_wrapper = StabilityAPIWrapper(api_key=api_key)
        super().__init__(api_wrapper=api_wrapper, **kwargs)
        
        # Log API key status (sanitized)
        if self.api_wrapper.api_key == "try-it-out":
            print("🔧 Stability Toolkit initialized with 'try-it-out' key (limited functionality)")
            print("   Get your FREE production API key at: https://portal.stabilityprotocol.com/")
        else:
            from langchain_community.utilities.stability import _sanitize_api_key_for_logging
            print(f"🔧 Stability Toolkit initialized with API key: {_sanitize_api_key_for_logging(self.api_wrapper.api_key)}")
    
    def get_tools(self) -> List[BaseTool]:
        """Get all Stability tools configured with this toolkit's API wrapper."""
        return [
            StabilityWriteTool(api_wrapper=self.api_wrapper),
            StabilityReadTool(api_wrapper=self.api_wrapper),
            StabilityWriteContractTool(api_wrapper=self.api_wrapper),
            StabilityDeployTool(api_wrapper=self.api_wrapper),
        ]
