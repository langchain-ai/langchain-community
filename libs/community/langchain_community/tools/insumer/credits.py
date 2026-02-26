"""Tool for checking attestation credit balance."""

from typing import Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_community.utilities.insumer import InsumerAPIWrapper


class InsumerCreditsSchema(BaseModel):
    """Input for InsumerCredits. No parameters required."""

    pass


class InsumerCredits(BaseTool):
    """Check the attestation credit balance for the current API key."""

    mode: str = "get_credits"
    name: str = "insumer_credits"
    description: str = (
        "Check the attestation credit balance, tier, and daily rate limit "
        "for the current Insumer API key. No parameters needed."
    )
    args_schema: Type[InsumerCreditsSchema] = InsumerCreditsSchema

    api_wrapper: InsumerAPIWrapper = Field(..., exclude=True)

    def __init__(self, api_wrapper: InsumerAPIWrapper) -> None:
        super().__init__(api_wrapper=api_wrapper)

    def _run(
        self,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Check credits."""
        return self.api_wrapper.run(mode=self.mode)
