"""Tool for creating a new merchant."""

from typing import Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_community.utilities.insumer import InsumerAPIWrapper


class InsumerCreateMerchantSchema(BaseModel):
    """Input for InsumerCreateMerchant."""

    company_name: str = Field(
        description="Company display name (max 100 characters).",
        max_length=100,
    )
    company_id: str = Field(
        description="Unique merchant ID (2-50 chars, alphanumeric/dashes/underscores).",
        min_length=2,
        max_length=50,
    )
    location: Optional[str] = Field(
        default=None,
        description="City or region (max 200 characters).",
        max_length=200,
    )


class InsumerCreateMerchant(BaseTool):
    """Create a new merchant on InsumerAPI.

    Each merchant receives 100 free verification credits. Maximum 10
    merchants per API key. After creation, configure tokens/NFTs and
    publish to the directory.
    """

    mode: str = "create_merchant"
    name: str = "insumer_create_merchant"
    description: str = (
        "Create a new merchant on InsumerAPI. Provide company name and "
        "unique ID. Receives 100 free credits. Max 10 per API key."
    )
    args_schema: Type[InsumerCreateMerchantSchema] = InsumerCreateMerchantSchema

    api_wrapper: InsumerAPIWrapper = Field(..., exclude=True)

    def __init__(self, api_wrapper: InsumerAPIWrapper) -> None:
        super().__init__(api_wrapper=api_wrapper)

    def _run(
        self,
        company_name: str,
        company_id: str,
        location: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Create merchant."""
        return self.api_wrapper.run(
            mode=self.mode,
            company_name=company_name,
            company_id=company_id,
            location=location,
        )
