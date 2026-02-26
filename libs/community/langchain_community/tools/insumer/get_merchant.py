"""Tool for getting a merchant's public profile."""

from typing import Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_community.utilities.insumer import InsumerAPIWrapper


class InsumerGetMerchantSchema(BaseModel):
    """Input for InsumerGetMerchant."""

    merchant_id: str = Field(description="Merchant ID to look up.")


class InsumerGetMerchant(BaseTool):
    """Get the full public profile of a merchant.

    Returns token tiers, NFT collections, discount mode, and verification
    status. No credits consumed.
    """

    mode: str = "get_merchant"
    name: str = "insumer_get_merchant"
    description: str = (
        "Get the full public profile of a specific merchant including token "
        "discount tiers, NFT collections, discount mode, and verification "
        "status. Use when you know the merchant ID."
    )
    args_schema: Type[InsumerGetMerchantSchema] = InsumerGetMerchantSchema

    api_wrapper: InsumerAPIWrapper = Field(..., exclude=True)

    def __init__(self, api_wrapper: InsumerAPIWrapper) -> None:
        super().__init__(api_wrapper=api_wrapper)

    def _run(
        self,
        merchant_id: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Get merchant profile."""
        return self.api_wrapper.run(mode=self.mode, merchant_id=merchant_id)
