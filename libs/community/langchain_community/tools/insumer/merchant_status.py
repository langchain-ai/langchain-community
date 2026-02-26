"""Tool for getting private merchant status."""

from typing import Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_community.utilities.insumer import InsumerAPIWrapper


class InsumerMerchantStatusSchema(BaseModel):
    """Input for InsumerMerchantStatus."""

    merchant_id: str = Field(description="Merchant ID to check status for.")


class InsumerMerchantStatus(BaseTool):
    """Get full private merchant details. Owner only.

    Returns credits, token configurations, NFT collections, directory
    status, verification status, and USDC payment settings.
    """

    mode: str = "get_merchant_status"
    name: str = "insumer_merchant_status"
    description: str = (
        "Get full private merchant details: credits, token configs, NFT "
        "collections, directory status, and USDC payment settings. Owner only."
    )
    args_schema: Type[InsumerMerchantStatusSchema] = InsumerMerchantStatusSchema

    api_wrapper: InsumerAPIWrapper = Field(..., exclude=True)

    def __init__(self, api_wrapper: InsumerAPIWrapper) -> None:
        super().__init__(api_wrapper=api_wrapper)

    def _run(
        self,
        merchant_id: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Get merchant status."""
        return self.api_wrapper.run(mode=self.mode, merchant_id=merchant_id)
