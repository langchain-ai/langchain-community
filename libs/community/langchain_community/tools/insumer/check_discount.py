"""Tool for checking wallet discount eligibility at a merchant."""

from typing import Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_community.utilities.insumer import InsumerAPIWrapper


class InsumerCheckDiscountSchema(BaseModel):
    """Input for InsumerCheckDiscount."""

    merchant_id: str = Field(description="Merchant ID to check discount at.")
    wallet: Optional[str] = Field(
        default=None,
        description="EVM wallet address (0x...).",
    )
    solana_wallet: Optional[str] = Field(
        default=None,
        description="Solana wallet address (base58).",
    )
    xrpl_wallet: Optional[str] = Field(
        default=None,
        description="XRPL wallet address (r-address).",
    )


class InsumerCheckDiscount(BaseTool):
    """Calculate the discount a wallet qualifies for at a specific merchant.

    Checks on-chain balances server-side and returns the tier and discount
    percentage per token -- never raw balance amounts.
    Free to call, no credits consumed.
    """

    mode: str = "check_discount"
    name: str = "insumer_check_discount"
    description: str = (
        "Calculate what discount a wallet qualifies for at a specific merchant. "
        "Returns tier and discount percentage per token -- never raw balance "
        "amounts. Free to call, no credits consumed."
    )
    args_schema: Type[InsumerCheckDiscountSchema] = InsumerCheckDiscountSchema

    api_wrapper: InsumerAPIWrapper = Field(..., exclude=True)

    def __init__(self, api_wrapper: InsumerAPIWrapper) -> None:
        super().__init__(api_wrapper=api_wrapper)

    def _run(
        self,
        merchant_id: str,
        wallet: Optional[str] = None,
        solana_wallet: Optional[str] = None,
        xrpl_wallet: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Check the discount."""
        return self.api_wrapper.run(
            mode=self.mode,
            merchant_id=merchant_id,
            wallet=wallet,
            solana_wallet=solana_wallet,
            xrpl_wallet=xrpl_wallet,
        )
