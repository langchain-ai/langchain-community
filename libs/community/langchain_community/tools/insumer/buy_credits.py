"""Tool for buying verification credits with USDC."""

from typing import Any, Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_community.utilities.insumer import InsumerAPIWrapper


class InsumerBuyCreditsSchema(BaseModel):
    """Input for InsumerBuyCredits."""

    tx_hash: str = Field(description="USDC transaction hash.")
    chain_id: Any = Field(
        description=(
            "Chain where USDC was sent: 1 (Ethereum), 8453 (Base), "
            '137 (Polygon), 42161 (Arbitrum), 10 (Optimism), 56 (BNB), '
            '43114 (Avalanche), or "solana".'
        ),
    )
    amount: float = Field(
        description="USDC amount sent. Minimum 5.",
        ge=5,
    )


class InsumerBuyCredits(BaseTool):
    """Buy verification credits with USDC.

    Rate: 25 credits per 1 USDC ($0.04/credit). Minimum 5 USDC.
    Server verifies the on-chain transaction receipt.
    """

    mode: str = "buy_credits"
    name: str = "insumer_buy_credits"
    description: str = (
        "Buy verification credits for the API key by submitting a USDC "
        "transaction hash. Rate: 25 credits per 1 USDC. Minimum 5 USDC."
    )
    args_schema: Type[InsumerBuyCreditsSchema] = InsumerBuyCreditsSchema

    api_wrapper: InsumerAPIWrapper = Field(..., exclude=True)

    def __init__(self, api_wrapper: InsumerAPIWrapper) -> None:
        super().__init__(api_wrapper=api_wrapper)

    def _run(
        self,
        tx_hash: str,
        chain_id: Any,
        amount: float,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Buy credits."""
        return self.api_wrapper.run(
            mode=self.mode,
            tx_hash=tx_hash,
            chain_id=chain_id,
            amount=amount,
        )
