"""Tool for requesting UCP-format discounts."""

import json
from typing import Any, Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_community.utilities.insumer import InsumerAPIWrapper


class InsumerUcpDiscountSchema(BaseModel):
    """Input for InsumerUcpDiscount."""

    merchant_id: str = Field(description="Merchant ID to request discount from.")
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
    items: Optional[str] = Field(
        default=None,
        description=(
            "JSON array of cart items for per-item allocation. "
            'Each item: {"path": "cart/item-name", "amount": 450}. '
            "Amount is in cents. Pass as a JSON string."
        ),
    )


class InsumerUcpDiscount(BaseTool):
    """Request a UCP-format discount for a wallet at a merchant.

    Returns Google Universal Commerce Protocol discount objects with
    title strings and the dev.ucp.shopping.discount extension.
    Costs 1 merchant credit.
    """

    mode: str = "ucp_discount"
    name: str = "insumer_ucp_discount"
    description: str = (
        "Request a UCP (Universal Commerce Protocol) discount for a wallet "
        "at a specific merchant. Returns discount objects with title, code, "
        "and optional per-item allocations. Costs 1 merchant credit."
    )
    args_schema: Type[InsumerUcpDiscountSchema] = InsumerUcpDiscountSchema

    api_wrapper: InsumerAPIWrapper = Field(..., exclude=True)

    def __init__(self, api_wrapper: InsumerAPIWrapper) -> None:
        super().__init__(api_wrapper=api_wrapper)

    def _run(
        self,
        merchant_id: str,
        wallet: Optional[str] = None,
        solana_wallet: Optional[str] = None,
        xrpl_wallet: Optional[str] = None,
        items: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Request UCP discount."""
        parsed_items: list[dict[str, Any]] | None = None
        if items:
            parsed_items = json.loads(items)
        return self.api_wrapper.run(
            mode=self.mode,
            merchant_id=merchant_id,
            wallet=wallet,
            solana_wallet=solana_wallet,
            xrpl_wallet=xrpl_wallet,
            items=parsed_items,
        )
