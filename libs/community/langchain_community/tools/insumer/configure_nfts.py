"""Tool for configuring merchant NFT collection discounts."""

import json
from typing import Any, Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_community.utilities.insumer import InsumerAPIWrapper


class InsumerConfigureNftsSchema(BaseModel):
    """Input for InsumerConfigureNfts."""

    merchant_id: str = Field(description="Merchant ID to configure NFTs for.")
    nft_collections: str = Field(
        description=(
            "JSON array of NFT collection configurations (0-4). Each: "
            '{"name": "Collection Name", "contractAddress": "0x...", '
            '"chainId": 1, "discount": 10}. Discount is 1-50%. '
            "Pass as a JSON string."
        ),
    )


class InsumerConfigureNfts(BaseTool):
    """Configure NFT collections that grant discounts at a merchant. Owner only.

    Max 4 NFT collections. Each specifies contract address, chain, and
    flat discount percentage (1-50%).
    """

    mode: str = "configure_nfts"
    name: str = "insumer_configure_nfts"
    description: str = (
        "Configure NFT collections that grant discounts at a merchant. "
        "Max 4 collections. Each specifies contract, chain, and discount "
        "percentage (1-50%). Owner only."
    )
    args_schema: Type[InsumerConfigureNftsSchema] = InsumerConfigureNftsSchema

    api_wrapper: InsumerAPIWrapper = Field(..., exclude=True)

    def __init__(self, api_wrapper: InsumerAPIWrapper) -> None:
        super().__init__(api_wrapper=api_wrapper)

    def _run(
        self,
        merchant_id: str,
        nft_collections: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Configure NFTs."""
        parsed: list[dict[str, Any]] = json.loads(nft_collections)
        return self.api_wrapper.run(
            mode=self.mode,
            merchant_id=merchant_id,
            nft_collections=parsed,
        )
