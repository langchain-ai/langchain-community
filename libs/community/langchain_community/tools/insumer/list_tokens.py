"""Tool for listing registered tokens and NFT collections."""

from typing import Any, Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_community.utilities.insumer import InsumerAPIWrapper


class InsumerListTokensSchema(BaseModel):
    """Input for InsumerListTokens."""

    chain: Optional[Any] = Field(
        default=None,
        description='Filter by chain ID (e.g. 1 for Ethereum, "solana" for Solana).',
    )
    symbol: Optional[str] = Field(
        default=None,
        description="Filter by token symbol (e.g. UNI).",
    )
    asset_type: Optional[str] = Field(
        default=None,
        description='Filter by type: "token" or "nft".',
    )


class InsumerListTokens(BaseTool):
    """List tokens and NFT collections registered with merchants."""

    mode: str = "list_tokens"
    name: str = "insumer_list_tokens"
    description: str = (
        "List all tokens and NFT collections registered in The Insumer Model "
        "ecosystem. Filter by blockchain, symbol, or asset type (token/nft). "
        "Returns contract addresses, chain IDs, and metadata."
    )
    args_schema: Type[InsumerListTokensSchema] = InsumerListTokensSchema

    api_wrapper: InsumerAPIWrapper = Field(..., exclude=True)

    def __init__(self, api_wrapper: InsumerAPIWrapper) -> None:
        super().__init__(api_wrapper=api_wrapper)

    def _run(
        self,
        chain: Optional[Any] = None,
        symbol: Optional[str] = None,
        asset_type: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """List tokens."""
        return self.api_wrapper.run(
            mode=self.mode,
            chain=chain,
            symbol=symbol,
            asset_type=asset_type,
        )
