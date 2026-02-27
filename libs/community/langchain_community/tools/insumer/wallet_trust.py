"""Tool for generating wallet trust fact profiles."""

from typing import Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_community.utilities.insumer import InsumerAPIWrapper


class InsumerWalletTrustSchema(BaseModel):
    """Input for InsumerWalletTrust."""

    wallet: str = Field(
        description="EVM wallet address (0x...) to profile.",
    )
    solana_wallet: Optional[str] = Field(
        default=None,
        description="Solana wallet address (base58). Adds USDC on Solana check.",
    )
    proof: Optional[str] = Field(
        default=None,
        description=(
            'Set to "merkle" for EIP-1186 Merkle storage proofs. '
            "Costs 6 credits instead of 3."
        ),
    )


class InsumerWalletTrust(BaseTool):
    """Generate a structured, ECDSA-signed wallet trust fact profile.

    Checks 17 curated conditions across stablecoins (7 chains), governance
    tokens (4), NFTs (3), and staking (3). Returns per-dimension pass/fail counts.
    Costs 3 credits (standard) or 6 credits (with proof="merkle").
    """

    mode: str = "wallet_trust"
    name: str = "insumer_wallet_trust"
    description: str = (
        "Generate a wallet trust fact profile across stablecoins, governance "
        "tokens, NFTs, and staking. Returns 17 curated checks organized by "
        "dimension with ECDSA-signed evidence. Costs 3 credits (6 with merkle proofs)."
    )
    args_schema: Type[InsumerWalletTrustSchema] = InsumerWalletTrustSchema

    api_wrapper: InsumerAPIWrapper = Field(..., exclude=True)

    def __init__(self, api_wrapper: InsumerAPIWrapper) -> None:
        super().__init__(api_wrapper=api_wrapper)

    def _run(
        self,
        wallet: str,
        solana_wallet: Optional[str] = None,
        proof: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Generate trust profile."""
        return self.api_wrapper.run(
            mode=self.mode,
            wallet=wallet,
            solana_wallet=solana_wallet,
            proof=proof,
        )
