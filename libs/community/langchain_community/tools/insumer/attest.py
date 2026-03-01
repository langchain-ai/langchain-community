"""Tool for creating privacy-preserving on-chain attestations."""

import json
from typing import Any, Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_community.utilities.insumer import InsumerAPIWrapper


class InsumerAttestSchema(BaseModel):
    """Input for InsumerAttest."""

    wallet: Optional[str] = Field(
        default=None,
        description="EVM wallet address (0x...) to verify.",
    )
    solana_wallet: Optional[str] = Field(
        default=None,
        description="Solana wallet address (base58) to verify.",
    )
    xrpl_wallet: Optional[str] = Field(
        default=None,
        description="XRPL wallet address (r-address) to verify.",
    )
    proof: Optional[str] = Field(
        default=None,
        description=(
            'Set to "merkle" to include EIP-1186 Merkle storage proofs in results. '
            "Proofs available for token_balance conditions on RPC chains "
            "(1, 56, 8453, 43114, 137, 42161, 10, 88888, 1868, 98866). "
            "Costs 2 credits instead of 1. Reveals raw balance to caller."
        ),
    )
    conditions: str = Field(
        description=(
            'JSON array of conditions. Each condition: {"type": "token_balance", '
            '"nft_ownership", or "eas_attestation", "contractAddress": "0x...", '
            '"chainId": 1, "threshold": 1000, "decimals": 6, "label": "..."}. '
            "For EAS attestations, use type \"eas_attestation\" with either "
            '"template": "coinbase_verified_account" (or coinbase_verified_country, '
            'coinbase_one) or raw "schemaId". No contractAddress/threshold needed. '
            "Supports 32 chains. Max 10 conditions."
        ),
    )


class InsumerAttest(BaseTool):
    """Verify on-chain token balances, NFT ownership, or EAS attestations.

    Returns only true/false per condition -- never exposes actual balances.
    The response includes an ECDSA P-256 signature for cryptographic proof.
    Costs 1 attestation credit per call, or 2 credits with proof="merkle".
    """

    mode: str = "attest"
    name: str = "insumer_attest"
    description: str = (
        "Verify on-chain conditions (token balances, NFT ownership, EAS "
        "attestations) across 32 blockchains. Returns a cryptographically signed "
        "true/false attestation without exposing actual wallet balances. For EAS "
        "attestations (e.g. Coinbase Verifications KYC), use compliance templates "
        "or raw schema IDs. Costs 1 credit. "
        'Pass proof="merkle" for EIP-1186 Merkle storage proofs (2 credits).'
    )
    args_schema: Type[InsumerAttestSchema] = InsumerAttestSchema

    api_wrapper: InsumerAPIWrapper = Field(..., exclude=True)

    def __init__(self, api_wrapper: InsumerAPIWrapper) -> None:
        super().__init__(api_wrapper=api_wrapper)

    def _run(
        self,
        conditions: str,
        wallet: Optional[str] = None,
        solana_wallet: Optional[str] = None,
        xrpl_wallet: Optional[str] = None,
        proof: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Execute the attestation."""
        parsed_conditions: list[dict[str, Any]] = json.loads(conditions)
        return self.api_wrapper.run(
            mode=self.mode,
            conditions=parsed_conditions,
            wallet=wallet,
            solana_wallet=solana_wallet,
            xrpl_wallet=xrpl_wallet,
            proof=proof,
        )
