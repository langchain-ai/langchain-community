"""Tool for batch wallet trust profiling."""

import json
from typing import Any, Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_community.utilities.insumer import InsumerAPIWrapper


class InsumerBatchWalletTrustSchema(BaseModel):
    """Input for InsumerBatchWalletTrust."""

    wallets: str = Field(
        description=(
            "JSON array of 1-10 wallet objects. Each: "
            '{"wallet": "0x...", "solanaWallet": "..."}. '
            "solanaWallet is optional. Pass as a JSON string."
        ),
    )
    proof: Optional[str] = Field(
        default=None,
        description='Set to "merkle" for Merkle proofs (6 credits/wallet).',
    )


class InsumerBatchWalletTrust(BaseTool):
    """Generate trust profiles for up to 10 wallets in one request.

    5-8x faster than sequential calls via shared block fetches.
    Supports partial success. Costs 3 credits per wallet (6 with merkle).
    """

    mode: str = "batch_wallet_trust"
    name: str = "insumer_batch_wallet_trust"
    description: str = (
        "Generate trust profiles for up to 10 wallets in a single request. "
        "5-8x faster than sequential calls. Partial success supported. "
        "Costs 3 credits per wallet (6 with merkle proofs)."
    )
    args_schema: Type[InsumerBatchWalletTrustSchema] = InsumerBatchWalletTrustSchema

    api_wrapper: InsumerAPIWrapper = Field(..., exclude=True)

    def __init__(self, api_wrapper: InsumerAPIWrapper) -> None:
        super().__init__(api_wrapper=api_wrapper)

    def _run(
        self,
        wallets: str,
        proof: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Batch trust profiles."""
        parsed_wallets: list[dict[str, Any]] = json.loads(wallets)
        return self.api_wrapper.run(
            mode=self.mode,
            wallets=parsed_wallets,
            proof=proof,
        )
