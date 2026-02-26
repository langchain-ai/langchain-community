"""Tool for configuring merchant token discount tiers."""

import json
from typing import Any, Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_community.utilities.insumer import InsumerAPIWrapper


class InsumerConfigureTokensSchema(BaseModel):
    """Input for InsumerConfigureTokens."""

    merchant_id: str = Field(description="Merchant ID to configure tokens for.")
    own_token: Optional[str] = Field(
        default=None,
        description=(
            "JSON object for the merchant's own token, or null to remove. "
            'Format: {"symbol": "TOKEN", "chainId": 1, "contractAddress": "0x...", '
            '"decimals": 18, "tiers": [{"name": "Gold", "threshold": 1000, '
            '"discount": 10}]}. Pass as a JSON string.'
        ),
    )
    partner_tokens: Optional[str] = Field(
        default=None,
        description=(
            "JSON array of partner token configurations. Same format as "
            "own_token but as an array. Max 8 tokens total. Pass as JSON string."
        ),
    )


class InsumerConfigureTokens(BaseTool):
    """Configure token discount tiers for a merchant. Owner only.

    Set the merchant's own token and/or partner tokens with balance
    threshold tiers and discount percentages. Max 8 tokens total.
    """

    mode: str = "configure_tokens"
    name: str = "insumer_configure_tokens"
    description: str = (
        "Configure token discount tiers for a merchant. Set own token "
        "and/or partner tokens with balance thresholds and discount "
        "percentages. Max 8 tokens total. Owner only."
    )
    args_schema: Type[InsumerConfigureTokensSchema] = InsumerConfigureTokensSchema

    api_wrapper: InsumerAPIWrapper = Field(..., exclude=True)

    def __init__(self, api_wrapper: InsumerAPIWrapper) -> None:
        super().__init__(api_wrapper=api_wrapper)

    def _run(
        self,
        merchant_id: str,
        own_token: Optional[str] = None,
        partner_tokens: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Configure tokens."""
        parsed_own: Optional[dict[str, Any]] = None
        parsed_partners: Optional[list[dict[str, Any]]] = None
        if own_token is not None:
            parsed_own = json.loads(own_token)
        if partner_tokens is not None:
            parsed_partners = json.loads(partner_tokens)
        return self.api_wrapper.run(
            mode=self.mode,
            merchant_id=merchant_id,
            own_token=parsed_own,
            partner_tokens=parsed_partners,
        )
