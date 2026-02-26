"""Tool for updating merchant settings."""

import json
from typing import Any, Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_community.utilities.insumer import InsumerAPIWrapper


class InsumerConfigureSettingsSchema(BaseModel):
    """Input for InsumerConfigureSettings."""

    merchant_id: str = Field(description="Merchant ID to configure settings for.")
    discount_mode: Optional[str] = Field(
        default=None,
        description='Discount stacking mode: "highest" or "stack".',
    )
    discount_cap: Optional[int] = Field(
        default=None,
        description="Maximum total discount percentage (1-100).",
        ge=1,
        le=100,
    )
    usdc_payment: Optional[str] = Field(
        default=None,
        description=(
            "JSON object for USDC payment settings, or null to disable. "
            '{"enabled": true, "evmAddress": "0x...", "solanaAddress": "...", '
            '"preferredChainId": 8453}. Pass as a JSON string.'
        ),
    )


class InsumerConfigureSettings(BaseTool):
    """Update merchant settings: discount mode, cap, USDC payments. Owner only.

    All fields are optional -- only provided fields are updated.
    """

    mode: str = "configure_settings"
    name: str = "insumer_configure_settings"
    description: str = (
        "Update merchant settings. Options: discount mode (highest/stack), "
        "discount cap (1-100%), and USDC payment configuration. "
        "All fields optional. Owner only."
    )
    args_schema: Type[InsumerConfigureSettingsSchema] = InsumerConfigureSettingsSchema

    api_wrapper: InsumerAPIWrapper = Field(..., exclude=True)

    def __init__(self, api_wrapper: InsumerAPIWrapper) -> None:
        super().__init__(api_wrapper=api_wrapper)

    def _run(
        self,
        merchant_id: str,
        discount_mode: Optional[str] = None,
        discount_cap: Optional[int] = None,
        usdc_payment: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Configure settings."""
        parsed_usdc: Optional[dict[str, Any]] = None
        if usdc_payment is not None:
            parsed_usdc = json.loads(usdc_payment)
        return self.api_wrapper.run(
            mode=self.mode,
            merchant_id=merchant_id,
            discount_mode=discount_mode,
            discount_cap=discount_cap,
            usdc_payment=parsed_usdc,
        )
