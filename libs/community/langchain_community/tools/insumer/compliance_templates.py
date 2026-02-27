"""Tool for listing available EAS compliance templates."""

from typing import Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_community.utilities.insumer import InsumerAPIWrapper


class InsumerComplianceTemplatesSchema(BaseModel):
    """Input for InsumerComplianceTemplates."""


class InsumerComplianceTemplates(BaseTool):
    """List available EAS compliance templates for attestation verification.

    Templates provide pre-configured schema IDs for KYC/identity providers
    (e.g. Coinbase Verifications on Base). Free, no auth or credits required.
    """

    mode: str = "get_compliance_templates"
    name: str = "insumer_compliance_templates"
    description: str = (
        "List available EAS compliance templates for attestation verification. "
        "Templates provide pre-configured schema IDs for KYC/identity providers "
        "like Coinbase Verifications on Base. Use these template names with the "
        'insumer_attest tool (type "eas_attestation", template "coinbase_verified_account"). '
        "Free, no credits required."
    )
    args_schema: Type[InsumerComplianceTemplatesSchema] = (
        InsumerComplianceTemplatesSchema
    )

    api_wrapper: InsumerAPIWrapper = Field(..., exclude=True)

    def __init__(self, api_wrapper: InsumerAPIWrapper) -> None:
        super().__init__(api_wrapper=api_wrapper)

    def _run(
        self,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """List compliance templates."""
        return self.api_wrapper.run(mode=self.mode)
