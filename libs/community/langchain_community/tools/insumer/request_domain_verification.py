"""Tool for requesting merchant domain verification."""

from typing import Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_community.utilities.insumer import InsumerAPIWrapper


class InsumerRequestDomainVerificationSchema(BaseModel):
    """Input for InsumerRequestDomainVerification."""

    merchant_id: str = Field(description="Merchant ID to verify domain for.")
    domain: str = Field(
        description='Domain to verify (e.g. "example.com").',
    )


class InsumerRequestDomainVerification(BaseTool):
    """Request domain verification for a merchant.

    Returns a DNS TXT record that must be added to the domain.
    Owner only.
    """

    mode: str = "request_domain_verification"
    name: str = "insumer_request_domain_verification"
    description: str = (
        "Request domain verification for a merchant. Returns a DNS TXT record "
        "to add to the domain. After adding, call verify_domain to complete."
    )
    args_schema: Type[InsumerRequestDomainVerificationSchema] = (
        InsumerRequestDomainVerificationSchema
    )

    api_wrapper: InsumerAPIWrapper = Field(..., exclude=True)

    def __init__(self, api_wrapper: InsumerAPIWrapper) -> None:
        super().__init__(api_wrapper=api_wrapper)

    def _run(
        self,
        merchant_id: str,
        domain: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Request domain verification."""
        return self.api_wrapper.run(
            mode=self.mode,
            merchant_id=merchant_id,
            domain=domain,
        )
