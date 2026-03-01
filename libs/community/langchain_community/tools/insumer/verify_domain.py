"""Tool for checking merchant domain verification status."""

from typing import Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_community.utilities.insumer import InsumerAPIWrapper


class InsumerVerifyDomainSchema(BaseModel):
    """Input for InsumerVerifyDomain."""

    merchant_id: str = Field(
        description="Merchant ID to check domain verification for.",
    )


class InsumerVerifyDomain(BaseTool):
    """Check domain verification status for a merchant.

    Call after adding the DNS TXT record from request_domain_verification.
    Owner only.
    """

    mode: str = "verify_domain"
    name: str = "insumer_verify_domain"
    description: str = (
        "Check domain verification status for a merchant. Call after adding "
        "the DNS TXT record from request_domain_verification."
    )
    args_schema: Type[InsumerVerifyDomainSchema] = InsumerVerifyDomainSchema

    api_wrapper: InsumerAPIWrapper = Field(..., exclude=True)

    def __init__(self, api_wrapper: InsumerAPIWrapper) -> None:
        super().__init__(api_wrapper=api_wrapper)

    def _run(
        self,
        merchant_id: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Check domain verification."""
        return self.api_wrapper.run(
            mode=self.mode,
            merchant_id=merchant_id,
        )
