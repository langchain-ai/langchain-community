"""Tool for getting the JWKS public signing key."""

from typing import Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_community.utilities.insumer import InsumerAPIWrapper


class InsumerJwksSchema(BaseModel):
    """Input for InsumerJwks. No parameters required."""

    pass


class InsumerJwks(BaseTool):
    """Get InsumerAPI's ECDSA P-256 public signing key in JWKS format.

    Match the ``kid`` from attestation and trust responses to verify
    signatures. No authentication required.
    """

    mode: str = "get_jwks"
    name: str = "insumer_jwks"
    description: str = (
        "Get the JWKS containing InsumerAPI's ECDSA P-256 public signing key. "
        "Match the kid field from attestation responses to verify signatures. "
        "No authentication or credits required."
    )
    args_schema: Type[InsumerJwksSchema] = InsumerJwksSchema

    api_wrapper: InsumerAPIWrapper = Field(..., exclude=True)

    def __init__(self, api_wrapper: InsumerAPIWrapper) -> None:
        super().__init__(api_wrapper=api_wrapper)

    def _run(
        self,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Get JWKS."""
        return self.api_wrapper.run(mode=self.mode)
