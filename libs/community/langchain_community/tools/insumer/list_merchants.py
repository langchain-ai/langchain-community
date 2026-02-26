"""Tool for listing merchants in the public directory."""

from typing import Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_community.utilities.insumer import InsumerAPIWrapper


class InsumerListMerchantsSchema(BaseModel):
    """Input for InsumerListMerchants."""

    token: Optional[str] = Field(
        default=None,
        description="Filter by accepted token symbol (e.g. UNI, SHIB).",
    )
    verified: Optional[bool] = Field(
        default=None,
        description="Filter by domain verification status.",
    )
    limit: int = Field(
        default=50,
        description="Results per page (max 200).",
    )
    offset: int = Field(
        default=0,
        description="Pagination offset.",
    )


class InsumerListMerchants(BaseTool):
    """Browse merchants that offer token-gated discounts."""

    mode: str = "list_merchants"
    name: str = "insumer_list_merchants"
    description: str = (
        "List merchants in The Insumer Model directory that offer discounts "
        "to token holders. Optionally filter by accepted token symbol or "
        "verification status. Returns merchant names, locations, accepted "
        "tokens, and discount structures."
    )
    args_schema: Type[InsumerListMerchantsSchema] = InsumerListMerchantsSchema

    api_wrapper: InsumerAPIWrapper = Field(..., exclude=True)

    def __init__(self, api_wrapper: InsumerAPIWrapper) -> None:
        super().__init__(api_wrapper=api_wrapper)

    def _run(
        self,
        token: Optional[str] = None,
        verified: Optional[bool] = None,
        limit: int = 50,
        offset: int = 0,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """List merchants."""
        return self.api_wrapper.run(
            mode=self.mode,
            token=token,
            verified=verified,
            limit=limit,
            offset=offset,
        )
