"""Tool for publishing a merchant to the public directory."""

from typing import Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_community.utilities.insumer import InsumerAPIWrapper


class InsumerPublishDirectorySchema(BaseModel):
    """Input for InsumerPublishDirectory."""

    merchant_id: str = Field(description="Merchant ID to publish.")


class InsumerPublishDirectory(BaseTool):
    """Publish or refresh a merchant listing in the public directory. Owner only.

    Call after creating a merchant and configuring tokens/NFTs/settings.
    Call again after any configuration change to refresh the listing.
    """

    mode: str = "publish_directory"
    name: str = "insumer_publish_directory"
    description: str = (
        "Publish or refresh a merchant's listing in the public directory. "
        "Call after creating a merchant and configuring tokens, NFTs, or "
        "settings. Owner only."
    )
    args_schema: Type[InsumerPublishDirectorySchema] = InsumerPublishDirectorySchema

    api_wrapper: InsumerAPIWrapper = Field(..., exclude=True)

    def __init__(self, api_wrapper: InsumerAPIWrapper) -> None:
        super().__init__(api_wrapper=api_wrapper)

    def _run(
        self,
        merchant_id: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Publish to directory."""
        return self.api_wrapper.run(mode=self.mode, merchant_id=merchant_id)
