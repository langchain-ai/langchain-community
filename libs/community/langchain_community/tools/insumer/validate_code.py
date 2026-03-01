"""Tool for validating discount codes at point of sale."""

from typing import Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_community.utilities.insumer import InsumerAPIWrapper


class InsumerValidateCodeSchema(BaseModel):
    """Input for InsumerValidateCode."""

    code: str = Field(
        description='Discount code to validate (e.g. "INSR-A7K3M").',
    )


class InsumerValidateCode(BaseTool):
    """Validate a discount code at point of sale.

    No API key required. Returns code validity, discount percentage,
    merchant, and expiry time.
    """

    mode: str = "validate_code"
    name: str = "insumer_validate_code"
    description: str = (
        "Validate an INSR-XXXXX discount code at point of sale. No API key "
        "required. Returns validity, discount percentage, merchant, and expiry."
    )
    args_schema: Type[InsumerValidateCodeSchema] = InsumerValidateCodeSchema

    api_wrapper: InsumerAPIWrapper = Field(..., exclude=True)

    def __init__(self, api_wrapper: InsumerAPIWrapper) -> None:
        super().__init__(api_wrapper=api_wrapper)

    def _run(
        self,
        code: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Validate discount code."""
        return self.api_wrapper.run(
            mode=self.mode,
            code=code,
        )
