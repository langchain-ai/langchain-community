from typing import Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_community.utilities.financial_datasets import FinancialDatasetsAPIWrapper


class FinancialSnapshotsSchema(BaseModel):
    """Input for FinancialSnapshots."""

    ticker: str = Field(
        description="The ticker symbol to fetch financial snapshots for.",
    )


class FinancialSnapshots(BaseTool):
    """
    Tool that gets financial snapshots for a given ticker over a given period.
    """

    mode: str = "get_financial_snapshots"
    name: str = "financial_snapshots"
    description: str = (
        "A wrapper around financial datasets's financial snapshots API. "
        "This tool is useful for fetching financial snapshots for a given ticker."
        "The tool fetches financial snapshots for a given ticker."
    )
    args_schema: Type[FinancialSnapshotsSchema] = FinancialSnapshotsSchema

    api_wrapper: FinancialDatasetsAPIWrapper = Field(..., exclude=True)

    def __init__(self, api_wrapper: FinancialDatasetsAPIWrapper):
        super().__init__(api_wrapper=api_wrapper)

    def _run(
        self,
        ticker: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the financial snapshots API tool."""
        return self.api_wrapper.run(mode=self.mode, ticker=ticker)
