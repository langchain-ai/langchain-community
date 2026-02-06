from typing import Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_community.utilities.financial_datasets import FinancialDatasetsAPIWrapper


class FinancialMetricsSchema(BaseModel):
    """Input for FinancialMetrics."""

    ticker: str = Field(
        description="The ticker symbol to fetch  financial metrics for.",
    )
    period: str = Field(
        description="The period of the financial metrics. "
        "Possible values are: "
        "annual, quarterly, ttm. "
        "Default is 'annual'.",
    )
    limit: int = Field(
        description="The number of financial metrics to return. Default is 10.",
    )


class FinancialMetrics(BaseTool):
    """
    Tool that gets  financial metrics for a given ticker over a given period.
    """

    mode: str = "get_financial_metrics"
    name: str = "financial_metrics"
    description: str = (
        "A wrapper around financial datasets's  financial metrics API. "
        "This tool is useful for fetching  financial metrics for a given ticker."
        "The tool fetches  financial metrics for a given ticker over a given period."
        "The period can be annual, quarterly, or trailing twelve months (ttm)."
        "The number of  financial metrics to return can also be "
        "specified using the limit parameter."
    )
    args_schema: Type[FinancialMetricsSchema] = FinancialMetricsSchema

    api_wrapper: FinancialDatasetsAPIWrapper = Field(..., exclude=True)

    def __init__(self, api_wrapper: FinancialDatasetsAPIWrapper):
        super().__init__(api_wrapper=api_wrapper)

    def _run(
        self,
        ticker: str,
        period: str,
        limit: Optional[int],
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the  financial metrics API tool."""
        return self.api_wrapper.run(
            mode=self.mode,
            ticker=ticker,
            period=period,
            limit=limit,
        )
