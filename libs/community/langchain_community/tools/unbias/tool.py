"""Tool for querying Unbias crypto sentiment API."""

from typing import Optional, Type

import requests
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class UnbiasInput(BaseModel):
    """Input for the Unbias crypto sentiment tool."""

    asset: str = Field(
        default="BTC",
        description="Cryptocurrency asset symbol. Supported: BTC, ETH",
    )
    days: int = Field(
        default=7,
        description="Number of days of historical data (1-365)",
    )


class UnbiasCryptoSentiment(BaseTool):
    """Tool for getting crypto market sentiment from Unbias.

    Unbias aggregates sentiment from 100+ professional crypto analysts
    and traders on Twitter/X, providing real-time market consensus data.

    Setup:
        Get an API key at https://unbias.fyi/api

        .. code-block:: bash

            export UNBIAS_API_KEY="your-api-key"

    Usage:
        .. code-block:: python

            from langchain_community.tools.unbias import UnbiasCryptoSentiment

            tool = UnbiasCryptoSentiment(api_key="your-api-key")
            result = tool.invoke({"asset": "BTC", "days": 7})
            print(result)
    """

    name: str = "unbias_crypto_sentiment"
    description: str = (
        "Get real-time crypto market sentiment from 100+ professional analysts. "
        "Use this tool when you need to know if analysts are bullish or bearish "
        "on Bitcoin (BTC) or Ethereum (ETH), or to get market consensus data. "
        "Returns consensus index (-100 to +100), analyst counts, and sentiment trends."
    )
    args_schema: Type[BaseModel] = UnbiasInput

    api_key: str = Field(description="Unbias API key")
    api_base: str = Field(
        default="https://unbias.fyi/api/v1",
        description="Unbias API base URL",
    )

    def _format_response(self, data: dict) -> str:
        """Format the API response into a readable string."""
        if "data" in data and data["data"]:
            # Historical data
            latest = data["data"][-1]
            period = data.get("period", {})

            lines = [
                f"## {data['asset']} Market Sentiment",
                f"Period: {period.get('start', 'N/A')} to {period.get('end', 'N/A')}",
                f"Data points: {data.get('count', len(data['data']))}",
                "",
                f"### Latest ({latest['date']})",
                f"- Consensus Index: {latest['consensus_index']:.1f} (-100 to +100)",
                f"- 30-Day MA: {latest['consensus_index_30d_ma']:.1f}",
            ]

            if latest.get("avg_sentiment_score") is not None:
                lines.append(
                    f"- Avg Sentiment Score: {latest['avg_sentiment_score']:.1f}/100"
                )

            lines.extend(
                [
                    f"- Bullish Analysts: {latest['bullish_analysts']}",
                    f"- Bearish Analysts: {latest['bearish_analysts']}",
                    f"- Total Analysts: {latest['total_analysts']}",
                ]
            )

            # Calculate trend
            if len(data["data"]) > 1:
                oldest = data["data"][0]
                change = latest["consensus_index"] - oldest["consensus_index"]
                trend = "↑" if change > 0 else "↓" if change < 0 else "→"
                lines.extend(
                    [
                        "",
                        "### Trend",
                        f"- {len(data['data'])}-day change: {trend} {change:+.1f} points",
                    ]
                )

            return "\n".join(lines)
        else:
            # Single data point
            lines = [
                f"## {data['asset']} Market Sentiment (Latest)",
                f"Date: {data.get('date', 'N/A')}",
                f"- Consensus Index: {data.get('consensus_index', 0):.1f} (-100 to +100)",
                f"- 30-Day MA: {data.get('consensus_index_30d_ma', 0):.1f}",
            ]

            if data.get("avg_sentiment_score") is not None:
                lines.append(
                    f"- Avg Sentiment Score: {data['avg_sentiment_score']:.1f}/100"
                )

            lines.extend(
                [
                    f"- Bullish Analysts: {data.get('bullish_analysts', 0)}",
                    f"- Bearish Analysts: {data.get('bearish_analysts', 0)}",
                    f"- Total Analysts: {data.get('total_analysts', 0)}",
                ]
            )

            return "\n".join(lines)

    def _run(
        self,
        asset: str = "BTC",
        days: int = 7,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Execute the tool to get crypto sentiment data.

        Args:
            asset: Cryptocurrency symbol (BTC or ETH)
            days: Number of days of historical data

        Returns:
            Formatted string with sentiment data
        """
        try:
            response = requests.get(
                f"{self.api_base}/consensus",
                params={
                    "api_key": self.api_key,
                    "asset": asset.upper(),
                    "days": days,
                },
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()

            if "error" in data:
                return f"API Error: {data['error']}"

            return self._format_response(data)

        except requests.exceptions.RequestException as e:
            return f"Error fetching sentiment data: {str(e)}"
        except Exception as e:
            return f"Unexpected error: {str(e)}"
