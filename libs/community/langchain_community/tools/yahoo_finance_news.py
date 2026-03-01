import re
from typing import Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from requests.exceptions import HTTPError, ReadTimeout
from urllib3.exceptions import ConnectionError


class YahooFinanceNewsInput(BaseModel):
    """Input for the YahooFinanceNews tool."""

    query: str = Field(
        description=(
            "Ticker symbol to look up (e.g., 'MSFT'). "
            "If you only have the company name, convert it to the ticker before "
            "calling."
        )
    )


def _looks_like_ticker(text: str) -> bool:
    """Return True if the text resembles a stock ticker."""
    return bool(re.fullmatch(r"[A-Za-z0-9.\-]{1,7}", text.strip()))


class YahooFinanceNewsTool(BaseTool):
    """Tool that searches financial news on Yahoo Finance."""

    name: str = "yahoo_finance_news"
    description: str = (
        "Useful for when you need to find financial news "
        "about a public company. "
        "Input must be a ticker symbol (for example, AAPL for Apple, MSFT for "
        "Microsoft). Convert company names to tickers before invoking this tool."
    )
    top_k: int = 10
    """The number of results to return."""

    args_schema: Type[BaseModel] = YahooFinanceNewsInput

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._last_resolved_symbol: Optional[str] = None

    def _resolve_symbol(self, query: str) -> Optional[str]:
        """Resolve a free-form query to a ticker symbol using Yahoo Finance search."""
        try:
            from yfinance.search import Search
        except (
            ImportError
        ) as exc:  # pragma: no cover - yfinance should be installed by user
            raise ImportError(
                "Could not import yfinance python package. "
                "Please install it with `pip install yfinance`."
            ) from exc

        stripped = query.strip()
        if _looks_like_ticker(stripped):
            return stripped.upper()

        try:
            search = Search(stripped, max_results=10)
            matches = search.quotes or []
        except (HTTPError, ReadTimeout, ConnectionError):
            matches = []

        if matches:
            symbol = matches[0].get("symbol")
            if symbol:
                return symbol.upper()

        return None

    def _parse_input(self, tool_input, tool_call_id=None):
        parsed = super()._parse_input(tool_input, tool_call_id=tool_call_id)
        if isinstance(parsed, dict):
            query = parsed.get("query", "")
            if isinstance(query, str) and not _looks_like_ticker(query):
                symbol = self._resolve_symbol(query)
                if symbol:
                    parsed["query"] = symbol
                    self._last_resolved_symbol = symbol
        return parsed

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """
        Use the Yahoo Finance News tool.

        Args:
            query: Company ticker symbol (e.g., 'AAPL' for Apple).
            run_manager: Optional callback manager.

        Returns:
            str: Formatted news results or error message.
        """
        try:
            import yfinance
        except ImportError:
            raise ImportError(
                "Could not import yfinance python package. "
                "Please install it with `pip install yfinance`."
            )

        if _looks_like_ticker(query):
            symbol = query.strip().upper()
        else:
            symbol = self._last_resolved_symbol or self._resolve_symbol(query)

        self._last_resolved_symbol = None

        if not symbol:
            return f"Could not find a company for query '{query}'."

        company = yfinance.Ticker(symbol)
        try:
            if company.isin is None:
                return f"Company ticker {symbol} not found."
        except (HTTPError, ReadTimeout, ConnectionError):
            return f"Company ticker {symbol} not found."

        try:
            news_items = company.get_news() or []
        except (HTTPError, ReadTimeout, ConnectionError):
            return f"Failed to fetch news for {symbol}."

        stories = [
            item
            for item in news_items
            if item.get("content", {}).get("contentType") == "STORY"
        ]
        if not stories:
            return f"No news found for company with ticker {symbol}."

        summaries = []
        for story in stories[: self.top_k]:
            content = story.get("content", {})
            title = content.get("title") or story.get("title") or ""
            summary = content.get("summary") or content.get("description") or ""
            publisher = (content.get("provider") or {}).get("displayName", "")
            link = (
                (content.get("canonicalUrl") or {}).get("url")
                or (content.get("clickThroughUrl") or {}).get("url")
                or ""
            )
            if not title and not summary:
                continue
            parts = [
                title.strip(),
                summary.strip(),
                f"Источник: {publisher}" if publisher else "",
                link,
            ]
            formatted = "\n".join(filter(None, parts))
            summaries.append(formatted)

        if not summaries:
            return f"No news summaries available for ticker {symbol}."

        return "\n\n---\n\n".join(summaries)
