import sys
import types

import pytest

from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool


@pytest.fixture()
def fake_yfinance(monkeypatch):
    """Install a minimal fake `yfinance` module to avoid network requests."""

    class DummySearch:
        def __init__(self, query: str, max_results: int = 10):
            self.query = query
            self.max_results = max_results
            if query.lower() == "microsoft":
                self.quotes = [{"symbol": "MSFT"}]
            else:
                self.quotes = [{"symbol": query.upper()}]

    class DummyTicker:
        def __init__(self, symbol: str):
            self.symbol = symbol
            self.isin = "DUMMYISIN"
            yf_module.last_symbol = symbol
            self._news = [
                {
                    "content": {
                        "contentType": "STORY",
                        "title": "Sample Headline",
                        "summary": "Sample Summary",
                        "provider": {"displayName": "Sample Publisher"},
                        "canonicalUrl": {"url": "https://example.com/article"},
                    }
                }
            ]

        def get_news(self):
            return self._news

    search_module = types.ModuleType("yfinance.search")
    search_module.Search = DummySearch

    yf_module = types.ModuleType("yfinance")
    yf_module.search = search_module
    yf_module.Ticker = DummyTicker

    monkeypatch.setitem(sys.modules, "yfinance", yf_module)
    monkeypatch.setitem(sys.modules, "yfinance.search", search_module)

    yf_module.last_symbol = None

    return yf_module


def test_tool_resolves_company_name(fake_yfinance):
    """The tool should resolve a company name to a ticker before executing."""
    tool = YahooFinanceNewsTool()
    output = tool.run("Microsoft")

    assert output
    assert fake_yfinance.last_symbol == "MSFT"
    assert "Sample Headline" in output
    assert "Sample Summary" in output
    assert "Sample Publisher" in output
    assert "https://example.com/article" in output


def test_tool_accepts_ticker(fake_yfinance):
    """The tool should accept ticker symbols directly."""
    tool = YahooFinanceNewsTool()
    output = tool.run("MSFT")

    # Output should still contain the formatted news snippet.
    assert "Sample Headline" in output
    assert "Sample Summary" in output
    assert fake_yfinance.last_symbol == "MSFT"
