"""Tests for L402 Lightning payment tools."""

from unittest.mock import MagicMock, patch

import pytest


def test_l402_fetch_tool_import() -> None:
    """Test that L402FetchTool can be imported."""
    from langchain_community.tools.l402 import L402FetchTool

    assert L402FetchTool is not None


def test_l402_spending_tool_import() -> None:
    """Test that L402SpendingTool can be imported."""
    from langchain_community.tools.l402 import L402SpendingTool

    assert L402SpendingTool is not None


def test_l402_fetch_tool_metadata() -> None:
    """Test tool name and description."""
    from langchain_community.tools.l402 import L402FetchTool

    with patch.dict("sys.modules", {"l402_requests": MagicMock()}):
        tool = L402FetchTool()
        assert tool.name == "l402_fetch"
        assert "402" in tool.description or "L402" in tool.description


def test_l402_spending_tool_no_client() -> None:
    """Test spending tool without client returns helpful message."""
    from langchain_community.tools.l402 import L402SpendingTool

    tool = L402SpendingTool()
    result = tool._run()
    assert "No L402 client" in result


def test_l402_fetch_tool_missing_dependency() -> None:
    """Test helpful error when l402-requests not installed."""
    from langchain_community.tools.l402.tool import L402FetchTool

    tool = L402FetchTool()

    with patch.dict("sys.modules", {"l402_requests": None}):
        with patch(
            "langchain_community.tools.l402.tool.L402FetchTool._get_client",
            side_effect=ImportError(
                "l402-requests is required for L402FetchTool. "
                "Install it with: pip install l402-requests"
            ),
        ):
            result = tool._run(url="https://example.com")
            assert "Error" in result
