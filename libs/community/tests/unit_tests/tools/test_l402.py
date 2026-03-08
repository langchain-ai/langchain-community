"""Tests for L402 Lightning payment tools."""

import json
from unittest.mock import MagicMock, patch


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
        result = tool._run(url="https://example.com")
        assert "Error" in result
        assert "l402-requests" in result


def test_l402_fetch_tool_get_success() -> None:
    """Test GET request returns JSON response."""
    from langchain_community.tools.l402 import L402FetchTool

    mock_response = MagicMock()
    mock_response.json.return_value = {"data": "test"}

    mock_client = MagicMock()
    mock_client.get.return_value = mock_response

    tool = L402FetchTool(l402_client=mock_client)
    result = tool._run(url="https://example.com/api")

    parsed = json.loads(result)
    assert parsed == {"data": "test"}
    mock_client.get.assert_called_once_with("https://example.com/api")


def test_l402_fetch_tool_post_success() -> None:
    """Test POST request with JSON body."""
    from langchain_community.tools.l402 import L402FetchTool

    mock_response = MagicMock()
    mock_response.json.return_value = {"received": True}

    mock_client = MagicMock()
    mock_client.post.return_value = mock_response

    tool = L402FetchTool(l402_client=mock_client)
    result = tool._run(
        url="https://example.com/api",
        method="POST",
        body='{"key": "value"}',
    )

    parsed = json.loads(result)
    assert parsed == {"received": True}
    mock_client.post.assert_called_once_with(
        "https://example.com/api", json={"key": "value"}
    )


def test_l402_fetch_tool_text_fallback() -> None:
    """Test non-JSON response falls back to text."""
    from langchain_community.tools.l402 import L402FetchTool

    mock_response = MagicMock()
    mock_response.json.side_effect = ValueError("not JSON")
    mock_response.text = "Hello plain text"

    mock_client = MagicMock()
    mock_client.get.return_value = mock_response

    tool = L402FetchTool(l402_client=mock_client)
    result = tool._run(url="https://example.com/text")

    assert "Hello plain text" in result


def test_l402_spending_tool_with_data() -> None:
    """Test spending tool returns summary when payments have been made."""
    from langchain_community.tools.l402 import L402SpendingTool

    mock_log = MagicMock()
    mock_log.total_spent.return_value = 500
    mock_log.spent_last_hour.return_value = 500
    mock_log.by_domain.return_value = {"api.example.com": 500}

    mock_client = MagicMock()
    mock_client.spending_log = mock_log

    tool = L402SpendingTool(l402_client=mock_client)
    result = tool._run()

    parsed = json.loads(result)
    assert parsed["total_sats"] == 500
    assert parsed["by_domain"]["api.example.com"] == 500
