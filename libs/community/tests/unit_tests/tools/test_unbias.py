"""Tests for Unbias crypto sentiment tool."""

import json
from unittest.mock import MagicMock, patch

import pytest

from langchain_community.tools.unbias import UnbiasCryptoSentiment


@pytest.fixture
def mock_response_data():
    """Sample API response data."""
    return {
        "asset": "BTC",
        "period": {"start": "2024-01-01", "end": "2024-01-07"},
        "granularity": "daily",
        "count": 7,
        "data": [
            {
                "date": "2024-01-01",
                "consensus_index": 15.0,
                "consensus_index_30d_ma": 10.0,
                "avg_sentiment_score": 58.5,
                "bullish_analysts": 70,
                "bearish_analysts": 35,
                "total_analysts": 105,
            },
            {
                "date": "2024-01-07",
                "consensus_index": 20.0,
                "consensus_index_30d_ma": 12.0,
                "avg_sentiment_score": 62.0,
                "bullish_analysts": 75,
                "bearish_analysts": 36,
                "total_analysts": 111,
            },
        ],
    }


def test_unbias_tool_initialization():
    """Test tool can be initialized with API key."""
    tool = UnbiasCryptoSentiment(api_key="test-key")
    assert tool.name == "unbias_crypto_sentiment"
    assert tool.api_key == "test-key"
    assert "crypto" in tool.description.lower()
    assert "sentiment" in tool.description.lower()


def test_unbias_tool_description():
    """Test tool description contains key information."""
    tool = UnbiasCryptoSentiment(api_key="test-key")
    assert "bullish" in tool.description.lower()
    assert "bearish" in tool.description.lower()
    assert "analyst" in tool.description.lower()


@patch("langchain_community.tools.unbias.tool.requests.get")
def test_unbias_tool_run(mock_get, mock_response_data):
    """Test tool execution with mocked API response."""
    mock_response = MagicMock()
    mock_response.json.return_value = mock_response_data
    mock_response.raise_for_status = MagicMock()
    mock_get.return_value = mock_response

    tool = UnbiasCryptoSentiment(api_key="test-key")
    result = tool._run(asset="BTC", days=7)

    assert "BTC" in result
    assert "Consensus Index" in result
    assert "Bullish" in result
    assert "Bearish" in result
    mock_get.assert_called_once()


@patch("langchain_community.tools.unbias.tool.requests.get")
def test_unbias_tool_handles_error(mock_get):
    """Test tool handles API errors gracefully."""
    mock_get.side_effect = Exception("API connection failed")

    tool = UnbiasCryptoSentiment(api_key="test-key")
    result = tool._run(asset="BTC", days=7)

    assert "error" in result.lower()


@patch("langchain_community.tools.unbias.tool.requests.get")
def test_unbias_tool_formats_trend(mock_get, mock_response_data):
    """Test that trend calculation is included in output."""
    mock_response = MagicMock()
    mock_response.json.return_value = mock_response_data
    mock_response.raise_for_status = MagicMock()
    mock_get.return_value = mock_response

    tool = UnbiasCryptoSentiment(api_key="test-key")
    result = tool._run(asset="BTC", days=7)

    assert "Trend" in result
    assert "↑" in result  # Price went up from 15 to 20
