"""Tests for AsterPay tools."""

from unittest.mock import MagicMock, patch

from langchain_community.tools.asterpay import (
    AsterPayKYADeepAnalysis,
    AsterPayKYATier,
    AsterPayKYATrustScore,
    AsterPayKYAVerify,
    AsterPaySettlementEstimate,
)
from langchain_community.utilities.asterpay import AsterPayAPIWrapper


def test_trust_score_tool_init() -> None:
    """Test AsterPayKYATrustScore initializes correctly."""
    tool = AsterPayKYATrustScore()
    assert tool.name == "asterpay_kya_trust_score"
    assert "trust score" in tool.description.lower()
    assert isinstance(tool.api_wrapper, AsterPayAPIWrapper)


def test_verify_tool_init() -> None:
    """Test AsterPayKYAVerify initializes correctly."""
    tool = AsterPayKYAVerify()
    assert tool.name == "asterpay_kya_verify"
    assert "ERC-8004" in tool.description


def test_tier_tool_init() -> None:
    """Test AsterPayKYATier initializes correctly."""
    tool = AsterPayKYATier()
    assert tool.name == "asterpay_kya_tier"
    assert "tier" in tool.description.lower()


def test_deep_analysis_tool_init() -> None:
    """Test AsterPayKYADeepAnalysis initializes correctly."""
    tool = AsterPayKYADeepAnalysis()
    assert tool.name == "asterpay_kya_deep_analysis"
    assert "x402" in tool.description


def test_settlement_tool_init() -> None:
    """Test AsterPaySettlementEstimate initializes correctly."""
    tool = AsterPaySettlementEstimate()
    assert tool.name == "asterpay_settlement_estimate"
    assert "EUR" in tool.description


def test_api_wrapper_default_url() -> None:
    """Test AsterPayAPIWrapper has correct default URL."""
    wrapper = AsterPayAPIWrapper()
    assert wrapper.base_url == "https://x402.asterpay.io"
    assert wrapper.timeout == 30


def test_trust_score_tool_with_mock() -> None:
    """Test trust score tool with mocked API."""
    mock_wrapper = MagicMock(spec=AsterPayAPIWrapper)
    mock_wrapper.trust_score.return_value = '{"score": 75, "tier": "Verified"}'
    tool = AsterPayKYATrustScore(api_wrapper=mock_wrapper)
    result = tool._run(address="0x1234567890abcdef1234567890abcdef12345678")
    mock_wrapper.trust_score.assert_called_once_with(
        "0x1234567890abcdef1234567890abcdef12345678"
    )
    assert "75" in result


def test_settlement_tool_with_mock() -> None:
    """Test settlement estimate tool with mocked API."""
    mock_wrapper = MagicMock(spec=AsterPayAPIWrapper)
    mock_wrapper.settlement_estimate.return_value = (
        '{"eur_amount": 92.50, "fee": 1.0}'
    )
    tool = AsterPaySettlementEstimate(api_wrapper=mock_wrapper)
    result = tool._run(amount_usdc=100.0)
    mock_wrapper.settlement_estimate.assert_called_once_with(100.0)
    assert "92.50" in result


def test_tool_handles_error() -> None:
    """Test tools handle API errors gracefully."""
    mock_wrapper = MagicMock(spec=AsterPayAPIWrapper)
    mock_wrapper.trust_score.side_effect = Exception("Connection error")
    tool = AsterPayKYATrustScore(api_wrapper=mock_wrapper)
    result = tool._run(address="0xinvalid")
    assert "Connection error" in result
