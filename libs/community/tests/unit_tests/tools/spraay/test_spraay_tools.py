"""Tests for Spraay batch payment tools."""

import os
from unittest.mock import MagicMock, patch

import pytest

from langchain_community.tools.spraay.tool import (
    SpraayBatchSendETH,
    SpraayBatchSendETHVariable,
    SpraayBatchSendToken,
    SpraayBatchSendTokenVariable,
)


def test_batch_send_eth_name() -> None:
    """Test tool name."""
    tool = SpraayBatchSendETH()
    assert tool.name == "spraay_batch_send_eth"


def test_batch_send_token_name() -> None:
    """Test tool name."""
    tool = SpraayBatchSendToken()
    assert tool.name == "spraay_batch_send_token"


def test_batch_send_eth_variable_name() -> None:
    """Test tool name."""
    tool = SpraayBatchSendETHVariable()
    assert tool.name == "spraay_batch_send_eth_variable"


def test_batch_send_token_variable_name() -> None:
    """Test tool name."""
    tool = SpraayBatchSendTokenVariable()
    assert tool.name == "spraay_batch_send_token_variable"


def test_batch_send_eth_description() -> None:
    """Test tool description mentions key features."""
    tool = SpraayBatchSendETH()
    assert "ETH" in tool.description
    assert "200" in tool.description
    assert "Spraay" in tool.description


def test_batch_send_eth_max_recipients() -> None:
    """Test that tool rejects more than 200 recipients."""
    tool = SpraayBatchSendETH()
    recipients = [f"0x{i:040x}" for i in range(201)]
    with patch.dict(os.environ, {"SPRAAY_PRIVATE_KEY": "0x" + "a" * 64}):
        with patch("langchain_community.tools.spraay.tool._get_web3") as mock_web3:
            mock_w3_cls = MagicMock()
            mock_web3.return_value = mock_w3_cls
            mock_w3_cls.to_checksum_address = lambda x: x
            mock_w3 = MagicMock()
            mock_w3_cls.return_value = mock_w3

            with patch(
                "langchain_community.tools.spraay.tool._get_connection"
            ) as mock_conn:
                mock_conn.return_value = (mock_w3, MagicMock(), MagicMock())
                result = tool._run(
                    recipients=recipients, amount_per_recipient_eth="0.01"
                )
                assert "Maximum 200" in result


def test_batch_send_eth_variable_length_mismatch() -> None:
    """Test that variable tool rejects mismatched lengths."""
    tool = SpraayBatchSendETHVariable()
    with patch.dict(os.environ, {"SPRAAY_PRIVATE_KEY": "0x" + "a" * 64}):
        with patch(
            "langchain_community.tools.spraay.tool._get_connection"
        ) as mock_conn:
            mock_conn.return_value = (MagicMock(), MagicMock(), MagicMock())
            result = tool._run(
                recipients=["0x" + "a" * 40, "0x" + "b" * 40],
                amounts_eth=["0.01"],
            )
            assert "same length" in result


def test_missing_private_key() -> None:
    """Test error when SPRAAY_PRIVATE_KEY not set."""
    tool = SpraayBatchSendETH()
    with patch.dict(os.environ, {}, clear=True):
        with patch("langchain_community.tools.spraay.tool._get_web3"):
            result = tool._run(
                recipients=["0x" + "a" * 40],
                amount_per_recipient_eth="0.01",
            )
            assert "SPRAAY_PRIVATE_KEY" in result


def test_all_tools_have_args_schema() -> None:
    """Test that all tools have proper input schemas."""
    tools = [
        SpraayBatchSendETH(),
        SpraayBatchSendToken(),
        SpraayBatchSendETHVariable(),
        SpraayBatchSendTokenVariable(),
    ]
    for tool in tools:
        assert tool.args_schema is not None
        assert hasattr(tool.args_schema, "model_fields")
