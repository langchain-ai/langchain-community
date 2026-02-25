"""Spraay batch payment tools for LangChain.

Tools for batch-sending ETH and ERC-20 tokens on Base using the
Spraay protocol. Enables AI agents to distribute payments to up to
200 recipients in a single transaction with ~80% gas savings.
"""

from __future__ import annotations

import json
import os
from typing import Any, List, Optional, Type

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

# Spraay V2 contract on Base Mainnet
DEFAULT_CONTRACT_ADDRESS = "0x1646452F98E36A3c9Cfc3eDD8868221E207B5eEC"
DEFAULT_RPC_URL = "https://mainnet.base.org"
BASE_CHAIN_ID = 8453

# Spraay V2 ABI (minimal - batch functions only)
SPRAAY_ABI = json.loads(
    """[
    {
        "inputs": [
            {"internalType": "address[]", "name": "recipients", "type": "address[]"},
            {"internalType": "uint256", "name": "amountPerRecipient", "type": "uint256"}
        ],
        "name": "batchSendETH",
        "outputs": [],
        "stateMutability": "payable",
        "type": "function"
    },
    {
        "inputs": [
            {"internalType": "address", "name": "token", "type": "address"},
            {"internalType": "address[]", "name": "recipients", "type": "address[]"},
            {"internalType": "uint256", "name": "amountPerRecipient", "type": "uint256"}
        ],
        "name": "batchSendToken",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [
            {"internalType": "address[]", "name": "recipients", "type": "address[]"},
            {"internalType": "uint256[]", "name": "amounts", "type": "uint256[]"}
        ],
        "name": "batchSendETHVariable",
        "outputs": [],
        "stateMutability": "payable",
        "type": "function"
    },
    {
        "inputs": [
            {"internalType": "address", "name": "token", "type": "address"},
            {"internalType": "address[]", "name": "recipients", "type": "address[]"},
            {"internalType": "uint256[]", "name": "amounts", "type": "uint256[]"}
        ],
        "name": "batchSendTokenVariable",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    }
]"""
)

ERC20_APPROVE_ABI = json.loads(
    """[
    {
        "inputs": [
            {"internalType": "address", "name": "spender", "type": "address"},
            {"internalType": "uint256", "name": "amount", "type": "uint256"}
        ],
        "name": "approve",
        "outputs": [{"internalType": "bool", "name": "", "type": "bool"}],
        "stateMutability": "nonpayable",
        "type": "function"
    }
]"""
)


def _get_web3():  # type: ignore[no-untyped-def]
    """Lazily import and return web3 module."""
    try:
        from web3 import Web3  # type: ignore[import-untyped]

        return Web3
    except ImportError:
        raise ImportError(
            "web3 is required for Spraay tools. "
            "Install it with: pip install web3"
        )


def _get_connection() -> tuple:
    """Get web3 connection and account from environment."""
    Web3 = _get_web3()
    rpc_url = os.environ.get("SPRAAY_RPC_URL", DEFAULT_RPC_URL)
    private_key = os.environ.get("SPRAAY_PRIVATE_KEY")
    if not private_key:
        raise ValueError(
            "SPRAAY_PRIVATE_KEY environment variable is required. "
            "Set it to the private key of the wallet that will send payments."
        )
    contract_address = os.environ.get(
        "SPRAAY_CONTRACT_ADDRESS", DEFAULT_CONTRACT_ADDRESS
    )

    w3 = Web3(Web3.HTTPProvider(rpc_url))
    account = w3.eth.account.from_key(private_key)
    contract = w3.eth.contract(
        address=Web3.to_checksum_address(contract_address),
        abi=SPRAAY_ABI,
    )
    return w3, account, contract


# --- Input schemas ---


class BatchSendETHInput(BaseModel):
    """Input for batch sending equal ETH amounts."""

    recipients: List[str] = Field(
        description="List of recipient wallet addresses (max 200)"
    )
    amount_per_recipient_eth: str = Field(
        description="Amount of ETH to send to each recipient (e.g. '0.01')"
    )


class BatchSendTokenInput(BaseModel):
    """Input for batch sending equal ERC-20 token amounts."""

    token_address: str = Field(description="ERC-20 token contract address")
    recipients: List[str] = Field(
        description="List of recipient wallet addresses (max 200)"
    )
    amount_per_recipient: str = Field(
        description="Amount of tokens per recipient in human-readable units (e.g. '100')"
    )
    token_decimals: int = Field(
        default=18, description="Token decimal places (default 18)"
    )


class BatchSendETHVariableInput(BaseModel):
    """Input for batch sending variable ETH amounts."""

    recipients: List[str] = Field(
        description="List of recipient wallet addresses (max 200)"
    )
    amounts_eth: List[str] = Field(
        description="List of ETH amounts corresponding to each recipient (e.g. ['0.01', '0.05'])"
    )


class BatchSendTokenVariableInput(BaseModel):
    """Input for batch sending variable ERC-20 token amounts."""

    token_address: str = Field(description="ERC-20 token contract address")
    recipients: List[str] = Field(
        description="List of recipient wallet addresses (max 200)"
    )
    amounts: List[str] = Field(
        description="List of token amounts corresponding to each recipient"
    )
    token_decimals: int = Field(
        default=18, description="Token decimal places (default 18)"
    )


# --- Tools ---


class SpraayBatchSendETH(BaseTool):  # type: ignore[override]
    """Tool for batch sending equal ETH to multiple recipients on Base.

    Uses the Spraay protocol to distribute equal ETH amounts to up to 200
    recipients in a single transaction, saving ~80% on gas vs individual
    transfers. A 0.3% protocol fee applies.

    Setup:
        Install ``web3``:

        .. code-block:: bash

            pip install web3

        Set environment variables:

        .. code-block:: bash

            export SPRAAY_PRIVATE_KEY="your-private-key"
            export SPRAAY_RPC_URL="https://mainnet.base.org"  # optional

    Example:
        .. code-block:: python

            from langchain_community.tools.spraay import SpraayBatchSendETH

            tool = SpraayBatchSendETH()
            result = tool.invoke({
                "recipients": ["0xAbc...", "0xDef..."],
                "amount_per_recipient_eth": "0.01"
            })
    """

    name: str = "spraay_batch_send_eth"
    description: str = (
        "Send equal amounts of ETH to multiple recipients on Base in a single "
        "transaction using Spraay. Supports up to 200 recipients with ~80% gas "
        "savings. Input: list of recipient addresses and ETH amount per recipient."
    )
    args_schema: Type[BaseModel] = BatchSendETHInput

    def _run(
        self,
        recipients: List[str],
        amount_per_recipient_eth: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Execute batch ETH send."""
        try:
            Web3 = _get_web3()
            w3, account, contract = _get_connection()

            if len(recipients) > 200:
                return "Error: Maximum 200 recipients per transaction."

            checksummed = [Web3.to_checksum_address(r) for r in recipients]
            amount_wei = w3.to_wei(amount_per_recipient_eth, "ether")
            fee = amount_wei * 30 // 10000  # 0.3% fee
            total_value = (amount_wei + fee) * len(checksummed)

            tx = contract.functions.batchSendETH(
                checksummed, amount_wei
            ).build_transaction(
                {
                    "from": account.address,
                    "value": total_value,
                    "nonce": w3.eth.get_transaction_count(account.address),
                    "chainId": BASE_CHAIN_ID,
                }
            )

            signed = account.sign_transaction(tx)
            tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash)

            return (
                f"Success! Sent {amount_per_recipient_eth} ETH to "
                f"{len(checksummed)} recipients. "
                f"Tx: https://basescan.org/tx/{receipt['transactionHash'].hex()}"
            )
        except Exception as e:
            return f"Error: {e!s}"


class SpraayBatchSendToken(BaseTool):  # type: ignore[override]
    """Tool for batch sending equal ERC-20 tokens to multiple recipients on Base.

    Uses the Spraay protocol to distribute equal token amounts to up to 200
    recipients. Automatically handles token approval. A 0.3% protocol fee applies.

    Setup:
        Install ``web3``:

        .. code-block:: bash

            pip install web3

        Set environment variables:

        .. code-block:: bash

            export SPRAAY_PRIVATE_KEY="your-private-key"

    Example:
        .. code-block:: python

            from langchain_community.tools.spraay import SpraayBatchSendToken

            tool = SpraayBatchSendToken()
            result = tool.invoke({
                "token_address": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
                "recipients": ["0xAbc...", "0xDef..."],
                "amount_per_recipient": "10",
                "token_decimals": 6
            })
    """

    name: str = "spraay_batch_send_token"
    description: str = (
        "Send equal amounts of an ERC-20 token to multiple recipients on Base "
        "in a single transaction using Spraay. Supports up to 200 recipients. "
        "Handles token approval automatically. Input: token address, recipients, "
        "amount per recipient, and token decimals."
    )
    args_schema: Type[BaseModel] = BatchSendTokenInput

    def _run(
        self,
        token_address: str,
        recipients: List[str],
        amount_per_recipient: str,
        token_decimals: int = 18,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Execute batch token send."""
        try:
            Web3 = _get_web3()
            w3, account, contract = _get_connection()

            if len(recipients) > 200:
                return "Error: Maximum 200 recipients per transaction."

            checksummed = [Web3.to_checksum_address(r) for r in recipients]
            token_addr = Web3.to_checksum_address(token_address)
            amount_raw = int(float(amount_per_recipient) * (10**token_decimals))
            total_amount = amount_raw * len(checksummed)

            contract_address = os.environ.get(
                "SPRAAY_CONTRACT_ADDRESS", DEFAULT_CONTRACT_ADDRESS
            )

            # Approve tokens
            token_contract = w3.eth.contract(
                address=token_addr, abi=ERC20_APPROVE_ABI
            )
            approve_tx = token_contract.functions.approve(
                Web3.to_checksum_address(contract_address), total_amount
            ).build_transaction(
                {
                    "from": account.address,
                    "nonce": w3.eth.get_transaction_count(account.address),
                    "chainId": BASE_CHAIN_ID,
                }
            )
            signed_approve = account.sign_transaction(approve_tx)
            w3.eth.send_raw_transaction(signed_approve.raw_transaction)
            w3.eth.wait_for_transaction_receipt(
                w3.eth.send_raw_transaction(signed_approve.raw_transaction)
            )

            # Send batch
            tx = contract.functions.batchSendToken(
                token_addr, checksummed, amount_raw
            ).build_transaction(
                {
                    "from": account.address,
                    "nonce": w3.eth.get_transaction_count(account.address),
                    "chainId": BASE_CHAIN_ID,
                }
            )
            signed = account.sign_transaction(tx)
            tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash)

            return (
                f"Success! Sent {amount_per_recipient} tokens to "
                f"{len(checksummed)} recipients. "
                f"Tx: https://basescan.org/tx/{receipt['transactionHash'].hex()}"
            )
        except Exception as e:
            return f"Error: {e!s}"


class SpraayBatchSendETHVariable(BaseTool):  # type: ignore[override]
    """Tool for batch sending variable ETH amounts to multiple recipients on Base.

    Uses the Spraay protocol to send different ETH amounts to each recipient
    in a single transaction. Useful for payroll, bounty distribution, or
    revenue sharing with varying amounts.

    Setup:
        Install ``web3``:

        .. code-block:: bash

            pip install web3

        Set environment variables:

        .. code-block:: bash

            export SPRAAY_PRIVATE_KEY="your-private-key"

    Example:
        .. code-block:: python

            from langchain_community.tools.spraay import SpraayBatchSendETHVariable

            tool = SpraayBatchSendETHVariable()
            result = tool.invoke({
                "recipients": ["0xAbc...", "0xDef..."],
                "amounts_eth": ["0.01", "0.05"]
            })
    """

    name: str = "spraay_batch_send_eth_variable"
    description: str = (
        "Send different amounts of ETH to multiple recipients on Base in a "
        "single transaction using Spraay. Each recipient gets their specified "
        "amount. Supports up to 200 recipients. Input: list of recipient "
        "addresses and corresponding ETH amounts."
    )
    args_schema: Type[BaseModel] = BatchSendETHVariableInput

    def _run(
        self,
        recipients: List[str],
        amounts_eth: List[str],
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Execute variable batch ETH send."""
        try:
            Web3 = _get_web3()
            w3, account, contract = _get_connection()

            if len(recipients) != len(amounts_eth):
                return "Error: recipients and amounts_eth must have the same length."
            if len(recipients) > 200:
                return "Error: Maximum 200 recipients per transaction."

            checksummed = [Web3.to_checksum_address(r) for r in recipients]
            amounts_wei = [w3.to_wei(a, "ether") for a in amounts_eth]
            fees = [a * 30 // 10000 for a in amounts_wei]
            total_value = sum(a + f for a, f in zip(amounts_wei, fees))

            tx = contract.functions.batchSendETHVariable(
                checksummed, amounts_wei
            ).build_transaction(
                {
                    "from": account.address,
                    "value": total_value,
                    "nonce": w3.eth.get_transaction_count(account.address),
                    "chainId": BASE_CHAIN_ID,
                }
            )

            signed = account.sign_transaction(tx)
            tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash)

            total_eth = sum(float(a) for a in amounts_eth)
            return (
                f"Success! Sent {total_eth} ETH total to "
                f"{len(checksummed)} recipients (variable amounts). "
                f"Tx: https://basescan.org/tx/{receipt['transactionHash'].hex()}"
            )
        except Exception as e:
            return f"Error: {e!s}"


class SpraayBatchSendTokenVariable(BaseTool):  # type: ignore[override]
    """Tool for batch sending variable ERC-20 token amounts on Base.

    Uses the Spraay protocol to send different token amounts to each recipient
    in a single transaction. Automatically handles token approval.

    Setup:
        Install ``web3``:

        .. code-block:: bash

            pip install web3

        Set environment variables:

        .. code-block:: bash

            export SPRAAY_PRIVATE_KEY="your-private-key"

    Example:
        .. code-block:: python

            from langchain_community.tools.spraay import SpraayBatchSendTokenVariable

            tool = SpraayBatchSendTokenVariable()
            result = tool.invoke({
                "token_address": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
                "recipients": ["0xAbc...", "0xDef..."],
                "amounts": ["100", "250"],
                "token_decimals": 6
            })
    """

    name: str = "spraay_batch_send_token_variable"
    description: str = (
        "Send different amounts of an ERC-20 token to multiple recipients on "
        "Base in a single transaction using Spraay. Each recipient gets their "
        "specified amount. Handles token approval automatically. Input: token "
        "address, recipients, corresponding amounts, and token decimals."
    )
    args_schema: Type[BaseModel] = BatchSendTokenVariableInput

    def _run(
        self,
        token_address: str,
        recipients: List[str],
        amounts: List[str],
        token_decimals: int = 18,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Execute variable batch token send."""
        try:
            Web3 = _get_web3()
            w3, account, contract = _get_connection()

            if len(recipients) != len(amounts):
                return "Error: recipients and amounts must have the same length."
            if len(recipients) > 200:
                return "Error: Maximum 200 recipients per transaction."

            checksummed = [Web3.to_checksum_address(r) for r in recipients]
            token_addr = Web3.to_checksum_address(token_address)
            amounts_raw = [
                int(float(a) * (10**token_decimals)) for a in amounts
            ]
            total_amount = sum(amounts_raw)

            contract_address = os.environ.get(
                "SPRAAY_CONTRACT_ADDRESS", DEFAULT_CONTRACT_ADDRESS
            )

            # Approve tokens
            token_contract = w3.eth.contract(
                address=token_addr, abi=ERC20_APPROVE_ABI
            )
            approve_tx = token_contract.functions.approve(
                Web3.to_checksum_address(contract_address), total_amount
            ).build_transaction(
                {
                    "from": account.address,
                    "nonce": w3.eth.get_transaction_count(account.address),
                    "chainId": BASE_CHAIN_ID,
                }
            )
            signed_approve = account.sign_transaction(approve_tx)
            w3.eth.send_raw_transaction(signed_approve.raw_transaction)
            w3.eth.wait_for_transaction_receipt(
                w3.eth.send_raw_transaction(signed_approve.raw_transaction)
            )

            # Send batch
            tx = contract.functions.batchSendTokenVariable(
                token_addr, checksummed, amounts_raw
            ).build_transaction(
                {
                    "from": account.address,
                    "nonce": w3.eth.get_transaction_count(account.address),
                    "chainId": BASE_CHAIN_ID,
                }
            )
            signed = account.sign_transaction(tx)
            tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash)

            total_tokens = sum(float(a) for a in amounts)
            return (
                f"Success! Sent {total_tokens} tokens total to "
                f"{len(checksummed)} recipients (variable amounts). "
                f"Tx: https://basescan.org/tx/{receipt['transactionHash'].hex()}"
            )
        except Exception as e:
            return f"Error: {e!s}"
