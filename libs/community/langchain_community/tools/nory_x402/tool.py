"""Nory x402 Payment Tools.

Tools for AI agents to make payments using the x402 HTTP protocol.
Supports Solana and 7 EVM chains with sub-400ms settlement.
"""

from __future__ import annotations

from typing import Any, Optional, Type

import requests
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, SecretStr


NORY_API_BASE = "https://noryx402.com"


class PaymentRequirementsInput(BaseModel):
    """Input for getting payment requirements."""

    resource: str = Field(description="The resource path requiring payment")
    amount: str = Field(description="Amount in human-readable format (e.g., '0.10')")
    network: Optional[str] = Field(
        default=None,
        description="Preferred network: solana-mainnet, base-mainnet, etc.",
    )


class VerifyPaymentInput(BaseModel):
    """Input for verifying a payment."""

    payload: str = Field(
        description="Base64-encoded payment payload containing signed transaction"
    )


class SettlePaymentInput(BaseModel):
    """Input for settling a payment."""

    payload: str = Field(description="Base64-encoded payment payload")


class TransactionLookupInput(BaseModel):
    """Input for looking up a transaction."""

    transaction_id: str = Field(description="Transaction ID or signature")
    network: str = Field(description="Network ID (e.g., solana-mainnet)")


class NoryPaymentRequirementsTool(BaseTool):
    """Tool for getting x402 payment requirements.

    Use this when you encounter an HTTP 402 Payment Required response
    and need to know how much to pay and where to send payment.

    Example usage:
        .. code-block:: python

            tool = NoryPaymentRequirementsTool()
            result = tool.run({
                "resource": "/api/premium/data",
                "amount": "0.10"
            })
    """

    name: str = "nory_get_payment_requirements"
    description: str = (
        "Get payment requirements for accessing a paid resource. "
        "Returns the amount, supported networks, and wallet address needed. "
        "Use when you encounter an HTTP 402 Payment Required response."
    )
    args_schema: Type[BaseModel] = PaymentRequirementsInput
    api_key: Optional[SecretStr] = None

    @classmethod
    def from_api_key(cls, api_key: str, **kwargs: Any) -> NoryPaymentRequirementsTool:
        """Create a tool from an API key.

        Args:
            api_key: The Nory API key to use.
            **kwargs: Any additional kwargs to pass to the tool.

        Returns:
            A configured tool instance.
        """
        return cls(api_key=SecretStr(api_key), **kwargs)

    def _run(
        self,
        resource: str,
        amount: str,
        network: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Get payment requirements for a resource.

        Args:
            resource: The resource path requiring payment.
            amount: Amount in human-readable format.
            network: Preferred network (optional).
            run_manager: Callback manager (optional).

        Returns:
            JSON string with payment requirements.
        """
        params = {"resource": resource, "amount": amount}
        if network:
            params["network"] = network

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key.get_secret_value()}"

        response = requests.get(
            f"{NORY_API_BASE}/api/x402/requirements",
            params=params,
            headers=headers,
            timeout=30,
        )
        return response.text


class NoryVerifyPaymentTool(BaseTool):
    """Tool for verifying a payment before settlement.

    Use this to validate that a signed payment transaction is correct
    before submitting it to the blockchain.

    Example usage:
        .. code-block:: python

            tool = NoryVerifyPaymentTool()
            result = tool.run({"payload": "base64_encoded_payload"})
    """

    name: str = "nory_verify_payment"
    description: str = (
        "Verify a signed payment transaction before submitting to blockchain. "
        "Use to validate a payment is correct before settlement."
    )
    args_schema: Type[BaseModel] = VerifyPaymentInput
    api_key: Optional[SecretStr] = None

    @classmethod
    def from_api_key(cls, api_key: str, **kwargs: Any) -> NoryVerifyPaymentTool:
        """Create a tool from an API key.

        Args:
            api_key: The Nory API key to use.
            **kwargs: Any additional kwargs to pass to the tool.

        Returns:
            A configured tool instance.
        """
        return cls(api_key=SecretStr(api_key), **kwargs)

    def _run(
        self,
        payload: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Verify a payment transaction.

        Args:
            payload: Base64-encoded payment payload.
            run_manager: Callback manager (optional).

        Returns:
            JSON string with verification result.
        """
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key.get_secret_value()}"

        response = requests.post(
            f"{NORY_API_BASE}/api/x402/verify",
            json={"payload": payload},
            headers=headers,
            timeout=30,
        )
        return response.text


class NorySettlePaymentTool(BaseTool):
    """Tool for settling a payment on-chain.

    Use this to submit a verified payment transaction to the blockchain.
    Settlement typically completes in under 400ms.

    Example usage:
        .. code-block:: python

            tool = NorySettlePaymentTool()
            result = tool.run({"payload": "base64_encoded_payload"})
    """

    name: str = "nory_settle_payment"
    description: str = (
        "Submit a verified payment to the blockchain for settlement. "
        "Returns transaction ID upon success (~400ms settlement time)."
    )
    args_schema: Type[BaseModel] = SettlePaymentInput
    api_key: Optional[SecretStr] = None

    @classmethod
    def from_api_key(cls, api_key: str, **kwargs: Any) -> NorySettlePaymentTool:
        """Create a tool from an API key.

        Args:
            api_key: The Nory API key to use.
            **kwargs: Any additional kwargs to pass to the tool.

        Returns:
            A configured tool instance.
        """
        return cls(api_key=SecretStr(api_key), **kwargs)

    def _run(
        self,
        payload: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Settle a payment on-chain.

        Args:
            payload: Base64-encoded payment payload.
            run_manager: Callback manager (optional).

        Returns:
            JSON string with settlement result and transaction ID.
        """
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key.get_secret_value()}"

        response = requests.post(
            f"{NORY_API_BASE}/api/x402/settle",
            json={"payload": payload},
            headers=headers,
            timeout=30,
        )
        return response.text


class NoryTransactionLookupTool(BaseTool):
    """Tool for looking up transaction status.

    Use this to check the status of a previously submitted payment.

    Example usage:
        .. code-block:: python

            tool = NoryTransactionLookupTool()
            result = tool.run({
                "transaction_id": "abc123...",
                "network": "solana-mainnet"
            })
    """

    name: str = "nory_get_transaction"
    description: str = (
        "Look up the status and details of a transaction by ID or signature."
    )
    args_schema: Type[BaseModel] = TransactionLookupInput
    api_key: Optional[SecretStr] = None

    @classmethod
    def from_api_key(cls, api_key: str, **kwargs: Any) -> NoryTransactionLookupTool:
        """Create a tool from an API key.

        Args:
            api_key: The Nory API key to use.
            **kwargs: Any additional kwargs to pass to the tool.

        Returns:
            A configured tool instance.
        """
        return cls(api_key=SecretStr(api_key), **kwargs)

    def _run(
        self,
        transaction_id: str,
        network: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Look up a transaction.

        Args:
            transaction_id: Transaction ID or signature.
            network: Network ID (e.g., solana-mainnet).
            run_manager: Callback manager (optional).

        Returns:
            JSON string with transaction details.
        """
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key.get_secret_value()}"

        response = requests.get(
            f"{NORY_API_BASE}/api/x402/transactions/{transaction_id}",
            params={"network": network},
            headers=headers,
            timeout=30,
        )
        return response.text


class NoryHealthCheckTool(BaseTool):
    """Tool for checking Nory service health.

    Use this to verify the payment service is operational.

    Example usage:
        .. code-block:: python

            tool = NoryHealthCheckTool()
            result = tool.run({})
    """

    name: str = "nory_health_check"
    description: str = (
        "Check health status of Nory x402 payment service and supported networks."
    )

    def _run(
        self,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Check service health.

        Args:
            run_manager: Callback manager (optional).

        Returns:
            JSON string with health status.
        """
        response = requests.get(f"{NORY_API_BASE}/api/x402/health", timeout=30)
        return response.text
