"""AsterPay tools for AI agent trust verification and EUR settlement.

AsterPay provides KYA (Know Your Agent) — the trust and settlement layer
for AI agent commerce. These tools allow LangChain agents to:

- Verify the trustworthiness of other AI agents before transacting
- Check ERC-8004 on-chain identity registration
- Estimate USDC → EUR settlement via SEPA Instant

No API key required. Docs: https://asterpay.io

Setup:
    No authentication needed. The KYA trust score API is free.

    .. code-block:: bash

        pip install langchain-community

Instantiate:
    .. code-block:: python

        from langchain_community.tools.asterpay import AsterPayKYATrustScore

        tool = AsterPayKYATrustScore()
        result = tool.invoke({"address": "0x1234..."})

Key init args:
    api_wrapper: AsterPayAPIWrapper
        The API wrapper instance. Created automatically if not provided.
"""

from typing import Any, Optional, Type

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_community.utilities.asterpay import AsterPayAPIWrapper


class _AddressInput(BaseModel):
    """Input schema for tools that take an Ethereum address."""

    address: str = Field(
        description="Ethereum address (0x...) of the AI agent or wallet."
    )


class _SettlementInput(BaseModel):
    """Input schema for settlement estimate."""

    amount_usdc: float = Field(
        description="Amount in USDC to estimate EUR settlement for."
    )


class AsterPayKYATrustScore(BaseTool):
    """Check if an AI agent or wallet is trustworthy.

    Returns a KYA trust score (0-100), tier, and component breakdown
    including ERC-8004 identity, sanctions screening, on-chain activity,
    and behavioral signals.

    Free — no API key or payment required.

    Setup:
        No setup needed:

        .. code-block:: bash

            pip install langchain-community

    Instantiate:
        .. code-block:: python

            tool = AsterPayKYATrustScore()

    Invoke:
        .. code-block:: python

            tool.invoke({"address": "0x1234..."})

    """

    name: str = "asterpay_kya_trust_score"
    description: str = (
        "Check the trust score (0-100) of an AI agent or wallet address "
        "before transacting. Returns trust tier (Open/Verified/Trusted/"
        "Enterprise), component scores (ERC-8004 identity, sanctions, "
        "on-chain activity, behavioral), and risk assessment. "
        "Use to verify another agent is safe to pay. Free, no API key needed."
    )
    args_schema: Type[BaseModel] = _AddressInput
    api_wrapper: AsterPayAPIWrapper = Field(default_factory=AsterPayAPIWrapper)

    def _run(
        self,
        address: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> str:
        """Get KYA trust score for an address."""
        try:
            return self.api_wrapper.trust_score(address)
        except Exception as e:
            return repr(e)


class AsterPayKYAVerify(BaseTool):
    """Verify if an Ethereum address is a registered AI agent (ERC-8004).

    Checks on-chain registration on Base and returns agent ID, owner,
    and metadata URI.

    Free — no API key or payment required.

    Setup:
        .. code-block:: bash

            pip install langchain-community

    Instantiate:
        .. code-block:: python

            tool = AsterPayKYAVerify()

    Invoke:
        .. code-block:: python

            tool.invoke({"address": "0x1234..."})

    """

    name: str = "asterpay_kya_verify"
    description: str = (
        "Verify whether an Ethereum address is registered as an AI agent "
        "via ERC-8004 on Base. Returns verified true/false, agent ID, "
        "owner address, and metadata URI. Use before trusting an unknown "
        "agent. Free, no API key needed."
    )
    args_schema: Type[BaseModel] = _AddressInput
    api_wrapper: AsterPayAPIWrapper = Field(default_factory=AsterPayAPIWrapper)

    def _run(
        self,
        address: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> str:
        """Verify agent identity."""
        try:
            return self.api_wrapper.verify_agent(address)
        except Exception as e:
            return repr(e)


class AsterPayKYATier(BaseTool):
    """Get the trust tier and spending limit for an AI agent.

    Returns tier classification (Open, Verified, Trusted, Enterprise)
    and associated maximum per-transaction limits.

    Free — no API key or payment required.

    Setup:
        .. code-block:: bash

            pip install langchain-community

    Instantiate:
        .. code-block:: python

            tool = AsterPayKYATier()

    Invoke:
        .. code-block:: python

            tool.invoke({"address": "0x1234..."})

    """

    name: str = "asterpay_kya_tier"
    description: str = (
        "Get the trust tier and spending limit for an agent address. "
        "Returns tier (Open/Verified/Trusted/Enterprise) and max "
        "per-transaction limit. Use to know how much you can transact "
        "with an agent. Free, no API key needed."
    )
    args_schema: Type[BaseModel] = _AddressInput
    api_wrapper: AsterPayAPIWrapper = Field(default_factory=AsterPayAPIWrapper)

    def _run(
        self,
        address: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> str:
        """Get agent tier."""
        try:
            return self.api_wrapper.agent_tier(address)
        except Exception as e:
            return repr(e)


class AsterPayKYADeepAnalysis(BaseTool):
    """Run deep behavioral trust analysis on an agent address.

    Returns velocity patterns, behavioral signals, transaction history
    analysis, and a risk recommendation.

    Note: This endpoint requires a $0.01 USDC payment via the x402
    payment protocol.

    Setup:
        .. code-block:: bash

            pip install langchain-community

    Instantiate:
        .. code-block:: python

            tool = AsterPayKYADeepAnalysis()

    Invoke:
        .. code-block:: python

            tool.invoke({"address": "0x1234..."})

    """

    name: str = "asterpay_kya_deep_analysis"
    description: str = (
        "Deep behavioral trust analysis for an agent — velocity patterns, "
        "behavioral signals, and risk recommendation. Use when you need "
        "detailed risk intelligence before a large transaction. "
        "Costs $0.01 USDC via x402 payment protocol."
    )
    args_schema: Type[BaseModel] = _AddressInput
    api_wrapper: AsterPayAPIWrapper = Field(default_factory=AsterPayAPIWrapper)

    def _run(
        self,
        address: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> str:
        """Run deep analysis."""
        try:
            return self.api_wrapper.deep_analysis(address)
        except Exception as e:
            return repr(e)


class AsterPaySettlementEstimate(BaseTool):
    """Estimate USDC to EUR settlement via SEPA Instant.

    Returns estimated EUR amount, fees, exchange rate, and settlement
    time (typically under 10 seconds).

    Free — no API key or payment required.

    Setup:
        .. code-block:: bash

            pip install langchain-community

    Instantiate:
        .. code-block:: python

            tool = AsterPaySettlementEstimate()

    Invoke:
        .. code-block:: python

            tool.invoke({"amount_usdc": 100.0})

    """

    name: str = "asterpay_settlement_estimate"
    description: str = (
        "Estimate USDC → EUR settlement amount via SEPA Instant. "
        "Returns estimated EUR amount, fees, exchange rate, and "
        "settlement time. Use when you need to know how much EUR "
        "a USDC amount will settle to. Free, no API key needed."
    )
    args_schema: Type[BaseModel] = _SettlementInput
    api_wrapper: AsterPayAPIWrapper = Field(default_factory=AsterPayAPIWrapper)

    def _run(
        self,
        amount_usdc: float,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> str:
        """Get settlement estimate."""
        try:
            return self.api_wrapper.settlement_estimate(amount_usdc)
        except Exception as e:
            return repr(e)
