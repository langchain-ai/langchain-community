"""Wrapper for the AsterPay API.

AsterPay provides KYA (Know Your Agent) trust scoring and EUR settlement
for AI agent commerce. Free API — no key required for trust scores.

Docs: https://asterpay.io
API:  https://x402.asterpay.io
"""

from __future__ import annotations

import json
from typing import Any, Optional

import requests
from pydantic import BaseModel, Field


class AsterPayAPIWrapper(BaseModel):
    """Wrapper around the AsterPay REST API.

    No API key is required for KYA trust scoring and settlement estimates.

    Example:
        .. code-block:: python

            from langchain_community.utilities.asterpay import AsterPayAPIWrapper

            api = AsterPayAPIWrapper()
            result = api.trust_score("0x1234...")
    """

    base_url: str = Field(
        default="https://x402.asterpay.io",
        description="AsterPay API base URL.",
    )
    timeout: int = Field(
        default=30,
        description="Request timeout in seconds.",
    )

    def _get(self, path: str, params: Optional[dict[str, Any]] = None) -> dict:
        """Make a GET request to the AsterPay API."""
        url = f"{self.base_url}{path}"
        response = requests.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def _post(self, path: str, data: Optional[dict[str, Any]] = None) -> dict:
        """Make a POST request to the AsterPay API."""
        url = f"{self.base_url}{path}"
        response = requests.post(url, json=data, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def trust_score(self, address: str) -> str:
        """Get KYA trust score (0-100) for an agent or wallet address.

        Args:
            address: Ethereum address to score.

        Returns:
            JSON string with trust score, tier, and component breakdown.
        """
        result = self._get(f"/v1/kya/trust-score/{address}")
        return json.dumps(result, indent=2, default=str)

    def verify_agent(self, address: str) -> str:
        """Verify if an address is a registered AI agent (ERC-8004).

        Args:
            address: Ethereum address to verify.

        Returns:
            JSON string with verification result, agent ID, owner, metadata.
        """
        result = self._get(f"/v1/kya/verify/{address}")
        return json.dumps(result, indent=2, default=str)

    def agent_tier(self, address: str) -> str:
        """Get the trust tier for an agent address.

        Args:
            address: Ethereum address to check.

        Returns:
            JSON string with tier (Open/Verified/Trusted/Enterprise)
            and spending limits.
        """
        result = self._get(f"/v1/kya/tier/{address}")
        return json.dumps(result, indent=2, default=str)

    def deep_analysis(self, address: str) -> str:
        """Run deep behavioral trust analysis on an agent address.

        Note: This endpoint costs $0.01 USDC via x402 payment protocol.

        Args:
            address: Ethereum address to analyze.

        Returns:
            JSON string with velocity, behavioral signals, recommendation.
        """
        result = self._get(f"/v1/kya/deep-analysis/{address}")
        return json.dumps(result, indent=2, default=str)

    def settlement_estimate(self, amount_usdc: float) -> str:
        """Estimate USDC to EUR settlement via SEPA Instant.

        Args:
            amount_usdc: Amount in USDC to settle.

        Returns:
            JSON string with estimated EUR amount, fees, exchange rate,
            and estimated settlement time.
        """
        result = self._get(
            "/v1/settlement/estimate", params={"amount": amount_usdc}
        )
        return json.dumps(result, indent=2, default=str)
