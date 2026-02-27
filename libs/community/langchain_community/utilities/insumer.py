"""Util that calls The Insumer Model On-Chain Verification API.

Docs: https://insumermodel.com/developers/
API reference: https://insumermodel.com/llms-full.txt
OpenAPI spec: https://insumermodel.com/openapi.yaml
"""

import json
from typing import Any, Optional

import requests
from langchain_core.utils import get_from_dict_or_env
from pydantic import BaseModel

INSUMER_BASE_URL = "https://us-central1-insumer-merchant.cloudfunctions.net/insumerApi/v1"


class InsumerAPIWrapper(BaseModel):
    """Wrapper for The Insumer Model Attestation API.

    Privacy-preserving on-chain verification across 31 blockchains.
    Verifies token balances, NFT ownership, and EAS attestations
    without exposing actual wallet balances.

    Args:
        insumer_api_key: API key in format ``insr_live_`` + 40 hex chars.
            Can also be set via ``INSUMER_API_KEY`` environment variable.
            Get a free key at https://insumermodel.com/developers/
    """

    insumer_api_key: Optional[str] = None

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        self.insumer_api_key = get_from_dict_or_env(
            data, "insumer_api_key", "INSUMER_API_KEY"
        )

    @property
    def _headers(self) -> dict[str, str]:
        if self.insumer_api_key is None:
            msg = (
                "API key is required for InsumerAPIWrapper. "
                "Provide it via insumer_api_key parameter or "
                "INSUMER_API_KEY environment variable. "
                "Get a free key at https://insumermodel.com/developers/"
            )
            raise ValueError(msg)
        return {
            "X-API-Key": self.insumer_api_key,
            "Content-Type": "application/json",
        }

    def attest(
        self,
        conditions: list[dict[str, Any]],
        wallet: Optional[str] = None,
        solana_wallet: Optional[str] = None,
        proof: Optional[str] = None,
    ) -> str:
        """Create a privacy-preserving on-chain attestation.

        Verifies 1-10 conditions and returns a signed true/false result.
        Costs 1 attestation credit (standard) or 2 credits (with
        proof="merkle").

        Args:
            conditions: Condition dicts with type, contractAddress, chainId,
                threshold, decimals, and label fields. For EAS attestations,
                use type "eas_attestation" with template or schemaId.
            wallet: EVM wallet address (0x...).
            solana_wallet: Solana wallet address (base58).
            proof: Set to "merkle" for EIP-1186 Merkle storage proofs.
                Available for token_balance conditions on RPC chains only.
                Costs 2 credits. Reveals raw balance to the caller.

        Returns:
            JSON string with attestation results and ECDSA signature.
            When proof="merkle", each result includes a proof object with
            accountProof, storageProof, storageHash, blockNumber, and
            mappingSlot fields.
        """
        body: dict[str, Any] = {"conditions": conditions}
        if wallet:
            body["wallet"] = wallet
        if solana_wallet:
            body["solanaWallet"] = solana_wallet
        if proof:
            body["proof"] = proof
        response = requests.post(
            f"{INSUMER_BASE_URL}/attest",
            headers=self._headers,
            json=body,
            timeout=30,
        )
        return json.dumps(response.json())

    def get_credits(self) -> str:
        """Check attestation credit balance for the API key.

        Returns:
            JSON string with credit balance, tier, and daily limit.
        """
        response = requests.get(
            f"{INSUMER_BASE_URL}/credits",
            headers=self._headers,
            timeout=30,
        )
        return json.dumps(response.json())

    def list_merchants(
        self,
        token: Optional[str] = None,
        verified: Optional[bool] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> str:
        """List merchants in the public directory.

        Args:
            token: Filter by accepted token symbol.
            verified: Filter by domain verification status.
            limit: Results per page (max 200).
            offset: Pagination offset.

        Returns:
            JSON string with merchant list.
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if token:
            params["token"] = token
        if verified is not None:
            params["verified"] = str(verified).lower()
        response = requests.get(
            f"{INSUMER_BASE_URL}/merchants",
            headers=self._headers,
            params=params,
            timeout=30,
        )
        return json.dumps(response.json())

    def get_merchant(self, merchant_id: str) -> str:
        """Get full public merchant profile with tier structures.

        Args:
            merchant_id: Merchant ID.

        Returns:
            JSON string with merchant details.
        """
        response = requests.get(
            f"{INSUMER_BASE_URL}/merchants/{merchant_id}",
            headers=self._headers,
            timeout=30,
        )
        return json.dumps(response.json())

    def list_tokens(
        self,
        chain: Optional[Any] = None,
        symbol: Optional[str] = None,
        asset_type: Optional[str] = None,
    ) -> str:
        """List registered tokens and NFT collections.

        Args:
            chain: Filter by chain ID (e.g. 1 for Ethereum).
            symbol: Filter by token symbol.
            asset_type: Filter by type: "token" or "nft".

        Returns:
            JSON string with token and NFT registry.
        """
        params: dict[str, Any] = {}
        if chain is not None:
            params["chain"] = chain
        if symbol:
            params["symbol"] = symbol
        if asset_type:
            params["type"] = asset_type
        response = requests.get(
            f"{INSUMER_BASE_URL}/tokens",
            headers=self._headers,
            params=params,
            timeout=30,
        )
        return json.dumps(response.json())

    def check_discount(
        self,
        merchant_id: str,
        wallet: Optional[str] = None,
        solana_wallet: Optional[str] = None,
    ) -> str:
        """Calculate discount for a wallet at a merchant.

        Free to call, no credits consumed.

        Args:
            merchant_id: Merchant ID.
            wallet: EVM wallet address.
            solana_wallet: Solana wallet address.

        Returns:
            JSON string with discount calculation.
        """
        params: dict[str, Any] = {"merchant": merchant_id}
        if wallet:
            params["wallet"] = wallet
        if solana_wallet:
            params["solanaWallet"] = solana_wallet
        response = requests.get(
            f"{INSUMER_BASE_URL}/discount/check",
            headers=self._headers,
            params=params,
            timeout=30,
        )
        return json.dumps(response.json())

    def verify(
        self,
        merchant_id: str,
        wallet: Optional[str] = None,
        solana_wallet: Optional[str] = None,
    ) -> str:
        """Create a signed discount code (INSR-XXXXX), valid 30 minutes.

        Costs 1 merchant credit.

        Args:
            merchant_id: Merchant ID.
            wallet: EVM wallet address.
            solana_wallet: Solana wallet address.

        Returns:
            JSON string with verification code and signature.
        """
        body: dict[str, Any] = {"merchantId": merchant_id}
        if wallet:
            body["wallet"] = wallet
        if solana_wallet:
            body["solanaWallet"] = solana_wallet
        response = requests.post(
            f"{INSUMER_BASE_URL}/verify",
            headers=self._headers,
            json=body,
            timeout=30,
        )
        return json.dumps(response.json())

    def get_jwks(self) -> str:
        """Get the JWKS containing InsumerAPI's ECDSA P-256 public signing key.

        No authentication required. The ``kid`` field matches the ``kid`` in
        attestation responses, enabling automatic key rotation.

        Returns:
            JSON string with JWKS document.
        """
        response = requests.get(
            f"{INSUMER_BASE_URL}/jwks",
            timeout=30,
        )
        return json.dumps(response.json())

    def get_compliance_templates(self) -> str:
        """List available EAS compliance templates.

        Returns pre-configured schema IDs for KYC/identity providers
        (e.g. Coinbase Verifications on Base). No auth or credits required.

        Returns:
            JSON string with available compliance templates.
        """
        response = requests.get(
            f"{INSUMER_BASE_URL}/compliance/templates",
            timeout=30,
        )
        return json.dumps(response.json())

    def wallet_trust(
        self,
        wallet: str,
        solana_wallet: Optional[str] = None,
        proof: Optional[str] = None,
    ) -> str:
        """Generate a structured wallet trust fact profile.

        Checks 17 curated conditions across stablecoins (7 chains),
        governance tokens (4), NFTs (3), and staking (3). Costs 3 credits
        (standard) or 6 credits (with proof="merkle").

        Args:
            wallet: EVM wallet address (0x...) to profile.
            solana_wallet: Solana wallet address (base58).
            proof: Set to "merkle" for EIP-1186 Merkle storage proofs.

        Returns:
            JSON string with trust profile and ECDSA signature.
        """
        body: dict[str, Any] = {"wallet": wallet}
        if solana_wallet:
            body["solanaWallet"] = solana_wallet
        if proof:
            body["proof"] = proof
        response = requests.post(
            f"{INSUMER_BASE_URL}/trust",
            headers=self._headers,
            json=body,
            timeout=30,
        )
        return json.dumps(response.json())

    def batch_wallet_trust(
        self,
        wallets: list[dict[str, Any]],
        proof: Optional[str] = None,
    ) -> str:
        """Generate trust profiles for up to 10 wallets in one request.

        Shared block fetches make this 5-8x faster than sequential calls.
        Supports partial success. Credits only charged for successful profiles.

        Args:
            wallets: List of 1-10 dicts, each with ``wallet`` (required)
                and optional ``solanaWallet``.
            proof: Set to "merkle" for Merkle storage proofs (6 credits/wallet).

        Returns:
            JSON string with results array and summary.
        """
        body: dict[str, Any] = {"wallets": wallets}
        if proof:
            body["proof"] = proof
        response = requests.post(
            f"{INSUMER_BASE_URL}/trust/batch",
            headers=self._headers,
            json=body,
            timeout=60,
        )
        return json.dumps(response.json())

    def buy_credits(
        self,
        tx_hash: str,
        chain_id: Any,
        amount: float,
    ) -> str:
        """Buy verification credits with USDC.

        Rate: 25 credits per 1 USDC. Minimum 5 USDC.

        Args:
            tx_hash: USDC transaction hash.
            chain_id: USDC chain (1, 8453, 137, 42161, 10, 56, 43114, or "solana").
            amount: USDC amount sent (minimum 5).

        Returns:
            JSON string with credit purchase confirmation.
        """
        response = requests.post(
            f"{INSUMER_BASE_URL}/credits/buy",
            headers=self._headers,
            json={"txHash": tx_hash, "chainId": chain_id, "amount": amount},
            timeout=30,
        )
        return json.dumps(response.json())

    def confirm_payment(
        self,
        code: str,
        tx_hash: str,
        chain_id: Any,
        amount: Any,
    ) -> str:
        """Confirm USDC payment for a discount code.

        Args:
            code: Verification code from verify (e.g. INSR-A7K3M).
            tx_hash: On-chain transaction hash or Solana signature.
            chain_id: USDC chain (1, 8453, 137, 42161, 10, 56, 43114, or "solana").
            amount: USDC amount sent.

        Returns:
            JSON string with payment confirmation.
        """
        response = requests.post(
            f"{INSUMER_BASE_URL}/payment/confirm",
            headers=self._headers,
            json={
                "code": code,
                "txHash": tx_hash,
                "chainId": chain_id,
                "amount": amount,
            },
            timeout=30,
        )
        return json.dumps(response.json())

    def create_merchant(
        self,
        company_name: str,
        company_id: str,
        location: Optional[str] = None,
    ) -> str:
        """Create a new merchant. Receives 100 free credits. Max 10 per API key.

        Args:
            company_name: Company display name (max 100 characters).
            company_id: Unique merchant ID (2-50 chars, alphanumeric/dashes/underscores).
            location: City or region (max 200 characters).

        Returns:
            JSON string with created merchant details.
        """
        body: dict[str, Any] = {
            "companyName": company_name,
            "companyId": company_id,
        }
        if location:
            body["location"] = location
        response = requests.post(
            f"{INSUMER_BASE_URL}/merchants",
            headers=self._headers,
            json=body,
            timeout=30,
        )
        return json.dumps(response.json())

    def get_merchant_status(self, merchant_id: str) -> str:
        """Get full private merchant details. Owner only.

        Args:
            merchant_id: Merchant ID.

        Returns:
            JSON string with credits, configs, directory status.
        """
        response = requests.get(
            f"{INSUMER_BASE_URL}/merchants/{merchant_id}/status",
            headers=self._headers,
            timeout=30,
        )
        return json.dumps(response.json())

    def configure_tokens(
        self,
        merchant_id: str,
        own_token: Optional[dict[str, Any]] = None,
        partner_tokens: Optional[list[dict[str, Any]]] = None,
    ) -> str:
        """Configure merchant token discount tiers. Max 8 tokens. Owner only.

        Args:
            merchant_id: Merchant ID.
            own_token: Merchant's own token config, or None to remove.
            partner_tokens: Partner token configurations.

        Returns:
            JSON string with updated token configuration.
        """
        body: dict[str, Any] = {}
        if own_token is not None:
            body["ownToken"] = own_token
        if partner_tokens is not None:
            body["partnerTokens"] = partner_tokens
        response = requests.put(
            f"{INSUMER_BASE_URL}/merchants/{merchant_id}/tokens",
            headers=self._headers,
            json=body,
            timeout=30,
        )
        return json.dumps(response.json())

    def configure_nfts(
        self,
        merchant_id: str,
        nft_collections: list[dict[str, Any]],
    ) -> str:
        """Configure NFT collections that grant discounts. Max 4. Owner only.

        Args:
            merchant_id: Merchant ID.
            nft_collections: NFT collection configs (0-4).

        Returns:
            JSON string with updated NFT configuration.
        """
        response = requests.put(
            f"{INSUMER_BASE_URL}/merchants/{merchant_id}/nfts",
            headers=self._headers,
            json={"nftCollections": nft_collections},
            timeout=30,
        )
        return json.dumps(response.json())

    def configure_settings(
        self,
        merchant_id: str,
        discount_mode: Optional[str] = None,
        discount_cap: Optional[int] = None,
        usdc_payment: Optional[dict[str, Any]] = None,
    ) -> str:
        """Update merchant settings. All fields optional. Owner only.

        Args:
            merchant_id: Merchant ID.
            discount_mode: "highest" or "stack".
            discount_cap: Maximum total discount percentage (1-100).
            usdc_payment: USDC payment settings dict, or None to disable.

        Returns:
            JSON string with updated settings.
        """
        body: dict[str, Any] = {}
        if discount_mode is not None:
            body["discountMode"] = discount_mode
        if discount_cap is not None:
            body["discountCap"] = discount_cap
        if usdc_payment is not None:
            body["usdcPayment"] = usdc_payment
        response = requests.put(
            f"{INSUMER_BASE_URL}/merchants/{merchant_id}/settings",
            headers=self._headers,
            json=body,
            timeout=30,
        )
        return json.dumps(response.json())

    def publish_directory(self, merchant_id: str) -> str:
        """Publish or refresh merchant listing in the public directory. Owner only.

        Args:
            merchant_id: Merchant ID.

        Returns:
            JSON string with publication confirmation.
        """
        response = requests.post(
            f"{INSUMER_BASE_URL}/merchants/{merchant_id}/directory",
            headers=self._headers,
            json={},
            timeout=30,
        )
        return json.dumps(response.json())

    def buy_merchant_credits(
        self,
        merchant_id: str,
        tx_hash: str,
        chain_id: Any,
        amount: float,
    ) -> str:
        """Buy merchant verification credits with USDC. Owner only.

        Rate: 25 credits per 1 USDC. Minimum 5 USDC.

        Args:
            merchant_id: Merchant ID.
            tx_hash: USDC transaction hash.
            chain_id: USDC chain (1, 8453, 137, 42161, 10, 56, 43114, or "solana").
            amount: USDC amount sent (minimum 5).

        Returns:
            JSON string with credit purchase confirmation.
        """
        response = requests.post(
            f"{INSUMER_BASE_URL}/merchants/{merchant_id}/credits",
            headers=self._headers,
            json={"txHash": tx_hash, "chainId": chain_id, "amount": amount},
            timeout=30,
        )
        return json.dumps(response.json())

    def run(self, mode: str, **kwargs: Any) -> str:
        """Run a specific API operation by mode.

        Args:
            mode: Operation to run.
            **kwargs: Arguments for the operation.

        Returns:
            JSON string with API response.
        """
        if mode == "attest":
            return self.attest(
                conditions=kwargs["conditions"],
                wallet=kwargs.get("wallet"),
                solana_wallet=kwargs.get("solana_wallet"),
                proof=kwargs.get("proof"),
            )
        elif mode == "get_credits":
            return self.get_credits()
        elif mode == "list_merchants":
            return self.list_merchants(
                token=kwargs.get("token"),
                verified=kwargs.get("verified"),
                limit=kwargs.get("limit", 50),
                offset=kwargs.get("offset", 0),
            )
        elif mode == "list_tokens":
            return self.list_tokens(
                chain=kwargs.get("chain"),
                symbol=kwargs.get("symbol"),
                asset_type=kwargs.get("asset_type"),
            )
        elif mode == "check_discount":
            return self.check_discount(
                merchant_id=kwargs["merchant_id"],
                wallet=kwargs.get("wallet"),
                solana_wallet=kwargs.get("solana_wallet"),
            )
        elif mode == "verify":
            return self.verify(
                merchant_id=kwargs["merchant_id"],
                wallet=kwargs.get("wallet"),
                solana_wallet=kwargs.get("solana_wallet"),
            )
        elif mode == "get_jwks":
            return self.get_jwks()
        elif mode == "get_compliance_templates":
            return self.get_compliance_templates()
        elif mode == "wallet_trust":
            return self.wallet_trust(
                wallet=kwargs["wallet"],
                solana_wallet=kwargs.get("solana_wallet"),
                proof=kwargs.get("proof"),
            )
        elif mode == "batch_wallet_trust":
            return self.batch_wallet_trust(
                wallets=kwargs["wallets"],
                proof=kwargs.get("proof"),
            )
        elif mode == "get_merchant":
            return self.get_merchant(merchant_id=kwargs["merchant_id"])
        elif mode == "buy_credits":
            return self.buy_credits(
                tx_hash=kwargs["tx_hash"],
                chain_id=kwargs["chain_id"],
                amount=kwargs["amount"],
            )
        elif mode == "confirm_payment":
            return self.confirm_payment(
                code=kwargs["code"],
                tx_hash=kwargs["tx_hash"],
                chain_id=kwargs["chain_id"],
                amount=kwargs["amount"],
            )
        elif mode == "create_merchant":
            return self.create_merchant(
                company_name=kwargs["company_name"],
                company_id=kwargs["company_id"],
                location=kwargs.get("location"),
            )
        elif mode == "get_merchant_status":
            return self.get_merchant_status(merchant_id=kwargs["merchant_id"])
        elif mode == "configure_tokens":
            return self.configure_tokens(
                merchant_id=kwargs["merchant_id"],
                own_token=kwargs.get("own_token"),
                partner_tokens=kwargs.get("partner_tokens"),
            )
        elif mode == "configure_nfts":
            return self.configure_nfts(
                merchant_id=kwargs["merchant_id"],
                nft_collections=kwargs["nft_collections"],
            )
        elif mode == "configure_settings":
            return self.configure_settings(
                merchant_id=kwargs["merchant_id"],
                discount_mode=kwargs.get("discount_mode"),
                discount_cap=kwargs.get("discount_cap"),
                usdc_payment=kwargs.get("usdc_payment"),
            )
        elif mode == "publish_directory":
            return self.publish_directory(merchant_id=kwargs["merchant_id"])
        elif mode == "buy_merchant_credits":
            return self.buy_merchant_credits(
                merchant_id=kwargs["merchant_id"],
                tx_hash=kwargs["tx_hash"],
                chain_id=kwargs["chain_id"],
                amount=kwargs["amount"],
            )
        else:
            msg = f"Invalid mode {mode} for Insumer API."
            raise ValueError(msg)
