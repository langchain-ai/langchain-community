"""Unit tests for Insumer Model tools and API wrapper."""

import json
from unittest.mock import patch

import pytest
import responses

from langchain_community.tools.insumer.attest import InsumerAttest
from langchain_community.tools.insumer.batch_wallet_trust import InsumerBatchWalletTrust
from langchain_community.tools.insumer.check_discount import InsumerCheckDiscount
from langchain_community.tools.insumer.compliance_templates import (
    InsumerComplianceTemplates,
)
from langchain_community.tools.insumer.jwks import InsumerJwks
from langchain_community.tools.insumer.validate_code import InsumerValidateCode
from langchain_community.tools.insumer.wallet_trust import InsumerWalletTrust
from langchain_community.utilities.insumer import INSUMER_BASE_URL, InsumerAPIWrapper

TEST_API_KEY = "insr_live_" + "a" * 40

# ---------------------------------------------------------------------------
# Mock response fixtures
# ---------------------------------------------------------------------------

MOCK_ATTEST_RESPONSE = {
    "ok": True,
    "data": {
        "attestation": {
            "id": "ATST-12345",
            "pass": True,
            "results": [
                {
                    "condition": 0,
                    "label": "Holds 100 USDC",
                    "met": True,
                    "type": "token_balance",
                    "chainId": 1,
                }
            ],
            "passCount": 1,
            "failCount": 0,
            "attestedAt": "2026-02-28T12:00:00.000Z",
            "expiresAt": "2026-02-28T12:30:00.000Z",
        },
        "sig": "MEQCIB..." + "A" * 60,
        "kid": "insumer-attest-v1",
    },
    "meta": {"creditsRemaining": 9, "creditsCharged": 1, "version": "1.0", "timestamp": "2026-02-28T12:00:00.000Z"},
}

MOCK_TRUST_RESPONSE = {
    "ok": True,
    "data": {
        "trust": {
            "id": "TRST-99999",
            "wallet": "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045",
            "conditionSetVersion": "v1",
            "dimensions": {
                "stablecoins": {"checks": [{"label": "USDC on Ethereum", "met": True, "chainId": 1}], "passCount": 2, "failCount": 5, "total": 7},
                "governance": {"checks": [], "passCount": 1, "failCount": 3, "total": 4},
                "nfts": {"checks": [], "passCount": 0, "failCount": 3, "total": 3},
                "staking": {"checks": [], "passCount": 0, "failCount": 3, "total": 3},
            },
            "summary": {"totalChecks": 17, "totalPassed": 3, "totalFailed": 14, "dimensionsWithActivity": 2, "dimensionsChecked": 4},
            "profiledAt": "2026-02-28T12:00:00.000Z",
            "expiresAt": "2026-02-28T12:30:00.000Z",
        },
        "sig": "MEQCIB..." + "B" * 60,
        "kid": "insumer-attest-v1",
    },
    "meta": {"creditsRemaining": 47, "creditsCharged": 3, "version": "1.0", "timestamp": "2026-02-28T12:00:00.000Z"},
}

MOCK_BATCH_TRUST_RESPONSE = {
    "ok": True,
    "data": {
        "results": [
            {
                "trust": {
                    "id": "TRST-11111",
                    "wallet": "0xAAA",
                    "conditionSetVersion": "v1",
                    "dimensions": {},
                    "summary": {"totalChecks": 17, "totalPassed": 0, "totalFailed": 17, "dimensionsWithActivity": 0, "dimensionsChecked": 4},
                    "profiledAt": "2026-02-28T12:00:00.000Z",
                    "expiresAt": "2026-02-28T12:30:00.000Z",
                },
                "sig": "sig1",
                "kid": "insumer-attest-v1",
            }
        ],
        "summary": {"requested": 1, "succeeded": 1, "failed": 0},
    },
    "meta": {"creditsRemaining": 47, "creditsCharged": 3, "version": "1.0", "timestamp": "2026-02-28T12:00:00.000Z"},
}

MOCK_CHECK_DISCOUNT_RESPONSE = {
    "ok": True,
    "data": {
        "eligible": True,
        "totalDiscount": 15,
        "discountMode": "highest",
        "breakdown": [{"symbol": "UNI", "tier": "Gold", "discount": 15, "chain": "Ethereum"}],
        "merchantId": "acme-corp",
        "merchantName": "Acme Corp",
        "chainsChecked": ["Ethereum"],
    },
    "meta": {"version": "1.0", "timestamp": "2026-02-28T12:00:00.000Z"},
}

MOCK_VALIDATE_CODE_RESPONSE = {
    "ok": True,
    "data": {
        "valid": True,
        "code": "INSR-A7K3M",
        "merchantId": "acme-corp",
        "discountPercent": 15,
        "expiresAt": "2026-02-28T12:30:00.000Z",
        "createdAt": "2026-02-28T12:00:00.000Z",
    },
    "meta": {"version": "1.0", "timestamp": "2026-02-28T12:00:00.000Z"},
}

MOCK_JWKS_RESPONSE = {
    "ok": True,
    "data": {
        "keys": [
            {
                "kty": "EC",
                "crv": "P-256",
                "x": "JtHPhDPnv8AfP0JSlGutxbOlxreV2Chey27Z76q3V2c",
                "y": "kn34HaxVSJfn8NxwNEBjjLkcrM_GDw1lgnqyADGuc4c",
                "use": "sig",
                "alg": "ES256",
                "kid": "insumer-attest-v1",
            }
        ]
    },
    "meta": {"version": "1.0", "timestamp": "2026-02-28T12:00:00.000Z"},
}

MOCK_COMPLIANCE_TEMPLATES_RESPONSE = {
    "ok": True,
    "data": {
        "templates": {
            "coinbase_verified_account": {
                "provider": "Coinbase",
                "description": "Coinbase Verified Account",
                "chainId": 8453,
                "chainName": "Base",
            }
        }
    },
    "meta": {"version": "1.0", "timestamp": "2026-02-28T12:00:00.000Z"},
}

TEST_WALLET = "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"
TEST_CONDITIONS = json.dumps(
    [{"chainId": 1, "label": "Holds 100 USDC", "threshold": 100, "type": "token_balance"}]
)


# ---------------------------------------------------------------------------
# InsumerAPIWrapper tests
# ---------------------------------------------------------------------------


class TestInsumerAPIWrapper:
    def _make_wrapper(self) -> InsumerAPIWrapper:
        return InsumerAPIWrapper(insumer_api_key=TEST_API_KEY)

    def test_headers_contain_api_key(self) -> None:
        wrapper = self._make_wrapper()
        assert wrapper._headers["X-API-Key"] == TEST_API_KEY
        assert wrapper._headers["Content-Type"] == "application/json"

    def test_missing_api_key_raises(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises((ValueError, Exception)):
                InsumerAPIWrapper(insumer_api_key=None)

    @responses.activate
    def test_attest(self) -> None:
        responses.add(
            responses.POST,
            f"{INSUMER_BASE_URL}/attest",
            json=MOCK_ATTEST_RESPONSE,
            status=200,
        )
        wrapper = self._make_wrapper()
        result = wrapper.attest(
            conditions=[{"type": "token_balance", "chainId": 1, "threshold": 100}],
            wallet=TEST_WALLET,
        )
        parsed = json.loads(result)
        assert parsed["data"]["attestation"]["id"] == "ATST-12345"
        assert parsed["data"]["attestation"]["results"][0]["met"] is True

    @responses.activate
    def test_wallet_trust(self) -> None:
        responses.add(
            responses.POST,
            f"{INSUMER_BASE_URL}/trust",
            json=MOCK_TRUST_RESPONSE,
            status=200,
        )
        wrapper = self._make_wrapper()
        result = wrapper.wallet_trust(wallet=TEST_WALLET)
        parsed = json.loads(result)
        assert parsed["data"]["trust"]["id"] == "TRST-99999"
        assert "stablecoins" in parsed["data"]["trust"]["dimensions"]

    @responses.activate
    def test_check_discount(self) -> None:
        responses.add(
            responses.GET,
            f"{INSUMER_BASE_URL}/discount/check",
            json=MOCK_CHECK_DISCOUNT_RESPONSE,
            status=200,
        )
        wrapper = self._make_wrapper()
        result = wrapper.check_discount(merchant_id="acme-corp", wallet=TEST_WALLET)
        parsed = json.loads(result)
        assert parsed["data"]["totalDiscount"] == 15
        assert parsed["data"]["breakdown"][0]["tier"] == "Gold"

    @responses.activate
    def test_validate_code(self) -> None:
        responses.add(
            responses.GET,
            f"{INSUMER_BASE_URL}/codes/INSR-A7K3M",
            json=MOCK_VALIDATE_CODE_RESPONSE,
            status=200,
        )
        wrapper = self._make_wrapper()
        result = wrapper.validate_code(code="INSR-A7K3M")
        parsed = json.loads(result)
        assert parsed["data"]["valid"] is True
        assert parsed["data"]["discountPercent"] == 15

    @responses.activate
    def test_get_jwks(self) -> None:
        responses.add(
            responses.GET,
            f"{INSUMER_BASE_URL}/jwks",
            json=MOCK_JWKS_RESPONSE,
            status=200,
        )
        wrapper = self._make_wrapper()
        result = wrapper.get_jwks()
        parsed = json.loads(result)
        assert parsed["data"]["keys"][0]["kid"] == "insumer-attest-v1"
        assert parsed["data"]["keys"][0]["alg"] == "ES256"

    @responses.activate
    def test_get_compliance_templates(self) -> None:
        responses.add(
            responses.GET,
            f"{INSUMER_BASE_URL}/compliance/templates",
            json=MOCK_COMPLIANCE_TEMPLATES_RESPONSE,
            status=200,
        )
        wrapper = self._make_wrapper()
        result = wrapper.get_compliance_templates()
        parsed = json.loads(result)
        assert "coinbase_verified_account" in parsed["data"]["templates"]

    @responses.activate
    def test_batch_wallet_trust(self) -> None:
        responses.add(
            responses.POST,
            f"{INSUMER_BASE_URL}/trust/batch",
            json=MOCK_BATCH_TRUST_RESPONSE,
            status=200,
        )
        wrapper = self._make_wrapper()
        result = wrapper.batch_wallet_trust(wallets=[{"wallet": TEST_WALLET}])
        parsed = json.loads(result)
        assert parsed["data"]["summary"]["succeeded"] == 1
        assert len(parsed["data"]["results"]) == 1


# ---------------------------------------------------------------------------
# Tool class tests
# ---------------------------------------------------------------------------


class TestInsumerAttest:
    def _make_tool(self) -> InsumerAttest:
        wrapper = InsumerAPIWrapper(insumer_api_key=TEST_API_KEY)
        return InsumerAttest(api_wrapper=wrapper)

    @responses.activate
    def test_run_evm_wallet(self) -> None:
        responses.add(
            responses.POST,
            f"{INSUMER_BASE_URL}/attest",
            json=MOCK_ATTEST_RESPONSE,
            status=200,
        )
        tool = self._make_tool()
        result = tool._run(conditions=TEST_CONDITIONS, wallet=TEST_WALLET)
        parsed = json.loads(result)
        assert parsed["data"]["attestation"]["id"] == "ATST-12345"

    @responses.activate
    def test_run_xrpl_wallet(self) -> None:
        responses.add(
            responses.POST,
            f"{INSUMER_BASE_URL}/attest",
            json=MOCK_ATTEST_RESPONSE,
            status=200,
        )
        tool = self._make_tool()
        result = tool._run(
            conditions=json.dumps(
                [{"type": "token_balance", "chainId": "xrpl", "threshold": 10}]
            ),
            xrpl_wallet="ra8xqX4QhcogFfxpMxMByvFnXyxw9E8rzY",
        )
        assert json.loads(result)["data"]["attestation"]["id"] == "ATST-12345"
        # Confirm xrplWallet was sent in the request body
        body = json.loads(responses.calls[0].request.body)
        assert body["xrplWallet"] == "ra8xqX4QhcogFfxpMxMByvFnXyxw9E8rzY"

    def test_tool_name_and_mode(self) -> None:
        tool = self._make_tool()
        assert tool.name == "insumer_attest"
        assert tool.mode == "attest"


class TestInsumerWalletTrust:
    def _make_tool(self) -> InsumerWalletTrust:
        wrapper = InsumerAPIWrapper(insumer_api_key=TEST_API_KEY)
        return InsumerWalletTrust(api_wrapper=wrapper)

    @responses.activate
    def test_run(self) -> None:
        responses.add(
            responses.POST,
            f"{INSUMER_BASE_URL}/trust",
            json=MOCK_TRUST_RESPONSE,
            status=200,
        )
        tool = self._make_tool()
        result = tool._run(wallet=TEST_WALLET)
        parsed = json.loads(result)
        assert parsed["data"]["trust"]["id"] == "TRST-99999"
        assert "stablecoins" in parsed["data"]["trust"]["dimensions"]

    @responses.activate
    def test_run_with_xrpl(self) -> None:
        responses.add(
            responses.POST,
            f"{INSUMER_BASE_URL}/trust",
            json=MOCK_TRUST_RESPONSE,
            status=200,
        )
        tool = self._make_tool()
        tool._run(
            wallet=TEST_WALLET,
            xrpl_wallet="ra8xqX4QhcogFfxpMxMByvFnXyxw9E8rzY",
        )
        body = json.loads(responses.calls[0].request.body)
        assert body["xrplWallet"] == "ra8xqX4QhcogFfxpMxMByvFnXyxw9E8rzY"
        assert body["wallet"] == TEST_WALLET

    def test_tool_name_and_mode(self) -> None:
        tool = self._make_tool()
        assert tool.name == "insumer_wallet_trust"
        assert tool.mode == "wallet_trust"


class TestInsumerBatchWalletTrust:
    def _make_tool(self) -> InsumerBatchWalletTrust:
        wrapper = InsumerAPIWrapper(insumer_api_key=TEST_API_KEY)
        return InsumerBatchWalletTrust(api_wrapper=wrapper)

    @responses.activate
    def test_run(self) -> None:
        responses.add(
            responses.POST,
            f"{INSUMER_BASE_URL}/trust/batch",
            json=MOCK_BATCH_TRUST_RESPONSE,
            status=200,
        )
        tool = self._make_tool()
        result = tool._run(wallets=json.dumps([{"wallet": TEST_WALLET}]))
        parsed = json.loads(result)
        assert parsed["data"]["summary"]["requested"] == 1
        assert parsed["data"]["summary"]["succeeded"] == 1


class TestInsumerCheckDiscount:
    def _make_tool(self) -> InsumerCheckDiscount:
        wrapper = InsumerAPIWrapper(insumer_api_key=TEST_API_KEY)
        return InsumerCheckDiscount(api_wrapper=wrapper)

    @responses.activate
    def test_run(self) -> None:
        responses.add(
            responses.GET,
            f"{INSUMER_BASE_URL}/discount/check",
            json=MOCK_CHECK_DISCOUNT_RESPONSE,
            status=200,
        )
        tool = self._make_tool()
        result = tool._run(merchant_id="acme-corp", wallet=TEST_WALLET)
        parsed = json.loads(result)
        assert parsed["data"]["breakdown"][0]["tier"] == "Gold"
        assert parsed["data"]["totalDiscount"] == 15

    def test_tool_name(self) -> None:
        tool = self._make_tool()
        assert tool.name == "insumer_check_discount"


class TestInsumerValidateCode:
    def _make_tool(self) -> InsumerValidateCode:
        wrapper = InsumerAPIWrapper(insumer_api_key=TEST_API_KEY)
        return InsumerValidateCode(api_wrapper=wrapper)

    @responses.activate
    def test_run_valid_code(self) -> None:
        responses.add(
            responses.GET,
            f"{INSUMER_BASE_URL}/codes/INSR-A7K3M",
            json=MOCK_VALIDATE_CODE_RESPONSE,
            status=200,
        )
        tool = self._make_tool()
        result = tool._run(code="INSR-A7K3M")
        parsed = json.loads(result)
        assert parsed["data"]["valid"] is True
        assert parsed["data"]["code"] == "INSR-A7K3M"

    @responses.activate
    def test_run_invalid_code(self) -> None:
        responses.add(
            responses.GET,
            f"{INSUMER_BASE_URL}/codes/INSR-XXXXX",
            json={"ok": True, "data": {"valid": False, "code": "INSR-XXXXX", "reason": "not_found"}, "meta": {"version": "1.0", "timestamp": "2026-02-28T12:00:00.000Z"}},
            status=200,
        )
        tool = self._make_tool()
        result = tool._run(code="INSR-XXXXX")
        parsed = json.loads(result)
        assert parsed["data"]["valid"] is False


class TestInsumerJwks:
    def _make_tool(self) -> InsumerJwks:
        wrapper = InsumerAPIWrapper(insumer_api_key=TEST_API_KEY)
        return InsumerJwks(api_wrapper=wrapper)

    @responses.activate
    def test_run(self) -> None:
        responses.add(
            responses.GET,
            f"{INSUMER_BASE_URL}/jwks",
            json=MOCK_JWKS_RESPONSE,
            status=200,
        )
        tool = self._make_tool()
        result = tool._run()
        parsed = json.loads(result)
        assert "keys" in parsed["data"]
        assert parsed["data"]["keys"][0]["alg"] == "ES256"

    def test_tool_name(self) -> None:
        tool = self._make_tool()
        assert tool.name == "insumer_jwks"


class TestInsumerComplianceTemplates:
    def _make_tool(self) -> InsumerComplianceTemplates:
        wrapper = InsumerAPIWrapper(insumer_api_key=TEST_API_KEY)
        return InsumerComplianceTemplates(api_wrapper=wrapper)

    @responses.activate
    def test_run(self) -> None:
        responses.add(
            responses.GET,
            f"{INSUMER_BASE_URL}/compliance/templates",
            json=MOCK_COMPLIANCE_TEMPLATES_RESPONSE,
            status=200,
        )
        tool = self._make_tool()
        result = tool._run()
        parsed = json.loads(result)
        assert "coinbase_verified_account" in parsed["data"]["templates"]

    def test_tool_name(self) -> None:
        tool = self._make_tool()
        assert tool.name == "insumer_compliance_templates"
