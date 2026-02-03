"""Integration tests for ClawPrint tools against the live API.

These tests require network access and hit https://clawprint.io.
Run with: CLAWPRINT_INTEGRATION=1 pytest tests/integration_tests/tools/test_clawprint.py
"""

from __future__ import annotations

import json
import os

import pytest

from langchain_community.tools.clawprint._client import ClawPrintClient
from langchain_community.tools.clawprint.tool import (
    ClawPrintDomainsTool,
    ClawPrintGetAgentTool,
    ClawPrintSearchTool,
    ClawPrintToolkit,
    ClawPrintTrustTool,
)

requires_clawprint = pytest.mark.skipif(
    not os.environ.get("CLAWPRINT_INTEGRATION"),
    reason="Set CLAWPRINT_INTEGRATION=1 to run live integration tests",
)


@pytest.fixture
def client() -> ClawPrintClient:
    """Live client pointed at production."""
    return ClawPrintClient(base_url="https://clawprint.io", timeout=15)


@requires_clawprint
class TestSearchIntegration:
    def test_search_returns_agents(self, client: ClawPrintClient) -> None:
        tool = ClawPrintSearchTool(client=client)
        result = tool.invoke({"query": "security"})
        parsed = json.loads(result)
        assert "agents" in parsed or isinstance(parsed, list)

    def test_search_with_domain_filter(self, client: ClawPrintClient) -> None:
        tool = ClawPrintSearchTool(client=client)
        result = tool.invoke({"query": "code", "domain": "security"})
        # Should not raise — even if empty
        assert isinstance(result, str)

    def test_search_with_trust_filter(self, client: ClawPrintClient) -> None:
        tool = ClawPrintSearchTool(client=client)
        result = tool.invoke({"query": "agent", "min_trust": 0.5})
        assert isinstance(result, str)


@requires_clawprint
class TestGetAgentIntegration:
    def test_get_known_agent(self, client: ClawPrintClient) -> None:
        tool = ClawPrintGetAgentTool(client=client)
        result = tool.invoke({"handle": "sentinel"})
        parsed = json.loads(result)
        assert "identity" in parsed or "handle" in parsed

    def test_get_unknown_agent(self, client: ClawPrintClient) -> None:
        tool = ClawPrintGetAgentTool(client=client)
        result = tool.invoke({"handle": "nonexistent-agent-xyz-999"})
        # Should return error string, not raise
        assert "Error" in result or "error" in result.lower()


@requires_clawprint
class TestDomainsIntegration:
    def test_list_domains(self, client: ClawPrintClient) -> None:
        tool = ClawPrintDomainsTool(client=client)
        result = tool.invoke({})
        parsed = json.loads(result)
        assert isinstance(parsed, (dict, list))


@requires_clawprint
class TestTrustIntegration:
    def test_trust_known_agent(self, client: ClawPrintClient) -> None:
        tool = ClawPrintTrustTool(client=client)
        result = tool.invoke({"handle": "sentinel"})
        parsed = json.loads(result)
        assert "score" in parsed or "trust" in parsed


@requires_clawprint
class TestToolkitIntegration:
    def test_toolkit_get_tool(self, client: ClawPrintClient) -> None:
        toolkit = ClawPrintToolkit()
        search = toolkit.get_tool("clawprint_search")
        assert search.name == "clawprint_search"

    def test_toolkit_get_tool_invalid(self) -> None:
        toolkit = ClawPrintToolkit()
        with pytest.raises(KeyError, match="No tool named"):
            toolkit.get_tool("nonexistent_tool")
