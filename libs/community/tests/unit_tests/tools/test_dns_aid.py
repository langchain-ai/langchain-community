"""Unit tests for DNS-AID tools."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from langchain_community.tools.dns_aid.tool import (
    DnsAidDiscoverTool,
    DnsAidPublishTool,
    DnsAidUnpublishTool,
    _import_create_backend,
    _import_dns_aid,
)


class TestDnsAidPublishTool:
    """Tests for DnsAidPublishTool."""

    def test_instantiation_defaults(self) -> None:
        tool = DnsAidPublishTool()
        assert tool.name == "dns_aid_publish"
        assert tool.backend_name is None
        assert tool.backend is None

    def test_instantiation_with_backend_name(self) -> None:
        tool = DnsAidPublishTool(backend_name="route53")
        assert tool.backend_name == "route53"

    def test_instantiation_with_backend(self) -> None:
        mock_backend = MagicMock()
        tool = DnsAidPublishTool(backend=mock_backend)
        assert tool.backend is mock_backend

    def test_get_backend_prefers_backend_over_name(self) -> None:
        mock_backend = MagicMock()
        tool = DnsAidPublishTool(backend=mock_backend, backend_name="route53")
        assert tool._get_backend() is mock_backend

    def test_get_backend_returns_none_when_no_config(self) -> None:
        tool = DnsAidPublishTool()
        assert tool._get_backend() is None

    def test_args_schema(self) -> None:
        tool = DnsAidPublishTool()
        schema = tool.args_schema.model_json_schema()
        assert "name" in schema["properties"]
        assert "domain" in schema["properties"]
        assert "endpoint" in schema["properties"]
        assert "protocol" in schema["properties"]
        assert "capabilities" in schema["properties"]

    @pytest.mark.asyncio
    async def test_arun_calls_dns_aid_publish(self) -> None:
        mock_result = MagicMock()
        mock_result.model_dump.return_value = {
            "success": True,
            "agent": {"name": "test-agent", "domain": "example.com"},
            "records_created": ["_test-agent._mcp._agents.example.com"],
        }

        with patch(
            "langchain_community.tools.dns_aid.tool._import_dns_aid"
        ) as mock_import:
            mock_dns_aid = MagicMock()
            mock_dns_aid.publish = AsyncMock(return_value=mock_result)
            mock_import.return_value = mock_dns_aid

            tool = DnsAidPublishTool()
            result = await tool._arun(
                name="test-agent",
                domain="example.com",
                protocol="mcp",
                endpoint="mcp.example.com",
                capabilities=["search"],
            )

        parsed = json.loads(result)
        assert parsed["success"] is True
        mock_dns_aid.publish.assert_awaited_once()
        call_kwargs = mock_dns_aid.publish.call_args.kwargs
        assert call_kwargs["name"] == "test-agent"
        assert call_kwargs["domain"] == "example.com"
        assert call_kwargs["protocol"] == "mcp"
        assert call_kwargs["endpoint"] == "mcp.example.com"
        assert call_kwargs["capabilities"] == ["search"]


class TestDnsAidDiscoverTool:
    """Tests for DnsAidDiscoverTool."""

    def test_instantiation(self) -> None:
        tool = DnsAidDiscoverTool()
        assert tool.name == "dns_aid_discover"

    def test_args_schema(self) -> None:
        tool = DnsAidDiscoverTool()
        schema = tool.args_schema.model_json_schema()
        assert "domain" in schema["properties"]
        assert "protocol" in schema["properties"]
        assert "require_dnssec" in schema["properties"]

    @pytest.mark.asyncio
    async def test_arun_calls_dns_aid_discover(self) -> None:
        mock_result = MagicMock()
        mock_result.model_dump.return_value = {
            "query": "_agents.example.com",
            "domain": "example.com",
            "agents": [
                {"name": "agent-1", "endpoint_url": "https://agent1.example.com:443"}
            ],
            "dnssec_validated": False,
            "count": 1,
        }

        with patch(
            "langchain_community.tools.dns_aid.tool._import_dns_aid"
        ) as mock_import:
            mock_dns_aid = MagicMock()
            mock_dns_aid.discover = AsyncMock(return_value=mock_result)
            mock_import.return_value = mock_dns_aid

            tool = DnsAidDiscoverTool()
            result = await tool._arun(domain="example.com", protocol="mcp")

        parsed = json.loads(result)
        assert parsed["count"] == 1
        assert len(parsed["agents"]) == 1
        mock_dns_aid.discover.assert_awaited_once()
        call_kwargs = mock_dns_aid.discover.call_args.kwargs
        assert call_kwargs["domain"] == "example.com"
        assert call_kwargs["protocol"] == "mcp"


class TestDnsAidUnpublishTool:
    """Tests for DnsAidUnpublishTool."""

    def test_instantiation(self) -> None:
        tool = DnsAidUnpublishTool()
        assert tool.name == "dns_aid_unpublish"

    def test_instantiation_with_backend(self) -> None:
        tool = DnsAidUnpublishTool(backend_name="cloudflare")
        assert tool.backend_name == "cloudflare"

    @pytest.mark.asyncio
    async def test_arun_unpublish_success(self) -> None:
        with patch(
            "langchain_community.tools.dns_aid.tool._import_dns_aid"
        ) as mock_import:
            mock_dns_aid = MagicMock()
            mock_dns_aid.unpublish = AsyncMock(return_value=True)
            mock_import.return_value = mock_dns_aid

            tool = DnsAidUnpublishTool()
            result = await tool._arun(
                name="old-agent", domain="example.com", protocol="mcp"
            )

        parsed = json.loads(result)
        assert parsed["success"] is True
        assert "unpublished" in parsed["message"]

    @pytest.mark.asyncio
    async def test_arun_unpublish_not_found(self) -> None:
        with patch(
            "langchain_community.tools.dns_aid.tool._import_dns_aid"
        ) as mock_import:
            mock_dns_aid = MagicMock()
            mock_dns_aid.unpublish = AsyncMock(return_value=False)
            mock_import.return_value = mock_dns_aid

            tool = DnsAidUnpublishTool()
            result = await tool._arun(
                name="missing-agent", domain="example.com", protocol="mcp"
            )

        parsed = json.loads(result)
        assert parsed["success"] is False
        assert "not found" in parsed["message"]


class TestDnsAidImportErrors:
    """Tests for lazy import error handling."""

    def test_import_dns_aid_raises_when_missing(self) -> None:
        with patch.dict("sys.modules", {"dns_aid": None}):
            with pytest.raises(ImportError, match="dns-aid"):
                _import_dns_aid()

    def test_import_create_backend_raises_when_missing(self) -> None:
        with patch.dict("sys.modules", {"dns_aid": None, "dns_aid.backends": None}):
            with pytest.raises(ImportError, match="dns-aid"):
                _import_create_backend()


class TestDnsAidToolkit:
    """Tests for DnsAidToolkit."""

    def test_get_tools_returns_three_tools(self) -> None:
        from langchain_community.agent_toolkits.dns_aid import DnsAidToolkit

        toolkit = DnsAidToolkit()
        tools = toolkit.get_tools()
        assert len(tools) == 3
        tool_names = {t.name for t in tools}
        assert tool_names == {"dns_aid_publish", "dns_aid_discover", "dns_aid_unpublish"}

    def test_toolkit_passes_backend_name(self) -> None:
        from langchain_community.agent_toolkits.dns_aid import DnsAidToolkit

        toolkit = DnsAidToolkit(backend_name="mock")
        tools = toolkit.get_tools()
        publish_tool = next(t for t in tools if t.name == "dns_aid_publish")
        assert publish_tool.backend_name == "mock"

    def test_toolkit_passes_backend_instance(self) -> None:
        from langchain_community.agent_toolkits.dns_aid import DnsAidToolkit

        mock_backend = MagicMock()
        toolkit = DnsAidToolkit(backend=mock_backend)
        tools = toolkit.get_tools()
        publish_tool = next(t for t in tools if t.name == "dns_aid_publish")
        assert publish_tool.backend is mock_backend
