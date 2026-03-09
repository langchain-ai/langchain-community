"""Unit tests for AgentModuleTool."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import requests

from langchain_community.tools.agent_module.tool import AgentModuleTool


class TestAgentModuleTool:
    def test_tool_name(self) -> None:
        tool = AgentModuleTool()
        assert tool.name == "agent_module_eu_ai_act"

    def test_tool_description_contains_key_modules(self) -> None:
        tool = AgentModuleTool()
        assert "ETH_016" in tool.description
        assert "ETH_013" in tool.description

    def test_build_headers_without_key(self) -> None:
        tool = AgentModuleTool()
        assert tool._build_headers() == {}

    def test_build_headers_with_key(self) -> None:
        tool = AgentModuleTool(am_key="am_test_key_123")
        assert tool._build_headers() == {"X-AM-Key": "am_test_key_123"}

    def test_to_node_id(self) -> None:
        tool = AgentModuleTool()
        assert tool._to_node_id("ETH_013", "ethics") == "node:ethics:eth013"
        assert tool._to_node_id("ETH_021", "ethics") == "node:ethics:eth021"
        assert tool._to_node_id("ETH_016", "travel") == "node:travel:eth016"

    def test_run_without_key(self) -> None:
        tool = AgentModuleTool()
        mock_response = MagicMock()
        mock_response.text = '{"module": "ETH_013", "records": []}'
        mock_response.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_response) as mock_get:
            result = tool._run(module="ETH_013")
            mock_get.assert_called_once_with(
                "https://api.agent-module.dev/api/demo",
                params={"vertical": "ethics", "node": "node:ethics:eth013"},
                headers={},
                timeout=10,
            )
            assert result == '{"module": "ETH_013", "records": []}'

    def test_run_with_key(self) -> None:
        tool = AgentModuleTool(am_key="am_test_key_123")
        mock_response = MagicMock()
        mock_response.text = '{"module": "ETH_016"}'
        mock_response.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_response) as mock_get:
            result = tool._run(module="ETH_016")
            assert mock_get.call_args.kwargs["headers"] == {"X-AM-Key": "am_test_key_123"}
            assert result == '{"module": "ETH_016"}'

    def test_run_custom_vertical(self) -> None:
        tool = AgentModuleTool()
        mock_response = MagicMock()
        mock_response.text = "{}"
        mock_response.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_response) as mock_get:
            tool._run(module="MOD_001", vertical="travel")
            assert mock_get.call_args.kwargs["params"] == {
                "vertical": "travel",
                "node": "node:travel:mod001",
            }

    def test_run_handles_http_error(self) -> None:
        tool = AgentModuleTool()
        mock_response = MagicMock()
        mock_response.status_code = 401
        http_error = requests.HTTPError(response=mock_response)
        mock_response.raise_for_status.side_effect = http_error

        with patch("requests.get", return_value=mock_response):
            result = tool._run(module="ETH_013")
            assert "Agent Module HTTP error" in result

    def test_run_handles_connection_error(self) -> None:
        tool = AgentModuleTool()
        with patch("requests.get", side_effect=requests.RequestException("timeout")):
            result = tool._run(module="ETH_013")
            assert "Agent Module connection error" in result

    @pytest.mark.asyncio
    async def test_arun_returns_response(self) -> None:
        tool = AgentModuleTool()
        mock_response = MagicMock()
        mock_response.text = '{"module": "ETH_013"}'
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await tool._arun(module="ETH_013")
            assert result == '{"module": "ETH_013"}'
            mock_client.get.assert_called_once_with(
                "https://api.agent-module.dev/api/demo",
                params={"vertical": "ethics", "node": "node:ethics:eth013"},
                headers={},
                timeout=10,
            )
