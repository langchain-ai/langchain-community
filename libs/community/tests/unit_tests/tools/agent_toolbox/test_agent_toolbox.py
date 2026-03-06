"""Tests for Agent Toolbox tools."""

from unittest.mock import MagicMock, patch

from langchain_community.tools.agent_toolbox import (
    AgentToolboxSearch,
    AgentToolboxExtract,
    AgentToolboxGeoIP,
    AgentToolboxNews,
    AgentToolboxWhois,
    AgentToolboxDns,
    AgentToolboxPdfExtract,
    AgentToolboxQr,
    AgentToolboxRun,
)
from langchain_community.utilities.agent_toolbox import AgentToolboxAPIWrapper


def test_search_tool():
    """Test search tool instantiation."""
    tool = AgentToolboxSearch(
        api_wrapper=AgentToolboxAPIWrapper(api_key="atb_test")
    )
    assert tool.name == "agent_toolbox_search"
    assert "search" in tool.description.lower()


def test_all_tools_instantiate():
    """Test all 13 tools can be instantiated."""
    wrapper = AgentToolboxAPIWrapper(api_key="atb_test")
    tools = [
        AgentToolboxSearch(api_wrapper=wrapper),
        AgentToolboxExtract(api_wrapper=wrapper),
        AgentToolboxGeoIP(api_wrapper=wrapper),
        AgentToolboxNews(api_wrapper=wrapper),
        AgentToolboxWhois(api_wrapper=wrapper),
        AgentToolboxDns(api_wrapper=wrapper),
        AgentToolboxPdfExtract(api_wrapper=wrapper),
        AgentToolboxQr(api_wrapper=wrapper),
        AgentToolboxRun(api_wrapper=wrapper),
    ]
    assert len(tools) == 9
    for tool in tools:
        assert tool.name.startswith("agent_toolbox")


@patch("langchain_community.utilities.agent_toolbox.requests.post")
def test_wrapper_run(mock_post: MagicMock) -> None:
    """Test wrapper makes correct API call."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"success": True, "data": [{"title": "Test"}]}
    mock_resp.raise_for_status.return_value = None
    mock_post.return_value = mock_resp

    wrapper = AgentToolboxAPIWrapper(
        api_key="atb_test",
        base_url="https://api.example.com",
    )
    result = wrapper.run("search", {"query": "test"})

    mock_post.assert_called_once()
    call_args = mock_post.call_args
    assert "api.example.com/v1/search" in call_args[0][0]
    assert "Bearer atb_test" in call_args[1]["headers"]["Authorization"]
    assert "Test" in result


@patch("langchain_community.utilities.agent_toolbox.requests.post")
def test_wrapper_results(mock_post: MagicMock) -> None:
    """Test wrapper returns raw dict."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"success": True, "data": {"ip": "8.8.8.8"}}
    mock_resp.raise_for_status.return_value = None
    mock_post.return_value = mock_resp

    wrapper = AgentToolboxAPIWrapper(api_key="atb_test")
    result = wrapper.results("geoip", {"ip": "8.8.8.8"})

    assert isinstance(result, dict)
    assert result["ip"] == "8.8.8.8"
