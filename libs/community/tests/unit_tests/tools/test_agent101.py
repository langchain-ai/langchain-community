"""Tests for Agent101 tool."""

from unittest.mock import MagicMock, patch

from langchain_community.tools.agent101.tool import Agent101SearchRun


def test_agent101_search_run_init() -> None:
    """Test Agent101SearchRun initialization."""
    tool = Agent101SearchRun()
    assert tool.name == "agent101_search"
    assert "Agent101" in tool.description
    assert tool.base_url == "https://agent101.ventify.ai"
    assert tool.max_results == 5


@patch("langchain_community.tools.agent101.tool.requests.get")
def test_agent101_search_run(mock_get: MagicMock) -> None:
    """Test Agent101SearchRun returns results."""
    mock_response = MagicMock()
    mock_response.json.return_value = [
        {"name": "TestTool", "category": "code", "description": "A test tool"}
    ]
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    tool = Agent101SearchRun()
    result = tool._run("code generation")

    assert "TestTool" in result
    mock_get.assert_called_once()


@patch("langchain_community.tools.agent101.tool.requests.get")
def test_agent101_search_run_error(mock_get: MagicMock) -> None:
    """Test Agent101SearchRun handles errors gracefully."""
    mock_get.side_effect = Exception("Connection error")

    tool = Agent101SearchRun()
    result = tool._run("test query")

    assert "Error" in result
