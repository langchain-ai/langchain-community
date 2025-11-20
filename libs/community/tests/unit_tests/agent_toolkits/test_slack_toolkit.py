import sys
from unittest.mock import MagicMock

import pytest

# Mock slack_sdk module and WebClient class.
mock_slack_sdk = MagicMock()
mock_webclient_class = MagicMock()
mock_slack_sdk.WebClient = mock_webclient_class
sys.modules["slack_sdk"] = mock_slack_sdk

# We need to import the toolkit after patching sys.modules.
# ruff: noqa: E402
from langchain_community.agent_toolkits.slack.toolkit import SlackToolkit
from langchain_community.tools.slack.base import SlackBaseTool


@pytest.fixture
def mock_webclient() -> MagicMock:
    mock_client = MagicMock()
    mock_client.api_test.return_value = {"ok": True}
    return mock_client


def test_slack_toolkit_default_instantiation() -> None:
    """Test SlackToolkit can be instantiated with default settings."""
    toolkit = SlackToolkit()
    assert toolkit is not None

    tools = toolkit.get_tools()

    assert len(tools) == 4
    for tool in tools:
        assert isinstance(tool, SlackBaseTool)
        assert tool.client == toolkit.client


def test_slack_toolkit_get_tools_provided_client(mock_webclient: MagicMock) -> None:
    """Test that get_tools() creates tools with the provided client."""
    toolkit = SlackToolkit(client=mock_webclient)

    tools = toolkit.get_tools()

    assert len(tools) == 4
    for tool in tools:
        assert isinstance(tool, SlackBaseTool)
        assert tool.client == mock_webclient


def test_slack_toolkit_provided_client_reuse(mock_webclient: MagicMock) -> None:
    """Test that provided client instance is reused across multiple get_tools()
    calls."""
    toolkit = SlackToolkit(client=mock_webclient)
    tools1 = toolkit.get_tools()
    tools2 = toolkit.get_tools()

    for i in range(len(tools1)):
        tool1 = tools1[i]
        tool2 = tools2[i]
        assert isinstance(tool1, SlackBaseTool)
        assert isinstance(tool2, SlackBaseTool)
        assert tool1.client == tool2.client == mock_webclient
