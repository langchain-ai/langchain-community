"""Unit tests for BoxliteBrowserTool."""

from langchain_community.tools.boxlite import BoxliteBrowserTool


def test_browser_tool_initialization() -> None:
    """Test BoxliteBrowserTool can be instantiated with default values."""
    tool = BoxliteBrowserTool()
    assert tool.name == "browser"
    assert "browser" in tool.description.lower()
    assert "cdp" in tool.description.lower() or "endpoint" in tool.description.lower()
    assert tool.browser == "chromium"
    assert tool.memory == 2048
    assert tool.cpus == 2


def test_browser_tool_custom_config() -> None:
    """Test BoxliteBrowserTool with custom configuration."""
    tool = BoxliteBrowserTool(
        browser="firefox",
        memory=4096,
        cpus=4,
    )
    assert tool.browser == "firefox"
    assert tool.memory == 4096
    assert tool.cpus == 4


def test_browser_tool_schema() -> None:
    """Test BoxliteBrowserTool has correct input schema."""
    tool = BoxliteBrowserTool()
    schema = tool.args_schema.model_json_schema()
    assert "action" in schema["properties"]
    action_enum = schema["properties"]["action"]["enum"]
    assert "start" in action_enum
    assert "stop" in action_enum
    assert "endpoint" in action_enum
