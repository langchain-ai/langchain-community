"""Unit tests for BoxliteComputerUseTool."""

from langchain_community.tools.boxlite import BoxliteComputerUseTool


def test_computer_use_initialization() -> None:
    """Test BoxliteComputerUseTool can be instantiated with default values."""
    tool = BoxliteComputerUseTool()
    assert tool.name == "computer_use"
    assert "desktop" in tool.description.lower()
    assert "gui" in tool.description.lower()
    assert tool.memory == 4096
    assert tool.cpus == 4


def test_computer_use_custom_config() -> None:
    """Test BoxliteComputerUseTool with custom configuration."""
    tool = BoxliteComputerUseTool(
        memory=8192,
        cpus=8,
        monitor_https_port=4001,
    )
    assert tool.memory == 8192
    assert tool.cpus == 8
    assert tool.monitor_https_port == 4001


def test_computer_use_schema() -> None:
    """Test BoxliteComputerUseTool has correct input schema."""
    tool = BoxliteComputerUseTool()
    schema = tool.args_schema.model_json_schema()
    assert "action" in schema["properties"]
    assert "x" in schema["properties"]
    assert "y" in schema["properties"]
    assert "text" in schema["properties"]
    # Check action enum values
    action_enum = schema["properties"]["action"]["enum"]
    assert "screenshot" in action_enum
    assert "click" in action_enum
    assert "type" in action_enum
    assert "key" in action_enum
