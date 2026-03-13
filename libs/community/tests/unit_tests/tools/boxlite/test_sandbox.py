"""Unit tests for BoxliteSandboxTool."""

from langchain_community.tools.boxlite import BoxliteSandboxTool


def test_sandbox_tool_initialization() -> None:
    """Test BoxliteSandboxTool can be instantiated with default values."""
    tool = BoxliteSandboxTool()
    assert tool.name == "sandbox"
    assert "secure" in tool.description.lower()
    assert "isolated" in tool.description.lower()
    assert tool.image == "alpine:latest"
    assert tool.memory_mib == 2048
    assert tool.cpus == 1


def test_sandbox_tool_custom_config() -> None:
    """Test BoxliteSandboxTool with custom configuration."""
    tool = BoxliteSandboxTool(
        image="ubuntu:22.04",
        memory_mib=4096,
        cpus=4,
        working_dir="/workspace",
        env=[("FOO", "bar")],
    )
    assert tool.image == "ubuntu:22.04"
    assert tool.memory_mib == 4096
    assert tool.cpus == 4
    assert tool.working_dir == "/workspace"
    assert tool.env == [("FOO", "bar")]


def test_sandbox_tool_schema() -> None:
    """Test BoxliteSandboxTool has correct input schema."""
    tool = BoxliteSandboxTool()
    schema = tool.args_schema.model_json_schema()
    assert "command" in schema["properties"]
    assert "args" in schema["properties"]
    assert schema["required"] == ["command"]
