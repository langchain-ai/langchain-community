"""Unit tests for BoxliteCodeInterpreterTool."""

from langchain_community.tools.boxlite import BoxliteCodeInterpreterTool


def test_code_interpreter_initialization() -> None:
    """Test BoxliteCodeInterpreterTool can be instantiated with default values."""
    tool = BoxliteCodeInterpreterTool()
    assert tool.name == "code_interpreter"
    assert "python" in tool.description.lower()
    assert "sandbox" in tool.description.lower()
    assert tool.image == "python:slim"
    assert tool.memory_mib == 2048
    assert tool.cpus == 2


def test_code_interpreter_custom_config() -> None:
    """Test BoxliteCodeInterpreterTool with custom configuration."""
    tool = BoxliteCodeInterpreterTool(
        image="python:3.11",
        memory_mib=4096,
        cpus=4,
    )
    assert tool.image == "python:3.11"
    assert tool.memory_mib == 4096
    assert tool.cpus == 4


def test_code_interpreter_schema() -> None:
    """Test BoxliteCodeInterpreterTool has correct input schema."""
    tool = BoxliteCodeInterpreterTool()
    schema = tool.args_schema.model_json_schema()
    assert "code" in schema["properties"]
    assert "packages" in schema["properties"]
    assert schema["required"] == ["code"]
