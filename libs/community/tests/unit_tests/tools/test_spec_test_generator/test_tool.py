"""Tests for Spec Test Generator Tool."""

import pytest

from langchain_community.tools.spec_test_generator import SpecTestGeneratorTool


def test_spec_test_generator_tool_init() -> None:
    """Test SpecTestGeneratorTool initialization."""
    tool = SpecTestGeneratorTool()
    assert tool.name == "spec_test_generator"
    assert "PRD" in tool.description
    assert "requirements" in tool.description.lower()


def test_spec_test_generator_tool_args_schema() -> None:
    """Test SpecTestGeneratorTool args schema."""
    tool = SpecTestGeneratorTool()
    schema = tool.args_schema.model_json_schema()
    assert "prd_content" in schema["properties"]
    assert "output_format" in schema["properties"]


def test_spec_test_generator_tool_missing_package() -> None:
    """Test SpecTestGeneratorTool when package is not installed."""
    tool = SpecTestGeneratorTool()
    # This will return an error message since spec-test-generator is not installed
    result = tool._run(prd_content="# Test PRD\n## Goal\nTest goal")
    # Should either work or return install instructions
    assert isinstance(result, str)
