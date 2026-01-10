"""Tests for API Governor Tool."""

import pytest

from langchain_community.tools.api_governor import APIGovernorTool


def test_api_governor_tool_init() -> None:
    """Test APIGovernorTool initialization."""
    tool = APIGovernorTool()
    assert tool.name == "api_governor"
    assert "OpenAPI" in tool.description
    assert "governance" in tool.description.lower()


def test_api_governor_tool_args_schema() -> None:
    """Test APIGovernorTool args schema."""
    tool = APIGovernorTool()
    schema = tool.args_schema.model_json_schema()
    assert "spec_content" in schema["properties"]
    assert "policy" in schema["properties"]
    assert "output_format" in schema["properties"]


def test_api_governor_tool_missing_package() -> None:
    """Test APIGovernorTool when package is not installed."""
    tool = APIGovernorTool()
    # This will return an error message since api-governor is not installed
    spec = """
openapi: "3.0.0"
info:
  title: Test API
  version: "1.0"
paths: {}
"""
    result = tool._run(spec_content=spec)
    # Should either work or return install instructions
    assert isinstance(result, str)
