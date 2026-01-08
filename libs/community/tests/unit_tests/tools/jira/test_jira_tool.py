from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from langchain_community.tools.jira.tool import JiraAction


def test_jira_tool_invocation_validates_input() -> None:
    """Test that JiraAction invokes api_wrapper with mode and instructions."""
    api_wrapper = MagicMock()
    api_wrapper.run.return_value = "ok"
    # Use model_construct to bypass JiraAPIWrapper validation
    tool = JiraAction.model_construct(
        name="jql_query",
        description="JQL search",
        mode="jql",
        api_wrapper=api_wrapper,
    )

    result = tool.invoke({"instructions": "assignee = currentUser()"})

    assert result == "ok"
    api_wrapper.run.assert_called_once_with("jql", "assignee = currentUser()")


def test_jira_tool_invocation_raises_validation_error() -> None:
    """Test that invoking JiraAction with wrong input key raises ValidationError."""
    api_wrapper = MagicMock()
    # Use model_construct to bypass JiraAPIWrapper validation
    tool = JiraAction.model_construct(
        name="jql_query",
        description="JQL search",
        mode="jql",
        api_wrapper=api_wrapper,
    )

    with pytest.raises(ValidationError):
        tool.invoke({"query": "assignee = currentUser()"})
