import json
import pytest
from unittest.mock import Mock
from langchain_community.utilities.splunk import SplunkAPIWrapper
from langchain_community.tools.splunk import (
    InfoSplunkTool,
    ListSplunkHostsTool,
    ListSplunkIndexesTool,
    ListSplunkSourcetypesTool,
    QueryCheckerSplunkTool,
    QuerySplunkTool,
)

@pytest.fixture
def mock_splunk_wrapper():
    mock = Mock(spec=SplunkAPIWrapper)
    mock.splunk_username = "admin"
    mock.splunk_password = "changeme"
    mock.splunk_token = None

    mock.get_summary_info.return_value = {"info": "ok"}
    mock.get_indexes.return_value = ["main", "weblogs"]
    mock.get_sourcetypes.return_value = ["access_combined", "syslog"]
    mock.get_hosts.return_value = ["host1", "host2"]

    # Fix here:
    mock.configure_mock(run=Mock(return_value=[{"_raw": "search results"}]))

    return mock


def test_info_splunk_tool(mock_splunk_wrapper):
    tool = InfoSplunkTool(splunk_wrapper=mock_splunk_wrapper)
    assert json.loads(tool.run("")) == {"info": "ok"}

def test_list_indexes_tool(mock_splunk_wrapper):
    tool = ListSplunkIndexesTool(splunk_wrapper=mock_splunk_wrapper)
    result = json.loads(tool.run(""))
    assert result["indexes"] == ["main", "weblogs"]

def test_list_sourcetypes_tool(mock_splunk_wrapper):
    tool = ListSplunkSourcetypesTool(splunk_wrapper=mock_splunk_wrapper)
    result = json.loads(tool.run(""))
    assert result["sourcetypes"] == ["access_combined", "syslog"]

def test_list_hosts_tool(mock_splunk_wrapper):
    tool = ListSplunkHostsTool(splunk_wrapper=mock_splunk_wrapper)
    result = json.loads(tool.run(""))
    assert result["hosts"] == ["host1", "host2"]

def test_query_tool(mock_splunk_wrapper):
    tool = QuerySplunkTool(splunk_wrapper=mock_splunk_wrapper)
    result = tool.run("index=weblogs")
    assert "search results" in result

def test_query_checker_tool(mock_splunk_wrapper):
    tool = QueryCheckerSplunkTool(splunk_wrapper=mock_splunk_wrapper)
    result = json.loads(tool.run("index=weblogs"))
    assert "validation" in result

def test_query_checker_with_llm(mock_splunk_wrapper):
    from langchain_core.language_models.fake import FakeListLLM
    mock_llm = FakeListLLM(responses=["Query is valid"])
    tool = QueryCheckerSplunkTool(splunk_wrapper=mock_splunk_wrapper, llm=mock_llm)
    result = json.loads(tool.run("index=weblogs"))
    assert result["validation"] == "Query is valid"

