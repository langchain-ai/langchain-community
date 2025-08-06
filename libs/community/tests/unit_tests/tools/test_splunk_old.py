"""Test Splunk standalone tools."""

import json
from unittest.mock import Mock
import pytest

from langchain_community.tools.splunk import (
    InfoSplunkTool,
    ListSplunkHostsTool,
    ListSplunkIndexesTool,
    ListSplunkSourcetypesTool,
    QueryCheckerSplunkTool,
    QuerySplunkTool,
)
from langchain_community.utilities.splunk import SplunkAPIWrapper


class TestStandaloneSplunkTools:
    """Test standalone Splunk tools functionality."""

    @pytest.fixture
    def mock_splunk_wrapper(self):
        """Create mock Splunk wrapper."""
        wrapper = Mock(spec=SplunkAPIWrapper)

        wrapper.splunk_token = "fake-token"
        wrapper.splunk_url = "https://mock-splunk.local"

        wrapper.get_summary_info.return_value = {
        "indexes": ["main", "security", "web"],
        "total_indexes": 3,
        "sample_sourcetypes": ["syslog", "json", "access_combined"],
        "sample_hosts": ["host1", "host2", "host3"],
        "connection_status": "connected"
        }
        wrapper.get_indexes.return_value = ["main", "security", "web_logs"]
        wrapper.get_sourcetypes.return_value = ["syslog", "json", "access_combined"]
        wrapper.get_hosts.return_value = ["server1", "server2", "server3"]
        wrapper.run_spl_query.return_value = [
        {"_time": "2023-01-01T00:00:00", "message": "test event"}
        ]
        wrapper.validate_spl_query.return_value = {"valid": True, "query": "test"}

        return wrapper

    def test_info_splunk_tool(self, mock_splunk_wrapper):
        """Test InfoSplunkTool as standalone tool."""
        tool = InfoSplunkTool(splunk_wrapper=mock_splunk_wrapper)
        
        # Test tool attributes
        assert tool.name == "splunk_info"
        assert "environment information" in tool.description.lower()
        
        # Test execution
        result = tool._run("")
        data = json.loads(result)
        assert data["connection_status"] == "connected"
        assert data["total_indexes"] == 3

    def test_list_indexes_tool(self, mock_splunk_wrapper):
        """Test ListSplunkIndexesTool as standalone tool."""
        tool = ListSplunkIndexesTool(splunk_wrapper=mock_splunk_wrapper)
        
        assert tool.name == "splunk_list_indexes"
        
        result = tool._run("")
        data = json.loads(result)
        assert "indexes" in data
        assert len(data["indexes"]) == 3
        assert "main" in data["indexes"]

    def test_list_sourcetypes_tool(self, mock_splunk_wrapper):
        """Test ListSplunkSourcetypesTool as standalone tool."""
        tool = ListSplunkSourcetypesTool(splunk_wrapper=mock_splunk_wrapper)
        
        assert tool.name == "splunk_list_sourcetypes"
        
        # Test without index specification
        result = tool._run("")
        data = json.loads(result)
        assert data["index"] == "all"
        assert len(data["sourcetypes"]) == 3
        
        # Test with index specification
        result = tool._run("main")
        data = json.loads(result)
        assert data["index"] == "main"
        mock_splunk_wrapper.get_sourcetypes.assert_called_with(index="main")

    def test_list_hosts_tool(self, mock_splunk_wrapper):
        """Test ListSplunkHostsTool as standalone tool."""
        tool = ListSplunkHostsTool(splunk_wrapper=mock_splunk_wrapper)
        
        assert tool.name == "splunk_list_hosts"
        
        result = tool._run("")
        data = json.loads(result)
        assert data["index"] == "all"
        assert len(data["hosts"]) == 3

    def test_query_tool(self, mock_splunk_wrapper):
        """Test QuerySplunkTool as standalone tool."""
        tool = QuerySplunkTool(splunk_wrapper=mock_splunk_wrapper)
        
        assert tool.name == "splunk_query"
        
        # Test successful query
        result = tool._run("search index=main error")
        data = json.loads(result)
        assert "query" in data
        assert "results_count" in data
        assert data["results_count"] == 1

    def test_query_checker_tool(self, mock_splunk_wrapper):
        """Test QueryCheckerSplunkTool as standalone tool."""
        tool = QueryCheckerSplunkTool(splunk_wrapper=mock_splunk_wrapper)
        
        assert tool.name == "splunk_query_checker"
        
        result = tool._run("search index=main error")
        data = json.loads(result)
        assert "query" in data
        assert "validation" in data
        assert "suggestions" in data

    def test_query_checker_with_llm(self, mock_splunk_wrapper):
        """Test QueryCheckerSplunkTool with LLM."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "Query looks good"
        mock_llm.invoke.return_value = mock_response
        
        tool = QueryCheckerSplunkTool(
            splunk_wrapper=mock_splunk_wrapper, 
            llm=mock_llm
        )
        
        result = tool._run("search index=main")
        data = json.loads(result)
        assert "llm_analysis" in data
        assert data["llm_analysis"] == "Query looks good"

    def test_tools_with_errors(self, mock_splunk_wrapper):
        """Test tools handle errors gracefully."""
        # Mock error condition
        mock_splunk_wrapper.get_summary_info.side_effect = Exception("Connection failed")
        
        tool = InfoSplunkTool(splunk_wrapper=mock_splunk_wrapper)
        result = tool._run("")
        
        assert "Error getting Splunk info" in result
        assert "Connection failed" in result

    def test_query_tool_empty_input(self, mock_splunk_wrapper):
        """Test QuerySplunkTool with empty input."""
        tool = QuerySplunkTool(splunk_wrapper=mock_splunk_wrapper)
        result = tool._run("")
        
        assert "Error: Empty query provided" in result

    def test_query_tool_auto_search_prefix(self, mock_splunk_wrapper):
        """Test QuerySplunkTool automatically adds search prefix."""
        tool = QuerySplunkTool(splunk_wrapper=mock_splunk_wrapper)
        tool._run("index=main error")
        
        # Verify search prefix was added
        call_args = mock_splunk_wrapper.run_spl_query.call_args[0]
        assert call_args[0].startswith("search ")

    def test_query_tool_preserves_pipe_commands(self, mock_splunk_wrapper):
        """Test QuerySplunkTool preserves pipe commands."""
        tool = QuerySplunkTool(splunk_wrapper=mock_splunk_wrapper)
        query = "| stats count by sourcetype"
        tool._run(query)
        
        # Verify pipe command was preserved
        call_args = mock_splunk_wrapper.run_spl_query.call_args[0]
        assert call_args[0] == query

    def test_tools_are_importable(self):
        """Test that all tools can be imported independently."""
        # This test verifies the tools can be imported and used standalone
        from langchain_community.tools.splunk import (
            InfoSplunkTool,
            ListSplunkHostsTool,
            ListSplunkIndexesTool,
            ListSplunkSourcetypesTool,
            QueryCheckerSplunkTool,
            QuerySplunkTool,
        )
        
        # Verify they are BaseTool instances
        from langchain_core.tools import BaseTool
        
        mock_wrapper = Mock(spec=SplunkAPIWrapper)
        
        tools = [
            InfoSplunkTool(splunk_wrapper=mock_wrapper),
            ListSplunkIndexesTool(splunk_wrapper=mock_wrapper),
            ListSplunkSourcetypesTool(splunk_wrapper=mock_wrapper),
            ListSplunkHostsTool(splunk_wrapper=mock_wrapper),
            QuerySplunkTool(splunk_wrapper=mock_wrapper),
            QueryCheckerSplunkTool(splunk_wrapper=mock_wrapper),
        ]
        
        for tool in tools:
            assert isinstance(tool, BaseTool)
            assert hasattr(tool, 'name')
            assert hasattr(tool, 'description')
            assert hasattr(tool, '_run')

    def test_tool_input_schemas(self, mock_splunk_wrapper):
        """Test that tools have proper input schemas."""
        tools = [
            InfoSplunkTool(splunk_wrapper=mock_splunk_wrapper),
            ListSplunkIndexesTool(splunk_wrapper=mock_splunk_wrapper),
            QuerySplunkTool(splunk_wrapper=mock_splunk_wrapper),
            QueryCheckerSplunkTool(splunk_wrapper=mock_splunk_wrapper),
        ]
        
        for tool in tools:
            assert hasattr(tool, 'args_schema')
            assert tool.args_schema is not None
