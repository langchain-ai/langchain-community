"""Integration tests for Splunk toolkit."""

import json
import os
import pytest

from langchain_community.agent_toolkits.splunk import (
    SplunkToolkit,
    create_splunk_agent_from_api_wrapper,
)
from langchain_community.tools.splunk import (
    InfoSplunkTool,
    ListSplunkIndexesTool,
    QuerySplunkTool,
)
from langchain_community.utilities.splunk import SplunkAPIWrapper


@pytest.mark.integration
class TestSplunkIntegration:
    """
    Integration tests that require a real Splunk instance.
    
    To run these tests, set the following environment variables:
    - SPLUNK_HOST: Your Splunk server hostname
    - SPLUNK_TOKEN: Your Splunk authentication token
    - SPLUNK_PORT: Splunk port (default: 8089)
    - SPLUNK_SCHEME: http or https (default: https)
    """

    @pytest.fixture
    def splunk_wrapper(self):
        """Create SplunkAPIWrapper from environment variables."""
        if not all([
            os.getenv("SPLUNK_HOST"),
            os.getenv("SPLUNK_TOKEN")
        ]):
            pytest.skip("Splunk credentials not provided")
        
        return SplunkAPIWrapper(
            splunk_host=os.getenv("SPLUNK_HOST"),
            splunk_token=os.getenv("SPLUNK_TOKEN"),
            splunk_port=int(os.getenv("SPLUNK_PORT", "8089")),
            splunk_scheme=os.getenv("SPLUNK_SCHEME", "https"),
            verify_ssl=False  # Often needed for test environments
        )

    def test_real_splunk_connection(self, splunk_wrapper):
        """Test connection to real Splunk instance."""
        assert splunk_wrapper.test_connection()

    def test_get_indexes(self, splunk_wrapper):
        """Test getting indexes from real Splunk."""
        indexes = splunk_wrapper.get_indexes()
        assert isinstance(indexes, list)
        assert len(indexes) > 0
        # Most Splunk instances should have 'main' index
        assert any('main' in idx for idx in indexes)

    def test_get_sourcetypes(self, splunk_wrapper):
        """Test getting sourcetypes from real Splunk."""
        sourcetypes = splunk_wrapper.get_sourcetypes(limit=10)
        assert isinstance(sourcetypes, list)
        # May be empty in test environments, so just check type

    def test_get_hosts(self, splunk_wrapper):
        """Test getting hosts from real Splunk."""
        hosts = splunk_wrapper.get_hosts(limit=10)
        assert isinstance(hosts, list)
        # May be empty in test environments, so just check type

    def test_real_query_execution(self, splunk_wrapper):
        """Test executing real SPL query."""
        results = splunk_wrapper.run_spl_query("search * | head 1")
        assert isinstance(results, list)
        # Results may be empty, but should be a list

    def test_query_validation(self, splunk_wrapper):
        """Test SPL query validation."""
        result = splunk_wrapper.validate_spl_query("search index=main")
        assert isinstance(result, dict)
        assert "valid" in result
        assert "query" in result

    def test_summary_info(self, splunk_wrapper):
        """Test getting summary information."""
        info = splunk_wrapper.get_summary_info()
        assert isinstance(info, dict)
        assert "connection_status" in info
        assert info["connection_status"] == "connected"
        assert "indexes" in info
        assert "total_indexes" in info

    def test_standalone_info_tool(self, splunk_wrapper):
        """Test InfoSplunkTool with real Splunk."""
        tool = InfoSplunkTool(splunk_wrapper=splunk_wrapper)
        result = tool.run("")
        
        data = json.loads(result)
        assert "indexes" in data
        assert "connection_status" in data
        assert data["connection_status"] == "connected"

    def test_standalone_list_indexes_tool(self, splunk_wrapper):
        """Test ListSplunkIndexesTool with real Splunk."""
        tool = ListSplunkIndexesTool(splunk_wrapper=splunk_wrapper)
        result = tool.run("")
        
        data = json.loads(result)
        assert "indexes" in data
        assert "count" in data
        assert isinstance(data["indexes"], list)

    def test_standalone_query_tool(self, splunk_wrapper):
        """Test QuerySplunkTool with real Splunk."""
        tool = QuerySplunkTool(splunk_wrapper=splunk_wrapper)
        result = tool.run("search * | head 1")
        
        # Should return either results or no results message
        assert isinstance(result, str)
        if "returned no results" not in result:
            data = json.loads(result)
            assert "query" in data
            assert "results_count" in data

    def test_toolkit_with_real_splunk(self, splunk_wrapper):
        """Test SplunkToolkit with real Splunk."""
        toolkit = SplunkToolkit(splunk_wrapper=splunk_wrapper)
        tools = toolkit.get_tools()
        
        assert len(tools) >= 3
        tool_names = [tool.name for tool in tools]
        assert "splunk_info" in tool_names
        assert "splunk_list_indexes" in tool_names
        assert "splunk_query" in tool_names

    def test_toolkit_tools_execution(self, splunk_wrapper):
        """Test executing toolkit tools with real Splunk."""
        toolkit = SplunkToolkit(splunk_wrapper=splunk_wrapper)
        tools = toolkit.get_tools()
        
        # Test info tool
        info_tool = next(tool for tool in tools if tool.name == "splunk_info")
        info_result = info_tool.run("")
        info_data = json.loads(info_result)
        assert info_data["connection_status"] == "connected"
        
        # Test list indexes tool  
        list_tool = next(tool for tool in tools if tool.name == "splunk_list_indexes")
        list_result = list_tool.run("")
        list_data = json.loads(list_result)
        assert "indexes" in list_data

    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"), 
        reason="OpenAI API key not provided"
    )
    def test_agent_with_real_splunk(self, splunk_wrapper):
        """Test creating and using agent with real Splunk."""
        from langchain.llms import OpenAI
        
        llm = OpenAI(temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))
        
        agent = create_splunk_agent_from_api_wrapper(
            llm=llm,
            splunk_wrapper=splunk_wrapper,
            verbose=True
        )
        
        # Test simple query
        result = agent.run("What indexes are available?")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_error_handling_with_bad_query(self, splunk_wrapper):
        """Test error handling with invalid query."""
        tool = QuerySplunkTool(splunk_wrapper=splunk_wrapper)
        result = tool.run("invalid SPL syntax here")
        
        # Should handle error gracefully
        assert "Error executing SPL query" in result

    def test_connection_with_bad_credentials(self):
        """Test connection failure with bad credentials."""
        bad_wrapper = SplunkAPIWrapper(
            splunk_host="nonexistent.splunk.com",
            splunk_token="invalid-token"
        )
        
        assert not bad_wrapper.test_connection()

    def test_query_with_time_range(self, splunk_wrapper):
        """Test query execution with time range."""
        results = splunk_wrapper.run_spl_query(
            "search * | head 1",
            earliest_time="-1h",
            latest_time="now"
        )
        assert isinstance(results, list)

    def test_index_specific_operations(self, splunk_wrapper):
        """Test operations on specific index if available."""
        indexes = splunk_wrapper.get_indexes()
        if indexes:
            first_index = indexes[0]
            
            # Test index-specific sourcetypes
            sourcetypes = splunk_wrapper.get_sourcetypes(index=first_index)
            assert isinstance(sourcetypes, list)
            
            # Test index-specific hosts
            hosts = splunk_wrapper.get_hosts(index=first_index)
            assert isinstance(hosts, list)


@pytest.mark.integration
class TestSplunkPerformance:
    """Performance tests for Splunk operations."""

    @pytest.fixture
    def splunk_wrapper(self):
        """Create SplunkAPIWrapper from environment variables."""
        if not all([
            os.getenv("SPLUNK_HOST"),
            os.getenv("SPLUNK_TOKEN")
        ]):
            pytest.skip("Splunk credentials not provided")
        
        return SplunkAPIWrapper(
            splunk_host=os.getenv("SPLUNK_HOST"),
            splunk_token=os.getenv("SPLUNK_TOKEN"),
            verify_ssl=False
        )

    def test_query_timeout(self, splunk_wrapper):
        """Test query timeout behavior."""
        import time
        
        # Set short timeout
        wrapper = SplunkAPIWrapper(
            splunk_host=splunk_wrapper.splunk_host,
            splunk_token=splunk_wrapper.splunk_token,
            timeout=5,  # 5 second timeout
            verify_ssl=False
        )
        
        # This query might timeout on large datasets
        start_time = time.time()
        try:
            results = wrapper.run_spl_query("search *")
            # If it completes, that's fine too
            assert isinstance(results, list)
        except Exception as e:
            # Timeout is expected behavior
            elapsed = time.time() - start_time
            assert elapsed < 30  # Should fail within reasonable time

    def test_large_result_limit(self, splunk_wrapper):
        """Test handling of large result sets."""
        # Test with larger result limit
        results = splunk_wrapper.run_spl_query(
            "search * | head 500",
            max_results=500
        )
        assert isinstance(results, list)
        assert len(results) <= 500

    def test_concurrent_queries(self, splunk_wrapper):
        """Test multiple concurrent queries."""
        import threading
        import time
        
        results = []
        errors = []
        
        def run_query(query_id):
            try:
                result = splunk_wrapper.run_spl_query(f"search * | head 1")
                results.append((query_id, result))
            except Exception as e:
                errors.append((query_id, str(e)))
        
        # Start multiple threads
        threads = []
        for i in range(3):
            t = threading.Thread(target=run_query, args=(i,))
            threads.append(t)
            t.start()
        
        # Wait for completion
        for t in threads:
            t.join(timeout=30)
        
        # Check results
        assert len(results) + len(errors) == 3
        # At least some should succeed
        assert len(results) > 0


if __name__ == "__main__":
    # Run integration tests if credentials are available
    if all([
        os.getenv("SPLUNK_HOST"),
        os.getenv("SPLUNK_TOKEN")
    ]):
        print("Running Splunk integration tests...")
        pytest.main([__file__, "-v", "-m", "integration"])
    else:
        print("Splunk credentials not provided. Set SPLUNK_HOST and SPLUNK_TOKEN to run integration tests.")
