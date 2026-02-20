"""Integration tests for INSPIRE HEP (requires internet connection)."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pytest
from langchain_community.utilities.inspire_hep import INSPIREHEPAPIWrapper
from langchain_community.tools.inspire_hep import (
    INSPIRESearchLiteratureTool,
    INSPIREGetPaperDetailsTool,
)


class TestINSPIREHEPIntegration:
    """Integration tests with real API calls."""
    
    def test_search_literature_real(self):
        """Test real literature search."""
        wrapper = INSPIREHEPAPIWrapper(top_k_results=3)
        result = wrapper.search_literature("quantum gravity")
        
        # Should return some results
        assert result != "No results found."
        assert "Title:" in result
        assert "Citations:" in result
    
    def test_get_paper_details_real(self):
        """Test real paper details (Maldacena's famous paper)."""
        wrapper = INSPIREHEPAPIWrapper()
        result = wrapper.get_paper_details("451647")
        
        # Should contain expected info about the paper
        assert "Maldacena" in result
        assert "Title:" in result
        assert "Citations:" in result
        assert "Abstract:" in result
    
    def test_invalid_record_returns_error(self):
        """Test that invalid record ID returns error message."""
        wrapper = INSPIREHEPAPIWrapper()
        result = wrapper.get_paper_details("999999999")
        
        assert "Error" in result or "not found" in result.lower()
    
    def test_search_tool_works(self):
        """Test that the search tool works end-to-end."""
        tool = INSPIRESearchLiteratureTool()
        result = tool.invoke({"query": "string theory"})
        
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_paper_details_tool_works(self):
        """Test that the paper details tool works end-to-end."""
        tool = INSPIREGetPaperDetailsTool()
        result = tool.invoke({"record_id": "451647"})
        
        assert "Maldacena" in result
        assert isinstance(result, str)