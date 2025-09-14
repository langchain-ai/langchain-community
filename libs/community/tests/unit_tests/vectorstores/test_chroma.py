"""Unit tests for Chroma vectorstore."""
import os
import unittest.mock
import pytest
from unittest.mock import patch, MagicMock

from langchain_community.vectorstores import Chroma


class TestChromaCPUGuard:
    """Test CPU guard functionality in Chroma initialization."""
    
    @patch('langchain_community.vectorstores.chroma.chromadb')
    @patch('os.cpu_count')
    def test_cpu_guard_caps_excessive_threads(self, mock_cpu_count, mock_chromadb):
        """Test that hnsw:num_threads is capped to CPU count when excessive."""
        mock_cpu_count.return_value = 4
        mock_client = MagicMock()
        mock_chromadb.Client.return_value = mock_client
        mock_chromadb.config.Settings.return_value = MagicMock()
        
        collection_metadata = {"hnsw:num_threads": 8}  # More than CPU count
        
        Chroma(
            collection_name="test_cpu_guard",
            collection_metadata=collection_metadata
        )
        
        # Verify the metadata was modified
        assert collection_metadata["hnsw:num_threads"] == 4

    @patch('langchain_community.vectorstores.chroma.chromadb')
    @patch('os.cpu_count')
    def test_cpu_guard_preserves_valid_threads(self, mock_cpu_count, mock_chromadb):
        """Test that valid thread counts are preserved."""
        mock_cpu_count.return_value = 8
        mock_client = MagicMock()
        mock_chromadb.Client.return_value = mock_client
        mock_chromadb.config.Settings.return_value = MagicMock()
        
        collection_metadata = {"hnsw:num_threads": 4}  # Less than CPU count
        
        Chroma(
            collection_name="test_cpu_guard_valid",
            collection_metadata=collection_metadata
        )
        
        # Verify the metadata was NOT modified
        assert collection_metadata["hnsw:num_threads"] == 4

    @patch('langchain_community.vectorstores.chroma.chromadb')
    def test_cpu_guard_handles_none_metadata(self, mock_chromadb):
        """Test that None collection_metadata doesn't cause issues."""
        mock_client = MagicMock()
        mock_chromadb.Client.return_value = mock_client
        mock_chromadb.config.Settings.return_value = MagicMock()
        
        # Should not raise any exceptions
        Chroma(
            collection_name="test_none_metadata",
            collection_metadata=None
        )

    @patch('langchain_community.vectorstores.chroma.chromadb')
    def test_cpu_guard_handles_missing_threads_key(self, mock_chromadb):
        """Test metadata without hnsw:num_threads key."""
        mock_client = MagicMock()
        mock_chromadb.Client.return_value = mock_client
        mock_chromadb.config.Settings.return_value = MagicMock()
        
        collection_metadata = {"hnsw:space": "cosine"}  # No num_threads key
        
        Chroma(
            collection_name="test_missing_key",
            collection_metadata=collection_metadata
        )
        
        # Should remain unchanged
        assert "hnsw:num_threads" not in collection_metadata
        assert collection_metadata["hnsw:space"] == "cosine"