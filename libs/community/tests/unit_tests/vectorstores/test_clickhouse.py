"""Test ClickHouse functionality."""

from unittest.mock import patch

import pytest

from langchain_community.vectorstores import Clickhouse, ClickhouseSettings
from tests.integration_tests.vectorstores.fake_embeddings import FakeEmbeddings


@pytest.mark.requires("clickhouse_connect")
def test_build_index_params_dict_format() -> None:
    """Test _build_index_params with dictionary format."""
    config = ClickhouseSettings()
    config.table = "test_table"
    
    with patch("clickhouse_connect.get_client"):
        clickhouse = Clickhouse(embedding=FakeEmbeddings(), config=config)
    
    # Test with non-empty dictionary
    config.index_param = {"metric": "L2Distance", "type": "hnsw", "m": 16}
    result = clickhouse._build_index_params(128)
    expected = "'metric=L2Distance','type=hnsw','m=16'"
    assert result == expected
    
    # Test with empty dictionary
    config.index_param = {}
    result = clickhouse._build_index_params(128)
    assert result == ""


@pytest.mark.requires("clickhouse_connect")
def test_build_index_params_list_format() -> None:
    """Test _build_index_params with list format."""
    config = ClickhouseSettings()
    config.table = "test_table"
    
    with patch("clickhouse_connect.get_client"):
        clickhouse = Clickhouse(embedding=FakeEmbeddings(), config=config)
    
    # Test with vector_similarity index type (should replace 3rd parameter)
    config.index_type = "vector_similarity"
    config.index_param = ["'hnsw'", "'L2Distance'", 64, "'additional_param'"]
    result = clickhouse._build_index_params(128)
    expected = "'hnsw','L2Distance',128,'additional_param'"
    assert result == expected
    
    # Test with non-vector_similarity index type (should not modify)
    config.index_type = "annoy"
    config.index_param = ["'angular'", 100, "'some_param'"]
    result = clickhouse._build_index_params(128)
    expected = "'angular',100,'some_param'"
    assert result == expected
    
    # Test with empty list
    config.index_param = []
    result = clickhouse._build_index_params(128)
    assert result == ""


@pytest.mark.requires("clickhouse_connect")
def test_build_index_params_string_format() -> None:
    """Test _build_index_params with string format."""
    config = ClickhouseSettings()
    config.table = "test_table"
    
    with patch("clickhouse_connect.get_client"):
        clickhouse = Clickhouse(embedding=FakeEmbeddings(), config=config)
    
    # Test with non-empty string
    config.index_param = "'hnsw','L2Distance',128"
    result = clickhouse._build_index_params(256)
    expected = "'hnsw','L2Distance',128"
    assert result == expected
    
    # Test with empty string
    config.index_param = ""
    result = clickhouse._build_index_params(128)
    assert result == ""
    
    # Test with None (should return empty string)
    config.index_param = None
    result = clickhouse._build_index_params(128)
    assert result == ""


@pytest.mark.requires("clickhouse_connect")
def test_build_index_params_default_behavior() -> None:
    """Test _build_index_params with default configuration."""
    config = ClickhouseSettings()
    config.table = "test_table"
    
    with patch("clickhouse_connect.get_client"):
        clickhouse = Clickhouse(embedding=FakeEmbeddings(), config=config)
    
    # Test with default settings (should use default list format)
    # Default: index_param = ["'hnsw'", "'L2Distance'", None]
    # Default: index_type = "vector_similarity"
    result = clickhouse._build_index_params(384)
    expected = "'hnsw','L2Distance',384"
    assert result == expected
    

@pytest.mark.requires("clickhouse_connect")
def test_build_index_params_vector_similarity_dimension_replacement() -> None:
    """Test that vector_similarity index correctly replaces dimension parameter."""
    config = ClickhouseSettings()
    config.table = "test_table"
    
    with patch("clickhouse_connect.get_client"):
        clickhouse = Clickhouse(embedding=FakeEmbeddings(), config=config)
    
    config.index_type = "vector_similarity"
    
    # Test with 4 parameters - should replace index 2 (3rd parameter)
    config.index_param = ["param1", "param2", "old_dim", "param4"]
    result = clickhouse._build_index_params(512)
    expected = "param1,param2,512,param4"
    assert result == expected
    
    # Test with 5 parameters - should replace index 2 (3rd parameter)
    config.index_param = ["a", "b", 999, "d", "e"]
    result = clickhouse._build_index_params(1024)
    expected = "a,b,1024,d,e"
    assert result == expected
    
    # Test that non-vector_similarity doesn't replace dimension
    config.index_type = "usearch"
    config.index_param = ["a", "b", 999, "d", "e"]
    result = clickhouse._build_index_params(1024)
    expected = "a,b,999,d,e"  # Should not replace the dimension
    assert result == expected
