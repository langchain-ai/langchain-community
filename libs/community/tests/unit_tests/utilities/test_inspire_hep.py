"""Tests for INSPIRE HEP wrapper."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pytest
from unittest.mock import patch, Mock
from langchain_community.utilities.inspire_hep import INSPIREHEPAPIWrapper



def test_wrapper_can_be_created():
    """Test that we can create a wrapper instance."""
    wrapper = INSPIREHEPAPIWrapper()
    
    assert wrapper is not None
    assert wrapper.base_url == "https://inspirehep.net/api"
    assert wrapper.top_k_results == 10


def test_wrapper_with_custom_settings():
    """Test creating wrapper with custom top_k_results."""
    wrapper = INSPIREHEPAPIWrapper(top_k_results=5)
    
    assert wrapper.top_k_results == 5


def test_invalid_top_k_too_small():
    """Test that top_k_results=0 raises an error."""
    with pytest.raises(Exception):  # Pydantic will raise validation error
        INSPIREHEPAPIWrapper(top_k_results=0)


def test_invalid_top_k_too_large():
    """Test that top_k_results > 1000 raises an error."""
    with pytest.raises(Exception):
        INSPIREHEPAPIWrapper(top_k_results=2000)


def test_wrapper_has_required_methods():
    """Test that wrapper has all the methods we expect."""
    wrapper = INSPIREHEPAPIWrapper()
    
    # Check the methods exist
    assert hasattr(wrapper, 'search_literature')
    assert hasattr(wrapper, 'get_author_papers')
    assert hasattr(wrapper, 'get_paper_details')
    
    # Check they're callable
    assert callable(wrapper.search_literature)
    assert callable(wrapper.get_author_papers)
    assert callable(wrapper.get_paper_details)


@patch('langchain_community.utilities.inspire_hep.requests.get')
def test_search_literature_success(mock_get):
    """Test search_literature with mocked API response."""
    # Create a fake API response
    fake_response = {
        'hits': {
            'hits': [
                {
                    'metadata': {
                        'titles': [{'title': 'Test Paper 1'}],
                        'citation_count': 100
                    }
                },
                {
                    'metadata': {
                        'titles': [{'title': 'Test Paper 2'}],
                        'citation_count': 50
                    }
                }
            ]
        }
    }
    
    # Tell the mock what to return
    mock_get.return_value.json.return_value = fake_response
    mock_get.return_value.status_code = 200
    mock_get.return_value.raise_for_status = Mock()  # Don't raise errors
    
    # Now test your wrapper
    wrapper = INSPIREHEPAPIWrapper()
    result = wrapper.search_literature("quantum gravity")
    
    # Check the result
    assert "Test Paper 1" in result
    assert "Test Paper 2" in result
    assert "100" in result
    assert "50" in result
    
    # Verify the API was called correctly
    mock_get.assert_called_once()
    call_args = mock_get.call_args
    assert "literature" in call_args[0][0]  # Check URL contains 'literature'


@patch('langchain_community.utilities.inspire_hep.requests.get')
def test_search_literature_no_results(mock_get):
    """Test search when no papers are found."""
    # Fake response with no results
    fake_response = {'hits': {'hits': []}}
    
    mock_get.return_value.json.return_value = fake_response
    mock_get.return_value.status_code = 200
    mock_get.return_value.raise_for_status = Mock()
    
    wrapper = INSPIREHEPAPIWrapper()
    result = wrapper.search_literature("nonexistent topic")
    
    assert result == "No results found."

@patch('langchain_community.utilities.inspire_hep.requests.get')
def test_get_author_papers_success(mock_get):
    """Test get_author_papers with mocked API response."""
    fake_response = {
        'hits': {
            'hits': [
                {
                    'metadata': {
                        'titles': [{'title': 'Famous Paper'}],
                        'citation_count': 1000
                    }
                }
            ]
        }
    }
    
    mock_get.return_value.json.return_value = fake_response
    mock_get.return_value.status_code = 200
    mock_get.return_value.raise_for_status = Mock()
    
    wrapper = INSPIREHEPAPIWrapper()
    result = wrapper.get_author_papers("E.Witten.1")
    
    assert "E.Witten.1" in result
    assert "Famous Paper" in result
    assert "1000 citations" in result


@patch('langchain_community.utilities.inspire_hep.requests.get')
def test_get_author_papers_not_found(mock_get):
    """Test get_author_papers when author not found."""
    fake_response = {'hits': {'hits': []}}
    
    mock_get.return_value.json.return_value = fake_response
    mock_get.return_value.status_code = 200
    mock_get.return_value.raise_for_status = Mock()
    
    wrapper = INSPIREHEPAPIWrapper()
    result = wrapper.get_author_papers("Unknown.Author.1")
    
    assert "No papers found" in result


@patch('langchain_community.utilities.inspire_hep.requests.get')
def test_get_paper_details_success(mock_get):
    """Test get_paper_details with mocked API response."""
    fake_response = {
        'metadata': {
            'titles': [{'title': 'Specific Paper Title'}],
            'citation_count': 500,
            'authors': [
                {'full_name': 'Author One'},
                {'full_name': 'Author Two'}
            ],
            'abstracts': [{'value': 'This is the abstract of the paper.'}]
        }
    }
    
    mock_get.return_value.json.return_value = fake_response
    mock_get.return_value.status_code = 200
    mock_get.return_value.raise_for_status = Mock()
    
    wrapper = INSPIREHEPAPIWrapper()
    result = wrapper.get_paper_details("451647")
    
    assert "Specific Paper Title" in result
    assert "Author One" in result
    assert "500" in result
    assert "This is the abstract" in result


@patch('langchain_community.utilities.inspire_hep.requests.get')
def test_error_handling_404(mock_get):
    """Test that 404 errors are handled gracefully."""
    import requests
    
    # Simulate a 404 error
    mock_response = Mock()
    mock_response.status_code = 404
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("404 Not Found")
    mock_get.return_value = mock_response
    
    wrapper = INSPIREHEPAPIWrapper()
    result = wrapper.get_paper_details("999999999")
    
    assert "Error" in result    
@patch('langchain_community.utilities.inspire_hep.requests.get')
def test_search_with_sort_mostcited(mock_get, mock_search_response):
    """Test that sort=mostcited is passed correctly."""
    mock_get.return_value.json.return_value = mock_search_response
    mock_get.return_value.status_code = 200
    mock_get.return_value.raise_for_status = Mock()
    
    wrapper = INSPIREHEPAPIWrapper()
    result = wrapper.search_literature("quantum gravity", sort="mostcited")
    
    # Verify the sort parameter was passed
    call_args = mock_get.call_args
    assert call_args[1]['params']['sort'] == 'mostcited'


@patch('langchain_community.utilities.inspire_hep.requests.get')
def test_search_with_sort_mostrecent(mock_get, mock_search_response):
    """Test that sort=mostrecent is passed correctly."""
    mock_get.return_value.json.return_value = mock_search_response
    mock_get.return_value.status_code = 200
    mock_get.return_value.raise_for_status = Mock()
    
    wrapper = INSPIREHEPAPIWrapper()
    result = wrapper.search_literature("string theory", sort="mostrecent")
    
    # Verify the sort parameter was passed
    call_args = mock_get.call_args
    assert call_args[1]['params']['sort'] == 'mostrecent'


@patch('langchain_community.utilities.inspire_hep.requests.get')
def test_author_papers_with_sort_mostcited(mock_get, mock_author_response):
    """Test that sort parameter works for author papers (mostcited)."""
    mock_get.return_value.json.return_value = mock_author_response
    mock_get.return_value.status_code = 200
    mock_get.return_value.raise_for_status = Mock()
    
    wrapper = INSPIREHEPAPIWrapper()
    result = wrapper.get_author_papers("E.Witten.1", sort="mostcited")
    
    # Verify the sort parameter was passed
    call_args = mock_get.call_args
    assert call_args[1]['params']['sort'] == 'mostcited'


@patch('langchain_community.utilities.inspire_hep.requests.get')
def test_search_with_sort_mostcited(mock_get):
    """Test that sort=mostcited is passed correctly."""
    mock_response = {
        'hits': {
            'hits': [
                {'metadata': {'titles': [{'title': 'Test'}], 'citation_count': 100}}
            ]
        }
    }
    mock_get.return_value.json.return_value = mock_response
    mock_get.return_value.status_code = 200
    mock_get.return_value.raise_for_status = Mock()
    
    wrapper = INSPIREHEPAPIWrapper()
    result = wrapper.search_literature("quantum gravity", sort="mostcited")
    
    # Verify the sort parameter was passed
    call_args = mock_get.call_args
    assert call_args[1]['params']['sort'] == 'mostcited'
    assert "Test" in result


@patch('langchain_community.utilities.inspire_hep.requests.get')
def test_search_with_sort_mostrecent(mock_get):
    """Test that sort=mostrecent is passed correctly."""
    mock_response = {
        'hits': {
            'hits': [
                {'metadata': {'titles': [{'title': 'Recent Paper'}], 'citation_count': 10}}
            ]
        }
    }
    mock_get.return_value.json.return_value = mock_response
    mock_get.return_value.status_code = 200
    mock_get.return_value.raise_for_status = Mock()
    
    wrapper = INSPIREHEPAPIWrapper()
    result = wrapper.search_literature("string theory", sort="mostrecent")
    
    # Verify the sort parameter was passed
    call_args = mock_get.call_args
    assert call_args[1]['params']['sort'] == 'mostrecent'


@patch('langchain_community.utilities.inspire_hep.requests.get')
def test_author_papers_with_sort_mostcited(mock_get):
    """Test that sort parameter works for author papers (mostcited)."""
    mock_response = {
        'hits': {
            'hits': [
                {'metadata': {'titles': [{'title': 'Top Paper'}], 'citation_count': 500}}
            ]
        }
    }
    mock_get.return_value.json.return_value = mock_response
    mock_get.return_value.status_code = 200
    mock_get.return_value.raise_for_status = Mock()
    
    wrapper = INSPIREHEPAPIWrapper()
    result = wrapper.get_author_papers("E.Witten.1", sort="mostcited")
    
    # Verify the sort parameter was passed
    call_args = mock_get.call_args
    assert call_args[1]['params']['sort'] == 'mostcited'


@patch('langchain_community.utilities.inspire_hep.requests.get')
def test_author_papers_with_sort_mostrecent(mock_get):
    """Test that sort parameter works for author papers (mostrecent)."""
    mock_response = {
        'hits': {
            'hits': [
                {'metadata': {'titles': [{'title': 'New Paper'}], 'citation_count': 5}}
            ]
        }
    }
    mock_get.return_value.json.return_value = mock_response
    mock_get.return_value.status_code = 200
    mock_get.return_value.raise_for_status = Mock()
    
    wrapper = INSPIREHEPAPIWrapper()
    result = wrapper.get_author_papers("E.Witten.1", sort="mostrecent")
    
    # Verify the sort parameter was passed
    call_args = mock_get.call_args
    assert call_args[1]['params']['sort'] == 'mostrecent'
