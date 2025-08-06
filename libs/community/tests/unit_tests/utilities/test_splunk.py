"""Test Splunk utility."""

import json
from unittest.mock import Mock, patch, MagicMock
import pytest
from langchain_community.utilities.splunk import SplunkAPIWrapper


class TestSplunkAPIWrapper:
    """Test SplunkAPIWrapper functionality."""

    @pytest.fixture
    def mock_splunk_wrapper(self):
        """Create a mock Splunk wrapper for testing."""
        with patch('requests.Session') as mock_session_class:
            mock_session = Mock()
            mock_session_class.return_value = mock_session
            
            wrapper = SplunkAPIWrapper(
                splunk_host="test.splunk.com",
                splunk_token="test-token-123",
                splunk_port=8089,
                verify_ssl=False
            )
            wrapper._session = mock_session
            return wrapper, mock_session

    def test_init_with_token(self):
        """Test initialization with token authentication."""
        with patch('requests.Session'):
            wrapper = SplunkAPIWrapper(
                splunk_host="test.com",
                splunk_token="test-token"
            )
            assert wrapper.splunk_host == "test.com"
            assert wrapper.splunk_token == "test-token"
            assert wrapper.splunk_port == 8089

    def test_init_with_username_password(self):
        """Test initialization with username/password authentication."""
        with patch('requests.Session'):
            wrapper = SplunkAPIWrapper(
                splunk_host="test.com",
                splunk_token="",  # Empty token
                splunk_username="admin",
                splunk_password="password"
            )
            assert wrapper.splunk_username == "admin"
            assert wrapper.splunk_password == "password"

    def test_init_validation_error(self):
        """Test validation error when no auth provided."""
        with pytest.raises(ValueError):
            SplunkAPIWrapper(
                splunk_host="test.com",
                splunk_token="",  # No token
                splunk_username=None,  # No username
                splunk_password=None   # No password
            )

    def test_base_url_property(self):
        """Test base_url property."""
        with patch('requests.Session'):
            wrapper = SplunkAPIWrapper(
                splunk_host="test.com",
                splunk_port=8089,
                splunk_token="test-token",
                splunk_scheme="https"
            )
            assert wrapper.base_url == "https://test.com:8089"

    def test_test_connection_success(self, mock_splunk_wrapper):
        """Test successful connection test."""
        wrapper, mock_session = mock_splunk_wrapper
        
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_session.get.return_value = mock_response
        
        assert wrapper.test_connection() is True
        mock_session.get.assert_called_once_with("https://test.splunk.com:8089/services/server/info")

    def test_test_connection_failure(self, mock_splunk_wrapper):
        """Test connection test failure."""
        wrapper, mock_session = mock_splunk_wrapper
        
        mock_session.get.side_effect = Exception("Connection failed")
        
        assert wrapper.test_connection() is False

    def test_run_spl_query_success(self, mock_splunk_wrapper):
        """Test successful SPL query execution."""
        wrapper, mock_session = mock_splunk_wrapper
        
        # Mock job creation response
        job_response = Mock()
        job_response.raise_for_status.return_value = None
        job_response.json.return_value = {"sid": "test_job_id"}
        
        # Mock job status response (completed)
        status_response = Mock()
        status_response.raise_for_status.return_value = None
        status_response.json.return_value = {
            "entry": [{"content": {"dispatchState": "DONE"}}]
        }
        
        # Mock results response
        results_response = Mock()
        results_response.raise_for_status.return_value = None
        results_response.json.return_value = {
            "results": [
                {"_time": "2023-01-01T00:00:00", "message": "test event 1"},
                {"_time": "2023-01-01T00:01:00", "message": "test event 2"}
            ]
        }
        
        mock_session.post.return_value = job_response
        mock_session.get.side_effect = [status_response, results_response]
        
        results = wrapper.run_spl_query("search index=main")
        
        assert len(results) == 2
        assert results[0]["message"] == "test event 1"
        assert results[1]["message"] == "test event 2"

    def test_run_spl_query_no_results(self, mock_splunk_wrapper):
        """Test SPL query with no results."""
        wrapper, mock_session = mock_splunk_wrapper
        
        # Mock responses for no results
        job_response = Mock()
        job_response.raise_for_status.return_value = None
        job_response.json.return_value = {"sid": "test_job_id"}
        
        status_response = Mock()
        status_response.raise_for_status.return_value = None
        status_response.json.return_value = {
            "entry": [{"content": {"dispatchState": "DONE"}}]
        }
        
        results_response = Mock()
        results_response.raise_for_status.return_value = None
        results_response.json.return_value = {"results": []}
        
        mock_session.post.return_value = job_response
        mock_session.get.side_effect = [status_response, results_response]
        
        results = wrapper.run_spl_query("search index=nonexistent")
        assert results == []

    def test_get_indexes(self, mock_splunk_wrapper):
        """Test getting list of indexes."""
        wrapper, mock_session = mock_splunk_wrapper
        
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "entry": [
                {"name": "main"},
                {"name": "security"}, 
                {"name": "web_logs"}
            ]
        }
        mock_session.get.return_value = mock_response
        
        indexes = wrapper.get_indexes()
        
        assert len(indexes) == 3
        assert "main" in indexes
        assert "security" in indexes
        assert "web_logs" in indexes
        assert indexes == ["main", "security", "web_logs"]  # Should be sorted

    def test_get_sourcetypes(self, mock_splunk_wrapper):
        """Test getting sourcetypes."""
        wrapper, mock_session = mock_splunk_wrapper
        
        # Mock the SPL query execution for sourcetypes
        #with patch.object(wrapper, 'run_spl_query') as mock_query:
        with patch.object(SplunkAPIWrapper, 'run_spl_query') as mock_query:
            mock_query.return_value = [
                {"sourcetype": "access_combined"},
                {"sourcetype": "syslog"},
                {"sourcetype": "json"}
            ]
            
            sourcetypes = wrapper.get_sourcetypes()
            
            assert len(sourcetypes) == 3
            assert "access_combined" in sourcetypes
            assert "syslog" in sourcetypes
            assert "json" in sourcetypes

    def test_get_sourcetypes_with_index(self, mock_splunk_wrapper):
        """Test getting sourcetypes for specific index."""
        wrapper, mock_session = mock_splunk_wrapper
        
        #with patch.object(wrapper, 'run_spl_query') as mock_query:
        with patch.object(SplunkAPIWrapper, 'run_spl_query') as mock_query:
            mock_query.return_value = [{"sourcetype": "web_access"}]
            
            sourcetypes = wrapper.get_sourcetypes("web")
            
            assert len(sourcetypes) == 1
            assert sourcetypes[0] == "web_access"
            # Verify the query included the index
            mock_query.assert_called_once()
            call_args = mock_query.call_args[0]
            assert "index=web" in call_args[0]

    def test_validate_spl_query(self, mock_splunk_wrapper):
        """Test SPL query validation."""
        wrapper, mock_session = mock_splunk_wrapper
        
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"parsed": "successfully"}
        mock_session.post.return_value = mock_response
        
        result = wrapper.validate_spl_query("search index=main")
        
        assert result["valid"] is True
        assert result["query"] == "search index=main"
        assert "parsed" in result

    def test_validate_spl_query_invalid(self, mock_splunk_wrapper):
        """Test SPL query validation with invalid query."""
        wrapper, mock_session = mock_splunk_wrapper
        
        mock_session.post.side_effect = Exception("Invalid syntax")
        
        result = wrapper.validate_spl_query("invalid query")
        
        assert result["valid"] is False
        assert result["query"] == "invalid query"
        assert "error" in result

    def test_get_summary_info(self, mock_splunk_wrapper):
        """Test getting summary information."""
        wrapper, mock_session = mock_splunk_wrapper
        
        # Mock the individual method calls
        with patch.object(SplunkAPIWrapper, 'test_connection', return_value=True), \
     	     patch.object(SplunkAPIWrapper, 'get_indexes', return_value=["main", "security"]), \
             patch.object(SplunkAPIWrapper, 'get_sourcetypes', return_value=["syslog", "json"]), \
             patch.object(SplunkAPIWrapper, 'get_hosts', return_value=["host1", "host2"]):
            
            info = wrapper.get_summary_info()
            
            assert info["connection_status"] == "connected"
            assert info["total_indexes"] == 2
            assert len(info["indexes"]) == 2
            assert len(info["sample_sourcetypes"]) == 2
            assert len(info["sample_hosts"]) == 2

    def test_session_creation_with_token(self):
        """Test session creation with token authentication."""
        with patch('requests.Session') as mock_session_class:
            mock_session = Mock()
            mock_session_class.return_value = mock_session
            
            wrapper = SplunkAPIWrapper(
                splunk_host="test.com",
                splunk_token="test-token-123"
            )
            
            # Verify session was configured with Bearer token
            expected_headers = {
                "Authorization": "Bearer test-token-123",
                "Content-Type": "application/x-www-form-urlencoded"
            }
            mock_session.headers.update.assert_called_once_with(expected_headers)

    def test_session_creation_with_basic_auth(self):
        """Test session creation with basic authentication."""
        with patch('requests.Session') as mock_session_class:
            mock_session = Mock()
            mock_session_class.return_value = mock_session
            
            wrapper = SplunkAPIWrapper(
                splunk_host="test.com",
                splunk_token="",  # Empty token to trigger basic auth
                splunk_username="admin",
                splunk_password="password"
            )
            
            # Verify session was configured with basic auth
            assert mock_session.auth == ("admin", "password")
