#!/usr/bin/env python3
"""
Simple test runner for Splunk utilities without pytest
This completely avoids pytest configuration issues
"""

import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Add current directory to path
sys.path.insert(0, os.path.abspath('.'))

def test_splunk_api_wrapper():
    """Test SplunkAPIWrapper functionality without pytest."""
    
    print("🧪 Testing SplunkAPIWrapper")
    print("=" * 40)
    
    try:
        from langchain_community.utilities.splunk import SplunkAPIWrapper
        print("✅ Import successful")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    
    # Test 1: Initialization with token
    print("\n1. Testing initialization with token...")
    try:
        with patch('requests.Session'):
            wrapper = SplunkAPIWrapper(
                splunk_host="test.com",
                splunk_token="test-token"
            )
        print(f"✅ Token initialization: host={wrapper.splunk_host}, token={wrapper.splunk_token}")
    except Exception as e:
        print(f"❌ Token initialization failed: {e}")
        return False
    
    # Test 2: Initialization with username/password
    print("\n2. Testing initialization with username/password...")
    try:
        with patch('requests.Session'):
            wrapper = SplunkAPIWrapper(
                splunk_host="test.com",
                splunk_token="",
                splunk_username="admin",
                splunk_password="password"
            )
        print(f"✅ Username/password initialization: user={wrapper.splunk_username}")
    except Exception as e:
        print(f"❌ Username/password initialization failed: {e}")
        return False
    
    # Test 3: Validation error
    print("\n3. Testing validation error...")
    try:
        with patch('requests.Session'):
            try:
                wrapper = SplunkAPIWrapper(
                    splunk_host="test.com",
                    splunk_token="",  # Empty token
                    splunk_username=None,  # No username
                    splunk_password=None   # No password
                )
                print("❌ Should have raised ValueError")
                return False
            except ValueError:
                print("✅ Validation error correctly raised")
    except Exception as e:
        print(f"❌ Validation test failed: {e}")
        return False
    
    # Test 4: Base URL property
    print("\n4. Testing base_url property...")
    try:
        with patch('requests.Session'):
            wrapper = SplunkAPIWrapper(
                splunk_host="test.com",
                splunk_port=8089,
                splunk_token="test-token",
                splunk_scheme="https"
            )
            expected_url = "https://test.com:8089"
            actual_url = wrapper.base_url
            if actual_url == expected_url:
                print(f"✅ Base URL correct: {actual_url}")
            else:
                print(f"❌ Base URL incorrect: expected {expected_url}, got {actual_url}")
                return False
    except Exception as e:
        print(f"❌ Base URL test failed: {e}")
        return False
    
    # Test 5: Connection test success
    print("\n5. Testing connection test...")
    try:
        with patch('requests.Session') as mock_session_class:
            mock_session = Mock()
            mock_session_class.return_value = mock_session
            
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_session.get.return_value = mock_response
            
            wrapper = SplunkAPIWrapper(
                splunk_host="test.com",
                splunk_token="test-token"
            )
            
            result = wrapper.test_connection()
            if result is True:
                print("✅ Connection test successful")
            else:
                print(f"❌ Connection test failed: {result}")
                return False
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        return False
    
    # Test 6: Connection test failure
    print("\n6. Testing connection failure...")
    try:
        with patch('requests.Session') as mock_session_class:
            mock_session = Mock()
            mock_session_class.return_value = mock_session
            mock_session.get.side_effect = Exception("Connection failed")
            
            wrapper = SplunkAPIWrapper(
                splunk_host="test.com",
                splunk_token="test-token"
            )
            
            result = wrapper.test_connection()
            if result is False:
                print("✅ Connection failure handled correctly")
            else:
                print(f"❌ Connection failure not handled: {result}")
                return False
    except Exception as e:
        print(f"❌ Connection failure test failed: {e}")
        return False
    
    # Test 7: Get indexes
    print("\n7. Testing get_indexes...")
    try:
        with patch('requests.Session') as mock_session_class:
            mock_session = Mock()
            mock_session_class.return_value = mock_session
            
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
            
            wrapper = SplunkAPIWrapper(
                splunk_host="test.com",
                splunk_token="test-token"
            )
            
            indexes = wrapper.get_indexes()
            expected_indexes = ["main", "security", "web_logs"]
            if indexes == expected_indexes:
                print(f"✅ Get indexes successful: {indexes}")
            else:
                print(f"❌ Get indexes failed: expected {expected_indexes}, got {indexes}")
                return False
    except Exception as e:
        print(f"❌ Get indexes test failed: {e}")
        return False
    
    print("\n🎉 All SplunkAPIWrapper tests passed!")
    return True


def test_tools():
    """Test Splunk tools without pytest."""
    
    print("\n🛠️ Testing Splunk Tools")
    print("=" * 40)
    
    try:
        from langchain_community.tools.splunk.tool import (
            InfoSplunkTool,
            QuerySplunkTool,
            ListSplunkIndexesTool
        )
        print("✅ Tools import successful")
    except ImportError as e:
        print(f"❌ Tools import failed: {e}")
        return False
    
    # Test tool creation
    print("\n1. Testing tool creation...")
    try:
        mock_wrapper = Mock()
        mock_wrapper.get_summary_info.return_value = {
            "indexes": ["main", "security"],
            "connection_status": "connected"
        }
        
        info_tool = InfoSplunkTool(splunk_wrapper=mock_wrapper)
        query_tool = QuerySplunkTool(splunk_wrapper=mock_wrapper)
        indexes_tool = ListSplunkIndexesTool(splunk_wrapper=mock_wrapper)
        
        print(f"✅ Tools created: {info_tool.name}, {query_tool.name}, {indexes_tool.name}")
    except Exception as e:
        print(f"❌ Tool creation failed: {e}")
        return False
    
    print("\n🎉 All tool tests passed!")
    return True


def test_toolkit():
    """Test SplunkToolkit without pytest."""
    
    print("\n🧰 Testing SplunkToolkit")
    print("=" * 40)
    
    try:
        from langchain_community.agent_toolkits.splunk.toolkit import SplunkToolkit
        print("✅ Toolkit import successful")
    except ImportError as e:
        print(f"❌ Toolkit import failed: {e}")
        return False
    
    # Test toolkit creation
    print("\n1. Testing toolkit creation...")
    try:
        mock_wrapper = Mock()
        toolkit = SplunkToolkit(splunk_wrapper=mock_wrapper)
        tools = toolkit.get_tools()
        
        print(f"✅ Toolkit created with {len(tools)} tools")
        
        tool_names = [tool.name for tool in tools]
        expected_tools = ["splunk_list_indexes", "splunk_info", "splunk_query"]
        
        for expected_tool in expected_tools:
            if expected_tool in tool_names:
                print(f"✅ Found expected tool: {expected_tool}")
            else:
                print(f"❌ Missing expected tool: {expected_tool}")
                return False
                
    except Exception as e:
        print(f"❌ Toolkit creation failed: {e}")
        return False
    
    print("\n🎉 All toolkit tests passed!")
    return True


def main():
    """Run all tests."""
    print("🚀 Splunk Toolkit Test Suite")
    print("=" * 50)
    
    success = True
    
    # Run all test suites
    if not test_splunk_api_wrapper():
        success = False
    
    if not test_tools():
        success = False
        
    if not test_toolkit():
        success = False
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Your Splunk toolkit is working correctly!")
    else:
        print("❌ Some tests failed!")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
