#!/usr/bin/env python3
"""
Simple test script to verify the fixed Splunk files work
Save this as test_fixed_files.py and run: python3 test_fixed_files.py
"""

import sys
import os
from unittest.mock import Mock, patch

# Add current directory to path
sys.path.insert(0, os.path.abspath('.'))

def test_imports():
    """Test that all imports work."""
    print("🔍 Testing imports...")
    
    try:
        from langchain_community.utilities.splunk import SplunkAPIWrapper
        print("✅ SplunkAPIWrapper imported")
    except Exception as e:
        print(f"❌ SplunkAPIWrapper import failed: {e}")
        return False
    
    try:
        from langchain_community.tools.splunk.tool import InfoSplunkTool, QuerySplunkTool
        print("✅ Tools imported")
    except Exception as e:
        print(f"❌ Tools import failed: {e}")
        return False
    
    try:
        from langchain_community.agent_toolkits.splunk.toolkit import SplunkToolkit
        print("✅ Toolkit imported")
    except Exception as e:
        print(f"❌ Toolkit import failed: {e}")
        return False
    
    return True

def test_creation():
    """Test creating objects."""
    print("\n🏗️ Testing object creation...")
    
    try:
        from langchain_community.utilities.splunk import SplunkAPIWrapper
        from langchain_community.tools.splunk.tool import InfoSplunkTool
        from langchain_community.agent_toolkits.splunk.toolkit import SplunkToolkit
        
        # Test SplunkAPIWrapper creation
        with patch('requests.Session'):
            wrapper = SplunkAPIWrapper(
                splunk_host="test.com",
                splunk_token="test-token"
            )
        print("✅ SplunkAPIWrapper created")
        
        # Test tool creation
        info_tool = InfoSplunkTool(splunk_wrapper=wrapper)
        print("✅ InfoSplunkTool created")
        
        # Test toolkit creation
        toolkit = SplunkToolkit(splunk_wrapper=wrapper)
        tools = toolkit.get_tools()
        print(f"✅ SplunkToolkit created with {len(tools)} tools")
        
        return True
        
    except Exception as e:
        print(f"❌ Object creation failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Testing Fixed Splunk Files")
    print("=" * 40)
    
    success = True
    
    if not test_imports():
        success = False
    
    if not test_creation():
        success = False
    
    print("\n" + "=" * 40)
    if success:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Your fixed files are working correctly!")
        print("\nNow you can try:")
        print("python3 -m pytest tests/unit_tests/utilities/test_splunk.py -v --override-ini='addopts='")
    else:
        print("❌ Some tests failed!")
        print("Make sure you've replaced the 3 files with the FIXED versions")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
