#!/usr/bin/env python3
"""
Quick test to check if methods exist on SplunkAPIWrapper
Run this to verify the fix worked
"""

import sys
from unittest.mock import patch

def test_methods_exist():
    print("🔍 Testing if SplunkAPIWrapper methods exist...")
    
    try:
        from langchain_community.utilities.splunk import SplunkAPIWrapper
        
        # Create instance with mocked session
        with patch('requests.Session'):
            wrapper = SplunkAPIWrapper(
                splunk_host="test.com",
                splunk_token="test-token"
            )
        
        # Check if methods exist
        methods_to_check = [
            'test_connection',
            'run_spl_query', 
            'get_indexes',
            'get_sourcetypes',
            'get_hosts',
            'get_summary_info',
            'validate_spl_query'
        ]
        
        missing_methods = []
        for method_name in methods_to_check:
            if hasattr(wrapper, method_name):
                print(f"✅ {method_name} exists")
            else:
                print(f"❌ {method_name} MISSING")
                missing_methods.append(method_name)
        
        if missing_methods:
            print(f"\n❌ Missing methods: {missing_methods}")
            return False
        else:
            print(f"\n✅ All {len(methods_to_check)} methods found!")
            return True
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_method_calls():
    print("\n🧪 Testing method calls...")
    
    try:
        from langchain_community.utilities.splunk import SplunkAPIWrapper
        from unittest.mock import Mock
        
        # Create instance with properly mocked session
        with patch('requests.Session') as mock_session_class:
            mock_session = Mock()
            mock_session_class.return_value = mock_session
            
            # Mock successful response
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {"entry": []}
            mock_session.get.return_value = mock_response
            
            wrapper = SplunkAPIWrapper(
                splunk_host="test.com",
                splunk_token="test-token"
            )
            
            # Try calling methods
            connection_result = wrapper.test_connection()
            print(f"✅ test_connection() returned: {connection_result}")
            
            indexes = wrapper.get_indexes()
            print(f"✅ get_indexes() returned: {indexes}")
            
            return True
            
    except Exception as e:
        print(f"❌ Method call error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🚀 Quick SplunkAPIWrapper Method Test")
    print("=" * 45)
    
    success = True
    
    if not test_methods_exist():
        success = False
    
    if not test_method_calls():
        success = False
    
    print("\n" + "=" * 45)
    if success:
        print("🎉 ALL METHOD TESTS PASSED!")
        print("✅ Your SplunkAPIWrapper should work with pytest now!")
    else:
        print("❌ Method tests failed!")
        print("Try replacing the file with the FINAL FIX version")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
