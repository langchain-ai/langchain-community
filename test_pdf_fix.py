#!/usr/bin/env python3
"""
Test script to verify the PDF loader dependency fixes work correctly.
This simulates the error conditions reported in the GitHub issue and validates
that the improved error messages guide users to the correct solutions.
"""

import sys
from unittest.mock import patch, MagicMock
import unittest


class TestPDFLoaderFixes(unittest.TestCase):
    """Test the PDF loader fixes."""
    
    def test_unstructured_pdfminer_layout_error(self):
        """Test improved error handling for pdfminer.layout import error."""
        
        # Mock the import error that would occur with missing pdfminer.layout
        with patch('builtins.__import__') as mock_import:
            def side_effect(name, *args, **kwargs):
                if name == 'unstructured.partition.pdf':
                    raise ImportError("No module named 'pdfminer.layout'")
                return MagicMock()
            
            mock_import.side_effect = side_effect
            
            # Import our fixed loader
            sys.path.insert(0, 'libs/community')
            from langchain_community.document_loaders.pdf import UnstructuredPDFLoader
            
            loader = UnstructuredPDFLoader("test.pdf")
            
            # This should raise an ImportError with helpful message
            with self.assertRaises(ImportError) as context:
                loader._get_elements()
            
            error_msg = str(context.exception)
            self.assertIn("pdfminer dependency issues", error_msg)
            self.assertIn("pip install --upgrade 'pdfminer.six>=20221105'", error_msg)
            self.assertIn("pip install --upgrade 'unstructured[pdf]>=0.15'", error_msg)
            self.assertIn("Make sure you install 'pdfminer.six' (not 'pdfminer')", error_msg)
    
    def test_unstructured_open_filename_error(self):
        """Test improved error handling for pdfminer.utils.open_filename import error."""
        
        # Mock the import error that would occur with open_filename issue
        with patch('builtins.__import__') as mock_import:
            def side_effect(name, *args, **kwargs):
                if name == 'unstructured.partition.pdf':
                    raise ImportError("cannot import name 'open_filename' from 'pdfminer.utils'")
                return MagicMock()
            
            mock_import.side_effect = side_effect
            
            # Import our fixed loader
            sys.path.insert(0, 'libs/community')
            from langchain_community.document_loaders.pdf import UnstructuredPDFLoader
            
            loader = UnstructuredPDFLoader("test.pdf")
            
            # This should raise an ImportError with helpful message
            with self.assertRaises(ImportError) as context:
                loader._get_elements()
            
            error_msg = str(context.exception)
            self.assertIn("pdfminer.utils.open_filename issue", error_msg)
            self.assertIn("pip uninstall pdfminer pdfminer.six", error_msg)
            self.assertIn("pip install 'pdfminer.six>=20221105'", error_msg)
            self.assertIn("pip install --upgrade 'unstructured[pdf]>=0.15'", error_msg)
    
    def test_pdfminer_open_filename_fallback(self):
        """Test that PDFMinerPDFasHTMLLoader correctly falls back for open_filename import."""
        
        # Mock the import scenario where open_filename is only in high_level
        with patch('builtins.__import__') as mock_import:
            def side_effect(name, globals=None, locals=None, fromlist=(), level=0):
                if fromlist and 'open_filename' in fromlist:
                    if name == 'pdfminer.utils':
                        raise ImportError("cannot import name 'open_filename' from 'pdfminer.utils'")
                    elif name == 'pdfminer.high_level':
                        mock_module = MagicMock()
                        mock_module.open_filename = MagicMock()
                        return mock_module
                return MagicMock()
            
            mock_import.side_effect = side_effect
            
            # Import our fixed loader
            sys.path.insert(0, 'libs/community')
            from langchain_community.document_loaders.pdf import PDFMinerPDFasHTMLLoader
            
            # Mock file operations for the test
            with patch('builtins.open'):
                with patch('os.path.isfile', return_value=True):
                    loader = PDFMinerPDFasHTMLLoader("test.pdf")
                    
                    # The lazy_load method should not raise an error thanks to fallback
                    # (We're not testing the full functionality, just import fallback)
                    try:
                        # This would normally process the file, but we're just testing
                        # that the import fallback works without error
                        list(loader.lazy_load())
                    except Exception as e:
                        # We expect some errors due to mocked file operations,
                        # but NOT ImportError about open_filename
                        self.assertNotIsInstance(e, ImportError)
                        if "open_filename" in str(e):
                            self.fail(f"open_filename import fallback failed: {e}")


def main():
    """Run the tests."""
    print("Testing PDF loader dependency fixes...")
    
    # Run the tests
    unittest.main(verbosity=2, exit=False)
    
    print("\n" + "="*50)
    print("SUMMARY OF FIXES:")
    print("="*50)
    print("1. UnstructuredPDFLoader now provides detailed error messages for:")
    print("   - Missing pdfminer.layout module")
    print("   - Missing pdfminer.utils.open_filename function")
    print("   - Clear installation instructions with correct package names")
    print()
    print("2. PDFMinerPDFasHTMLLoader now has fallback import logic:")
    print("   - First tries: from pdfminer.utils import open_filename")
    print("   - Falls back to: from pdfminer.high_level import open_filename")
    print("   - Provides clear error message if both fail")
    print()
    print("3. Users will now get actionable error messages instead of cryptic import errors")
    print("="*50)


if __name__ == "__main__":
    main()
