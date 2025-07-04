import os
import unittest
from unittest.mock import patch
from langchain_community.tools.websearchplus.tool import WebSearchPlusResults
from langchain_community.utilities.websearchplus_search import WebSearchPlusAPIWrapper, WebSearchPlusInput

os.environ["WEBSEARCHPLUS_API_KEY"] = "your_api_key_here"  # Replace with your actual API key for testing

class TestWebSearchPlusTool(unittest.TestCase):
    @patch(
        "langchain_community.tools.websearchplus.tool.WebSearchPlusResults.invoke",
        return_value=[{"title": "Test Result", "link": "https://example.com", "snippet": "This is a test result."}],
    )
    def test_invoke(self, mock_run):
        query = "apple inc."
        input_data = WebSearchPlusInput(query=query)
        api_wrapper = WebSearchPlusAPIWrapper(websearchplus_api_key="your_api_key_here")  # type: ignore[arg-type]
        websearchplus_tool = WebSearchPlusResults(api_wrapper=api_wrapper)  # type: ignore[call-arg]
        input_dict = input_data.model_dump(exclude_unset=True, exclude_none=True)
        results = websearchplus_tool.invoke(input = input_dict)
        expected_result = [{"title": "Test Result", "link": "https://example.com", "snippet": "This is a test result."}]
        self.assertEqual(results, expected_result)

if __name__ == '__main__':
    unittest.main()