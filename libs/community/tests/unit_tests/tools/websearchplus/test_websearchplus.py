import os
import unittest
from unittest.mock import patch
from langchain_community.tools.websearchplus.tool import WebSearchPlusResults
from langchain_community.utilities.websearchplus_search import WebSearchPlusAPIWrapper, WebSearchPlusInput, WebSearchOptions

os.environ["WEBSEARCHPLUS_API_KEY"] = "your_api_key_here"  # Replace with your actual API key for testing

class TestWebSearchPlusTool(unittest.TestCase):

    @patch(
        "langchain_community.tools.websearchplus.tool.WebSearchPlusResults._run",
        return_value=[{"content": "High-relevance Result", "url": "https://example.com", "title": "This is a title of result.", "score": 0.95}],
    )
    def test_invoke(self, mock_run):
        query = "langchain"
        options: WebSearchOptions = WebSearchOptions(type="news",result_type="list")
        input_data = WebSearchPlusInput(query=query, options=options)
        api_wrapper = WebSearchPlusAPIWrapper(websearchplus_api_key="your_api_key_here")  # type: ignore[arg-type]
        websearchplus_tool = WebSearchPlusResults(api_wrapper=api_wrapper)  # type: ignore[call-arg]
        input_dict = input_data.model_dump(exclude_unset=True, exclude_none=True)
        results = websearchplus_tool.invoke(input = input_dict)
        expected_result = [{"content": "High-relevance Result", "url": "https://example.com", "title": "This is a title of result.", "score": 0.95}]
        self.assertEqual(results, expected_result)

if __name__ == '__main__':
    unittest.main()