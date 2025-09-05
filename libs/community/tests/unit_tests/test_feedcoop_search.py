"""Unit tests for FeedCoopSearchAPIWrapper utility."""
import unittest
from unittest.mock import patch, MagicMock
from langchain_community.utilities.feedcoop_search import FeedCoopSearchAPIWrapper
from pydantic import SecretStr


class TestFeedCoopSearchAPIWrapperUnit(unittest.TestCase):
    def setUp(self):
        self.api_key = SecretStr("fake-key")
        self.wrapper = FeedCoopSearchAPIWrapper(feedcoop_api_key=self.api_key)

    @patch("langchain_community.utilities.feedcoop_search.requests.post")
    def test_raw_results_success(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "ResponseMetadata": {},
            "Result": {"WebResults": [{"Title": "t", "SiteName": "s", "Url": "u", "Snippet": "sn", "Summary": "su", "Content": "c", "PublishTime": "p", "LogoUrl": "l", "RankScore": 1, "AuthInfoDes": "a", "AuthInfoLevel": 2}]}
        }
        mock_post.return_value = mock_response
        result = self.wrapper.raw_results("test")
        self.assertIn("Result", result)
        self.assertIn("WebResults", result["Result"])

    @patch("langchain_community.utilities.feedcoop_search.requests.post")
    def test_raw_results_error(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "ResponseMetadata": {"Error": {"CodeN": 123, "Message": "fail"}},
            "Result": {}
        }
        mock_post.return_value = mock_response
        with self.assertRaises(Exception) as cm:
            self.wrapper.raw_results("test")
        self.assertIn("FeedCoop API failed", str(cm.exception))

    def test_clean_results(self):
        input_results = [{
            "Title": "t", "SiteName": "s", "Url": "u", "Snippet": "sn", "Summary": "su", "Content": "c", "PublishTime": "p", "LogoUrl": "l", "RankScore": 1, "AuthInfoDes": "a", "AuthInfoLevel": 2
        }]
        clean = self.wrapper.clean_results(input_results)
        self.assertEqual(clean[0]["title"], "t")
        self.assertEqual(clean[0]["site_name"], "s")
