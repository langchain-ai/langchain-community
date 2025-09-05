"""Unit tests for FeedCoopSearchAPIWrapper utility."""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from pydantic import SecretStr

from langchain_community.utilities.feedcoop_search import FeedCoopSearchAPIWrapper


class TestFeedCoopSearchAPIWrapperUnit(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.api_key = SecretStr("fake-key")
        self.wrapper = FeedCoopSearchAPIWrapper(feedcoop_api_key=self.api_key)

    @patch("langchain_community.utilities.feedcoop_search.requests.post")
    def test_raw_results_success(self, mock_post: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "ResponseMetadata": {},
            "Result": {
                "WebResults": [
                    {
                        "Title": "t",
                        "SiteName": "s",
                        "Url": "u",
                        "Snippet": "sn",
                        "Summary": "su",
                        "Content": "c",
                        "PublishTime": "p",
                        "LogoUrl": "l",
                        "RankScore": 1,
                        "AuthInfoDes": "a",
                        "AuthInfoLevel": 2,
                    }
                ]
            },
        }
        mock_post.return_value = mock_response
        result = self.wrapper.raw_results("test")
        self.assertIn("Result", result)
        self.assertIn("WebResults", result["Result"])

    @patch("langchain_community.utilities.feedcoop_search.requests.post")
    def test_raw_results_error(self, mock_post: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "ResponseMetadata": {"Error": {"CodeN": 123, "Message": "fail"}},
            "Result": {},
        }
        mock_post.return_value = mock_response
        with self.assertRaises(Exception) as cm:
            self.wrapper.raw_results("test")
        self.assertIn("FeedCoop API failed", str(cm.exception))

    def test_clean_results(self) -> None:
        input_results = [
            {
                "Title": "t",
                "SiteName": "s",
                "Url": "u",
                "Snippet": "sn",
                "Summary": "su",
                "Content": "c",
                "PublishTime": "p",
                "LogoUrl": "l",
                "RankScore": 1,
                "AuthInfoDes": "a",
                "AuthInfoLevel": 2,
            }
        ]
        clean = self.wrapper.clean_results(input_results)
        self.assertEqual(clean[0]["title"], "t")
        self.assertEqual(clean[0]["site_name"], "s")

    @patch(
        "langchain_community.utilities.feedcoop_search.FeedCoopSearchAPIWrapper.raw_results_async",
        new_callable=AsyncMock,
    )  # noqa: E501
    async def test_results_async(self, mock_raw_async: AsyncMock) -> None:
        mock_raw_async.return_value = {
            "Result": {
                "WebResults": [
                    {
                        "Title": "t",
                        "SiteName": "s",
                        "Url": "u",
                        "Snippet": "sn",
                        "Summary": "su",
                        "Content": "c",
                        "PublishTime": "p",
                        "LogoUrl": "l",
                        "RankScore": 1,
                        "AuthInfoDes": "a",
                        "AuthInfoLevel": 2,
                    }
                ]
            }
        }
        result = await self.wrapper.results_async("test")
        self.assertEqual(result[0]["title"], "t")

    @patch("langchain_community.utilities.feedcoop_search.aiohttp.ClientSession")
    def test_raw_results_async_error(self, mock_session: AsyncMock) -> None:
        mock_response = MagicMock()
        mock_response.status = 400
        from unittest.mock import AsyncMock

        mock_post = AsyncMock()
        mock_post.__aenter__.return_value = mock_response
        mock_session.return_value.__aenter__.return_value.post.return_value = mock_post

        with self.assertRaises(Exception):
            coro = self.wrapper.raw_results_async("test")
            asyncio.run(coro)
