import warnings
from unittest.mock import AsyncMock, patch

import responses

from langchain_community.retrievers.you import YouRetriever

from ..utilities.test_you import (
    LIMITED_PARSED_OUTPUT,
    MOCK_PARSED_OUTPUT,
    MOCK_RESPONSE_RAW,
    NEWS_RESPONSE_PARSED,
    NEWS_RESPONSE_RAW,
    TEST_ENDPOINT,
)


class TestYouRetriever:
    @responses.activate
    def test_invoke(self) -> None:
        responses.add(
            responses.GET,
            f"{TEST_ENDPOINT}/v1/search",
            json=MOCK_RESPONSE_RAW,
            status=200,
        )
        query = "Test query text"
        you_wrapper = YouRetriever(ydc_api_key="test")
        results = you_wrapper.invoke(query)
        assert results == MOCK_PARSED_OUTPUT

    @responses.activate
    def test_invoke_max_docs(self) -> None:
        responses.add(
            responses.GET,
            f"{TEST_ENDPOINT}/v1/search",
            json=MOCK_RESPONSE_RAW,
            status=200,
        )
        query = "Test query text"
        you_wrapper = YouRetriever(k=2, ydc_api_key="test")
        results = you_wrapper.invoke(query)
        expected_result = [MOCK_PARSED_OUTPUT[0], MOCK_PARSED_OUTPUT[1]]
        assert results == expected_result

    @responses.activate
    def test_invoke_limit_snippets(self) -> None:
        responses.add(
            responses.GET,
            f"{TEST_ENDPOINT}/v1/search",
            json=MOCK_RESPONSE_RAW,
            status=200,
        )
        query = "Test query text"
        you_wrapper = YouRetriever(n_snippets_per_hit=1, ydc_api_key="test")
        results = you_wrapper.results(query)
        assert results == LIMITED_PARSED_OUTPUT

    @responses.activate
    def test_invoke_news(self) -> None:
        responses.add(
            responses.GET,
            f"{TEST_ENDPOINT}/v1/search",
            json=NEWS_RESPONSE_RAW,
            status=200,
        )
        query = "Test news text"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            you_wrapper = YouRetriever(endpoint_type="news", ydc_api_key="test")
        results = you_wrapper.results(query)
        assert results == NEWS_RESPONSE_PARSED

    async def test_ainvoke(self) -> None:
        instance = YouRetriever(ydc_api_key="test_api_key")

        mock_response = AsyncMock()
        mock_response.__aenter__.return_value = mock_response
        mock_response.__aexit__.return_value = None
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=MOCK_RESPONSE_RAW)
        mock_response.raise_for_status = lambda: None

        with patch("aiohttp.ClientSession.get", return_value=mock_response):
            results = await instance.ainvoke("test query")
            assert results == MOCK_PARSED_OUTPUT
