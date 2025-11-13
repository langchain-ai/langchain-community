"""Test Parallel Search API wrapper."""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic import SecretStr

from langchain_community.utilities.parallel_search import ParallelSearchAPIWrapper


def test_api_key_explicit() -> None:
    """Test that the API key is correctly set when provided explicitly."""
    explicit_key = "explicit-api-key"
    wrapper = ParallelSearchAPIWrapper(parallel_api_key=SecretStr(explicit_key))
    assert wrapper.parallel_api_key.get_secret_value() == explicit_key


def test_api_key_from_env(monkeypatch: Any) -> None:
    """Test that the API key is correctly obtained from the environment variable."""
    env_key = "env-api-key"
    monkeypatch.setenv("PARALLEL_API_KEY", env_key)
    # Do not pass the api_key explicitly
    wrapper = ParallelSearchAPIWrapper()
    assert wrapper.parallel_api_key.get_secret_value() == env_key


def test_api_key_missing(monkeypatch: Any) -> None:
    """Test that instantiation fails when no API key is provided."""
    # Ensure that the environment variable is not set
    monkeypatch.delenv("PARALLEL_API_KEY", raising=False)
    with pytest.raises(ValueError):
        # This should raise an error because no api_key is available.
        ParallelSearchAPIWrapper()


def test_validate_requires_objective_or_search_queries() -> None:
    """Test that either objective or search_queries must be provided."""
    wrapper = ParallelSearchAPIWrapper(parallel_api_key=SecretStr("test-key"))
    with pytest.raises(ValueError, match="Either 'objective' or 'search_queries'"):
        wrapper.raw_results()


@patch("langchain_community.utilities.parallel_search.requests.post")
def test_raw_results_success(mock_post: MagicMock) -> None:
    """Test successful raw_results call."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "results": [
            {
                "url": "https://example.com",
                "title": "Example",
                "excerpts": ["This is an example"],
            }
        ]
    }
    mock_response.raise_for_status = MagicMock()
    mock_post.return_value = mock_response

    wrapper = ParallelSearchAPIWrapper(parallel_api_key=SecretStr("test-key"))
    result = wrapper.raw_results(objective="test objective")

    assert "results" in result
    mock_post.assert_called_once()
    call_kwargs = mock_post.call_args
    assert call_kwargs[1]["headers"]["x-api-key"] == "test-key"
    assert call_kwargs[1]["json"]["objective"] == "test objective"


@patch("langchain_community.utilities.parallel_search.requests.post")
def test_results_cleans_output(mock_post: MagicMock) -> None:
    """Test that results method cleans the output correctly."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "results": [
            {
                "url": "https://example.com",
                "title": "Example Title",
                "excerpts": ["Excerpt 1", "Excerpt 2"],
                "publish_date": "2025-01-01",
            }
        ]
    }
    mock_response.raise_for_status = MagicMock()
    mock_post.return_value = mock_response

    wrapper = ParallelSearchAPIWrapper(parallel_api_key=SecretStr("test-key"))
    results = wrapper.results(search_queries=["test query"])

    assert len(results) == 1
    assert results[0]["url"] == "https://example.com"
    assert results[0]["title"] == "Example Title"
    assert results[0]["excerpts"] == ["Excerpt 1", "Excerpt 2"]
    assert results[0]["publish_date"] == "2025-01-01"


def test_clean_results() -> None:
    """Test clean_results method."""
    wrapper = ParallelSearchAPIWrapper(parallel_api_key=SecretStr("test-key"))
    raw_results = [
        {
            "url": "https://example.com",
            "title": "Example",
            "excerpts": ["Excerpt 1"],
            "publish_date": "2025-01-01",
        },
        {
            "url": "https://example2.com",
            "title": "Example 2",
            "excerpts": [],
        },
    ]

    cleaned = wrapper.clean_results(raw_results)

    assert len(cleaned) == 2
    assert cleaned[0]["url"] == "https://example.com"
    assert cleaned[0]["title"] == "Example"
    assert cleaned[0]["excerpts"] == ["Excerpt 1"]
    assert cleaned[0]["publish_date"] == "2025-01-01"
    assert cleaned[1]["url"] == "https://example2.com"
    assert "publish_date" not in cleaned[1]
