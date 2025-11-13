"""Test Parallel Search tools."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from langchain_community.tools.parallel_search.tool import (
    ParallelSearchResults,
    ParallelSearchRun,
)
from langchain_community.utilities.parallel_search import ParallelSearchAPIWrapper


def test_parallel_search_run_initialization() -> None:
    """Test ParallelSearchRun initialization."""
    tool = ParallelSearchRun(
        parallel_api_key="test-key",
        processor="base",
        max_results=5,
    )
    assert tool.name == "parallel_search"
    assert tool.processor == "base"
    assert tool.max_results == 5


def test_parallel_search_results_initialization() -> None:
    """Test ParallelSearchResults initialization."""
    tool = ParallelSearchResults(
        parallel_api_key="test-key",
        processor="pro",
        max_results=10,
    )
    assert tool.name == "parallel_search_results_json"
    assert tool.processor == "pro"
    assert tool.max_results == 10


@patch(
    "langchain_community.tools.parallel_search.tool.ParallelSearchAPIWrapper.results"
)
def test_parallel_search_run_invoke(mock_results: MagicMock) -> None:
    """Test ParallelSearchRun invoke method."""
    mock_results.return_value = [
        {
            "url": "https://example.com",
            "title": "Example",
            "excerpts": ["This is an example excerpt"],
        }
    ]

    tool = ParallelSearchRun(parallel_api_key="test-key")
    result = tool.invoke(
        {
            "objective": "test objective",
            "search_queries": ["test query"],
        }
    )

    assert "Result 1" in result
    assert "Example" in result
    assert "https://example.com" in result
    mock_results.assert_called_once()


@patch(
    "langchain_community.tools.parallel_search.tool.ParallelSearchAPIWrapper.results_async"
)
@pytest.mark.asyncio
async def test_parallel_search_run_ainvoke(mock_results_async: AsyncMock) -> None:
    """Test ParallelSearchRun async invoke method."""
    mock_results_async.return_value = [
        {
            "url": "https://example.com",
            "title": "Example",
            "excerpts": ["This is an example excerpt"],
        }
    ]

    tool = ParallelSearchRun(parallel_api_key="test-key")
    result = await tool.ainvoke(
        {
            "objective": "test objective",
            "search_queries": ["test query"],
        }
    )

    assert "Result 1" in result
    assert "Example" in result
    mock_results_async.assert_called_once()


@patch(
    "langchain_community.tools.parallel_search.tool.ParallelSearchAPIWrapper.results"
)
def test_parallel_search_results_invoke(mock_results: MagicMock) -> None:
    """Test ParallelSearchResults invoke method."""
    mock_results.return_value = [
        {
            "url": "https://example.com",
            "title": "Example",
            "excerpts": ["This is an example excerpt"],
        }
    ]

    tool = ParallelSearchResults(parallel_api_key="test-key")
    # Call _run directly to test the tuple return
    content, artifact = tool._run(
        objective="test objective",
        search_queries=["test query"],
    )

    assert isinstance(content, str)
    assert isinstance(artifact, list)
    assert len(artifact) == 1
    assert artifact[0]["url"] == "https://example.com"
    mock_results.assert_called_once()


@patch(
    "langchain_community.tools.parallel_search.tool.ParallelSearchAPIWrapper.results_async"
)
@pytest.mark.asyncio
async def test_parallel_search_results_ainvoke(mock_results_async: AsyncMock) -> None:
    """Test ParallelSearchResults async invoke method."""
    mock_results_async.return_value = [
        {
            "url": "https://example.com",
            "title": "Example",
            "excerpts": ["This is an example excerpt"],
        }
    ]

    tool = ParallelSearchResults(parallel_api_key="test-key")
    # Call _arun directly to test the tuple return
    content, artifact = await tool._arun(
        objective="test objective",
        search_queries=["test query"],
    )

    assert isinstance(content, str)
    assert isinstance(artifact, list)
    assert len(artifact) == 1
    mock_results_async.assert_called_once()


def test_parallel_search_run_format_results() -> None:
    """Test ParallelSearchRun result formatting."""
    tool = ParallelSearchRun(parallel_api_key="test-key")
    results = [
        {
            "url": "https://example.com",
            "title": "Example Title",
            "excerpts": ["Excerpt 1", "Excerpt 2"],
        },
        {
            "url": "https://example2.com",
            "title": "Example 2",
            "excerpts": [],
        },
    ]

    formatted = tool._format_results(results)

    assert "Result 1" in formatted
    assert "Example Title" in formatted
    assert "https://example.com" in formatted
    assert "Excerpt 1" in formatted
    assert "Result 2" in formatted
    assert "Example 2" in formatted


def test_parallel_search_run_no_results() -> None:
    """Test ParallelSearchRun with no results."""
    tool = ParallelSearchRun(parallel_api_key="test-key")
    formatted = tool._format_results([])
    assert formatted == "No results found."


@patch(
    "langchain_community.tools.parallel_search.tool.ParallelSearchAPIWrapper.results"
)
def test_parallel_search_run_error_handling(mock_results: MagicMock) -> None:
    """Test ParallelSearchRun error handling."""
    mock_results.side_effect = Exception("API Error")

    tool = ParallelSearchRun(parallel_api_key="test-key")
    result = tool.invoke({"objective": "test"})

    assert "API Error" in result
