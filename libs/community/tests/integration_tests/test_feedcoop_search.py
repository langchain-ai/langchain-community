"""Integration tests for FeedCoopSearchAPIWrapper utility."""

import os

import pytest
from pydantic import SecretStr

from langchain_community.utilities.feedcoop_search import FeedCoopSearchAPIWrapper


def test_feedcoop_search_real() -> None:
    api_key = os.environ.get("FEEDCOOP_API_KEY")
    if not api_key:
        pytest.skip("FEEDCOOP_API_KEY not set in environment")
    wrapper = FeedCoopSearchAPIWrapper(feedcoop_api_key=SecretStr(api_key))

    # 基本搜索
    results = wrapper.results("langchain", count=2)
    assert isinstance(results, list)
    assert len(results) >= 0
    if results:
        first = results[0]
        assert "title" in first
        assert "url" in first
        assert isinstance(first["title"], str)
        assert isinstance(first["url"], str)
        assert "site_name" in first
        assert "snippet" in first
        assert "content" in first
        assert "publish_time" in first

    # 搜索带 need_content
    results_content = wrapper.results("langchain", count=1, need_content=True)
    if results_content:
        assert results_content[0]["content"] is not None

    # 搜索带 need_url
    results_url = wrapper.results("langchain", count=1, need_url=True)
    if results_url:
        assert results_url[0]["url"].startswith("http")

    # 搜索带 need_summary
    results_summary = wrapper.results("langchain", count=1, need_summary=True)
    if results_summary:
        assert "summary" in results_summary[0]

    # 搜索带 include_domains
    results_domain = wrapper.results(
        "langchain", count=1, include_domains=["aliyun.com"]
    )
    if results_domain:
        assert (
            "aliyun.com" in results_domain[0]["url"]
            or results_domain[0]["site_name"].lower() == "aliyun"
        )
