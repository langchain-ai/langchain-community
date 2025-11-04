"""Test FireCrawlLoader."""

import sys
from typing import Generator, List, Tuple
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from langchain_community.document_loaders import FireCrawlLoader


# firecrawl 모듈을 모킹하여 sys.modules에 등록
@pytest.fixture(autouse=True)
def mock_firecrawl() -> Generator[Tuple[MagicMock, MagicMock], None, None]:
    """Mock firecrawl module for all tests."""
    mock_module = MagicMock()
    mock_client = MagicMock()
    # FirecrawlApp 클래스로 수정
    mock_module.FirecrawlApp.return_value = mock_client

    # extract 메서드의 반환값 설정
    response_dict = {
        "success": True,
        "data": {
            "title": "extracted title",
            "main contents": "extracted main contents",
        },
        "status": "completed",
        "expiresAt": "2025-03-12T12:42:09.000Z",
    }
    mock_client.extract.return_value = response_dict

    # sys.modules에 모의 모듈 삽입
    sys.modules["firecrawl"] = mock_module
    yield mock_module, mock_client  # 테스트에서 필요할 경우 접근할 수 있도록 yield

    # 테스트 후 정리
    if "firecrawl" in sys.modules:
        del sys.modules["firecrawl"]


class TestFireCrawlLoader:
    """Test FireCrawlLoader."""

    def test_load_extract_mode(
        self, mock_firecrawl: Tuple[MagicMock, MagicMock]
    ) -> List[Document]:
        """Test loading in extract mode."""
        # fixture에서 모킹된 객체 가져오기
        _, mock_client = mock_firecrawl

        params = {
            "prompt": "extract the title and main contents(write your own prompt here)",
            "schema": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "main contents": {"type": "string"},
                },
                "required": ["title", "main contents"],
            },
            "enableWebSearch": False,
            "ignoreSitemap": False,
            "showSources": False,
            "scrapeOptions": {
                "formats": ["markdown"],
                "onlyMainContent": True,
                "headers": {},
                "waitFor": 0,
                "mobile": False,
                "skipTlsVerification": False,
                "timeout": 30000,
                "removeBase64Images": True,
                "blockAds": True,
                "proxy": "basic",
            },
        }

        # FireCrawlLoader 인스턴스 생성 및 실행
        loader = FireCrawlLoader(
            url="https://example.com", api_key="fake-key", mode="extract", params=params
        )
        docs = list(loader.lazy_load())  # lazy_load 메서드 호출

        # 검증
        assert len(docs) == 1
        assert isinstance(docs[0].page_content, str)

        # v2: extract called with explicit kwargs (no params dict)
        forwarded = mock_client.extract.call_args
        assert forwarded is not None
        args, kwargs = forwarded
        assert args[0] == ["https://example.com"]
        assert kwargs.get("prompt") == params["prompt"]
        assert isinstance(kwargs.get("schema"), dict)
        assert kwargs.get("integration") == "langchain"

        # 응답이 문자열로 변환되었으므로 각 속성이 문자열에 포함되어 있는지 확인
        assert "extracted title" in docs[0].page_content
        assert "extracted main contents" in docs[0].page_content
        assert "success" in docs[0].page_content

        # print("[EXTRACT] docs:", docs)
        # print("[EXTRACT] forwarded:", mock_client.extract.call_args.kwargs)
        return docs

    def test_crawl_mode_strips_maxdepth_and_supports_prompt(
        self, mock_firecrawl: Tuple[MagicMock, MagicMock]
    ) -> List[Document]:
        """maxDepth should be ignored and not forwarded to the SDK in v2."""
        _, mock_client = mock_firecrawl

        crawl_response = {
            "success": True,
            "data": [
                {
                    "markdown": "Example page content",
                    "metadata": {"sourceURL": "https://example.com"},
                }
            ],
        }
        # v2 uses 'crawl'
        mock_client.crawl.return_value = crawl_response

        params = {
            "maxDepth": 3,
            "prompt": "Only crawl docs and blog posts",
            "crawlerOptions": {"maxDepth": 5},
        }

        loader = FireCrawlLoader(
            url="https://example.com", api_key="fake-key", mode="crawl", params=params
        )
        docs = list(loader.lazy_load())

        assert len(docs) == 1
        assert isinstance(docs[0], Document)
        # Ensure maxDepth was stripped from both top-level and crawlerOptions
        forwarded_params = mock_client.crawl.call_args.kwargs
        # Ensure legacy maxDepth not forwarded
        assert "maxDepth" not in forwarded_params
        assert "crawlerOptions" not in forwarded_params
        # Prompt should pass through
        assert forwarded_params["prompt"] == "Only crawl docs and blog posts"

        # print("[CRAWL] docs:", docs)
        # print("[CRAWL] forwarded:", forwarded_params)
        return docs

    def test_map_mode_maps_legacy_sitemap_flags(
        self, mock_firecrawl: Tuple[MagicMock, MagicMock]
    ) -> List[Document]:
        """ignoreSitemap/sitemapOnly should map to `sitemap` in v2."""
        _, mock_client = mock_firecrawl
        mock_client.map.return_value = {
            "links": ["https://example.com/a", "https://e.com/b"]
        }

        # Case 1: explicit sitemap=skip
        params = {"sitemap": "skip"}
        loader = FireCrawlLoader(
            url="https://example.com", api_key="fake-key", mode="map", params=params
        )
        docs = list(loader.lazy_load())
        assert len(docs) == 2
        assert isinstance(docs[0], Document)
        forwarded_params = mock_client.map.call_args.kwargs
        assert forwarded_params.get("sitemap") == "skip"
        # print("[MAP-1] docs:", docs)
        # print("[MAP-1] forwarded:", forwarded_params)

        # Case 2: explicit sitemap=only
        mock_client.map.reset_mock()
        params = {"sitemap": "only"}
        loader = FireCrawlLoader(
            url="https://example.com", api_key="fake-key", mode="map", params=params
        )
        _ = list(loader.lazy_load())
        forwarded_params = mock_client.map.call_args.kwargs
        assert forwarded_params.get("sitemap") == "only"
        # print("[MAP-2] forwarded:", forwarded_params)
        return docs

    def test_map_mode_handles_links_object_response(
        self, mock_firecrawl: Tuple[MagicMock, MagicMock]
    ) -> List[Document]:
        """Firecrawl v2 returns an object with `links` with url/title/description."""
        _, mock_client = mock_firecrawl
        mock_client.map.return_value = {
            "links": [
                {
                    "url": "https://firecrawl.dev",
                    "title": "Firecrawl",
                    "description": (
                        "Firecrawl is a platform for crawling and mapping websites."
                    ),
                },
                {
                    "url": "https://firecrawl.dev/blog",
                    "title": "Firecrawl Blog",
                    "description": "Firecrawl Blog is a blog about Firecrawl.",
                },
            ]
        }

        loader = FireCrawlLoader(
            url="https://firecrawl.dev", api_key="fake-key", mode="map", params={}
        )
        docs = list(loader.lazy_load())

        assert len(docs) == 2
        assert docs[0].page_content == "https://firecrawl.dev"
        assert docs[0].metadata.get("title") == "Firecrawl"
        assert "description" in docs[0].metadata

        # print("[MAP-LINKS] docs:", docs)
        return docs

    def test_search_mode_forwards_sources_param(
        self, mock_firecrawl: Tuple[MagicMock, MagicMock]
    ) -> List[Document]:
        """Ensure `sources` param is forwarded for search requests."""
        _, mock_client = mock_firecrawl
        mock_client.search.return_value = [
            {
                "markdown": "Result content",
                "metadata": {"sourceURL": "https://example.com/res"},
            }
        ]

        params = {
            "query": "firecrawl",
            "sources": [
                {"type": "web"},
                {"type": "images"},
                {"type": "news"},
            ],
        }

        loader = FireCrawlLoader(
            url="https://example.com", api_key="fake-key", mode="search", params=params
        )
        docs = list(loader.lazy_load())

        assert len(docs) == 1
        forwarded = mock_client.search.call_args.kwargs
        assert forwarded["query"] == "firecrawl"
        assert "sources" in forwarded
        assert isinstance(forwarded["sources"], list)
        # print("[SEARCH] docs:", docs)
        # print("[SEARCH] forwarded:", forwarded)
        return docs
