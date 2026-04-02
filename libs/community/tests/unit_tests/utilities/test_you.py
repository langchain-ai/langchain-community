import json
import warnings
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, patch

import responses
from langchain_core.documents import Document

from langchain_community.utilities.you import YouSearchAPIWrapper

TEST_ENDPOINT = "https://ydc-index.io"
TEST_RESEARCH_ENDPOINT = "https://api.you.com"

# Mock v1/search response
MOCK_RESPONSE_RAW: Dict[str, Any] = {
    "results": {
        "web": [
            {
                "description": "Test description",
                "snippets": ["yo", "bird up"],
                "thumbnail_url": "https://example.com/image.gif",
                "title": "Test title 1",
                "url": "https://example.com/article.html",
                "favicon_url": "https://example.com/favicon.ico",
                "page_age": "2024-01-15T00:00:00Z",
            },
            {
                "description": "Test description 2",
                "snippets": ["worst show", "on tv"],
                "thumbnail_url": "https://example.com/image2.gif",
                "title": "Test title 2",
                "url": "https://example.com/article2.html",
                "favicon_url": "https://example.com/favicon2.ico",
                "page_age": "2024-01-16T00:00:00Z",
            },
        ],
        "news": [],
    }
}


def generate_parsed_metadata(num: Optional[int] = 0) -> Dict[str, Any]:
    """Generate metadata for testing."""
    if num is None:
        num = 0
    hit = MOCK_RESPONSE_RAW["results"]["web"][num]
    return {
        "url": hit["url"],
        "thumbnail_url": hit["thumbnail_url"],
        "title": hit["title"],
        "description": hit["description"],
        "favicon_url": hit["favicon_url"],
        "page_age": hit["page_age"],
    }


def generate_parsed_output(num: Optional[int] = 0) -> List[Document]:
    """Generate parsed output for testing."""
    if num is None:
        num = 0
    hit = MOCK_RESPONSE_RAW["results"]["web"][num]
    output = []
    for snippet in hit["snippets"]:
        doc = Document(page_content=snippet, metadata=generate_parsed_metadata(num))
        output.append(doc)
    return output


# Mock results after parsing
MOCK_PARSED_OUTPUT = generate_parsed_output()
MOCK_PARSED_OUTPUT.extend(generate_parsed_output(1))
# Single-snippet per hit
LIMITED_PARSED_OUTPUT = []
LIMITED_PARSED_OUTPUT.append(generate_parsed_output()[0])
LIMITED_PARSED_OUTPUT.append(generate_parsed_output(1)[0])

NEWS_RESPONSE_RAW: Dict[str, Any] = {
    "results": {
        "web": [],
        "news": [
            {
                "title": "Breaking News about the World's Greatest Search Engine!",
                "description": "Search on YDC for the news",
                "page_age": "2023-10-12T23:00:00Z",
                "thumbnail_url": "https://reuters.com/news.jpg",
                "url": "https://news.you.com",
            }
        ],
    }
}

NEWS_RESPONSE_PARSED = [
    Document(page_content=str(result["description"]), metadata=result)
    for result in NEWS_RESPONSE_RAW["results"]["news"]
]

NEWS_LIVECRAWL_RESPONSE_RAW: Dict[str, Any] = {
    "results": {
        "web": [],
        "news": [
            {
                "title": "Breaking News!",
                "description": "Short description",
                "page_age": "2024-01-01T00:00:00Z",
                "thumbnail_url": "https://example.com/news.jpg",
                "url": "https://news.example.com",
                "contents": {
                    "markdown": "# Full News Article\n\nDetailed content here.",
                },
            }
        ],
    }
}

LIVECRAWL_RESPONSE_RAW: Dict[str, Any] = {
    "results": {
        "web": [
            {
                "description": "Test description",
                "snippets": ["short snippet"],
                "thumbnail_url": "https://example.com/image.gif",
                "title": "Livecrawled Page",
                "url": "https://example.com/page.html",
                "favicon_url": "https://example.com/favicon.ico",
                "page_age": "2024-06-01T00:00:00Z",
                "contents": {
                    "markdown": "# Full Page Content\n\n"
                    "This is the livecrawled markdown.",
                },
            },
            {
                "description": "No livecrawl hit",
                "snippets": ["fallback snippet 1", "fallback snippet 2"],
                "thumbnail_url": None,
                "title": "Regular Page",
                "url": "https://example.com/regular.html",
                "favicon_url": "https://example.com/favicon2.ico",
                "page_age": None,
            },
        ],
        "news": [],
    }
}

LIVECRAWL_HTML_RESPONSE_RAW: Dict[str, Any] = {
    "results": {
        "web": [
            {
                "description": "HTML only",
                "snippets": ["snippet"],
                "thumbnail_url": None,
                "title": "HTML Page",
                "url": "https://example.com/html.html",
                "favicon_url": None,
                "page_age": None,
                "contents": {
                    "html": "<h1>Full HTML</h1><p>Content here.</p>",
                },
            },
        ],
        "news": [],
    }
}

MOCK_CONTENTS_RESPONSE: List[Dict[str, Any]] = [
    {
        "url": "https://example.com",
        "title": "Example Page",
        "html": "<h1>Hello</h1>",
        "markdown": "# Hello",
        "metadata": {
            "site_name": "Example",
            "favicon_url": "https://example.com/favicon.ico",
        },
    }
]

MOCK_CONTENTS_PARSED = [
    Document(
        page_content="# Hello",
        metadata={
            "url": "https://example.com",
            "title": "Example Page",
            "site_name": "Example",
            "favicon_url": "https://example.com/favicon.ico",
        },
    )
]

MOCK_RESEARCH_RESPONSE: Dict[str, Any] = {
    "output": {
        "content": "Quantum computing has seen major advances recently.[1]",
        "content_type": "text",
        "sources": [
            {
                "url": "https://nature.com/quantum",
                "title": "Nature Quantum",
                "snippets": ["Major breakthroughs in qubit stability."],
            },
            {
                "url": "https://arxiv.org/quantum",
                "snippets": ["Error correction improvements."],
            },
        ],
    }
}

MOCK_RESEARCH_TEXT = (
    "Quantum computing has seen major advances recently.[1]"
    "\n\n## Sources\n\n"
    "1. [Nature Quantum](https://nature.com/quantum)\n"
    "2. [https://arxiv.org/quantum](https://arxiv.org/quantum)"
)


@responses.activate
def test_raw_results() -> None:
    responses.add(
        responses.GET,
        f"{TEST_ENDPOINT}/v1/search",
        json=MOCK_RESPONSE_RAW,
        status=200,
    )
    query = "Test query text"
    you_wrapper = YouSearchAPIWrapper(ydc_api_key="test")
    raw_results = you_wrapper.raw_results(query)
    assert raw_results == MOCK_RESPONSE_RAW


@responses.activate
def test_raw_results_with_snippet_endpoint() -> None:
    responses.add(
        responses.GET,
        f"{TEST_ENDPOINT}/v1/search",
        json=MOCK_RESPONSE_RAW,
        status=200,
    )
    query = "Test query text"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        you_wrapper = YouSearchAPIWrapper(endpoint_type="snippet", ydc_api_key="test")
    raw_results = you_wrapper.raw_results(query)
    assert raw_results == MOCK_RESPONSE_RAW
    # Verify endpoint_type was NOT mutated
    assert you_wrapper.endpoint_type == "snippet"


@responses.activate
def test_results() -> None:
    responses.add(
        responses.GET,
        f"{TEST_ENDPOINT}/v1/search",
        json=MOCK_RESPONSE_RAW,
        status=200,
    )
    query = "Test query text"
    you_wrapper = YouSearchAPIWrapper(ydc_api_key="test")
    results = you_wrapper.results(query)
    assert results == MOCK_PARSED_OUTPUT


@responses.activate
def test_results_max_docs() -> None:
    responses.add(
        responses.GET,
        f"{TEST_ENDPOINT}/v1/search",
        json=MOCK_RESPONSE_RAW,
        status=200,
    )
    query = "Test query text"
    you_wrapper = YouSearchAPIWrapper(k=2, ydc_api_key="test")
    results = you_wrapper.results(query)
    expected_result = generate_parsed_output()
    assert results == expected_result


@responses.activate
def test_results_limit_snippets() -> None:
    responses.add(
        responses.GET,
        f"{TEST_ENDPOINT}/v1/search",
        json=MOCK_RESPONSE_RAW,
        status=200,
    )
    query = "Test query text"
    you_wrapper = YouSearchAPIWrapper(n_snippets_per_hit=1, ydc_api_key="test")
    results = you_wrapper.results(query)
    assert results == LIMITED_PARSED_OUTPUT


@responses.activate
def test_results_news() -> None:
    responses.add(
        responses.GET,
        f"{TEST_ENDPOINT}/v1/search",
        json=NEWS_RESPONSE_RAW,
        status=200,
    )
    query = "Test news text"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        you_wrapper = YouSearchAPIWrapper(endpoint_type="news", ydc_api_key="test")
    results = you_wrapper.results(query)
    assert results == NEWS_RESPONSE_PARSED


@responses.activate
def test_contents() -> None:
    responses.add(
        responses.POST,
        f"{TEST_ENDPOINT}/v1/contents",
        json=MOCK_CONTENTS_RESPONSE,
        status=200,
    )
    you_wrapper = YouSearchAPIWrapper(ydc_api_key="test")
    results = you_wrapper.contents(["https://example.com"])
    assert results == MOCK_CONTENTS_PARSED


@responses.activate
def test_search_params_passed_through() -> None:
    responses.add(
        responses.GET,
        f"{TEST_ENDPOINT}/v1/search",
        json=MOCK_RESPONSE_RAW,
        status=200,
    )
    you_wrapper = YouSearchAPIWrapper(
        ydc_api_key="test",
        count=5,
        freshness="week",
        offset=1,
        livecrawl="web",
        livecrawl_formats="markdown",
        country="US",
        safesearch="strict",
        language="fr",
    )
    you_wrapper.raw_results("test")
    request = responses.calls[0].request
    assert request.url is not None
    assert "count=5" in request.url
    assert "freshness=week" in request.url
    assert "offset=1" in request.url
    assert "livecrawl=web" in request.url
    assert "livecrawl_formats=markdown" in request.url
    assert "country=US" in request.url
    assert "safesearch=strict" in request.url
    assert "language=fr" in request.url


@responses.activate
def test_num_web_results_maps_to_count() -> None:
    responses.add(
        responses.GET,
        f"{TEST_ENDPOINT}/v1/search",
        json=MOCK_RESPONSE_RAW,
        status=200,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        you_wrapper = YouSearchAPIWrapper(ydc_api_key="test", num_web_results=5)
    you_wrapper.raw_results("test")
    request = responses.calls[0].request
    assert request.url is not None
    assert "count=5" in request.url


def test_deprecated_num_web_results_warning() -> None:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        YouSearchAPIWrapper(ydc_api_key="test", num_web_results=5)
        deprecation_warnings = [
            x for x in w if issubclass(x.category, DeprecationWarning)
        ]
        assert any("num_web_results" in str(x.message) for x in deprecation_warnings)


def test_deprecated_news_endpoint_warning() -> None:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        YouSearchAPIWrapper(ydc_api_key="test", endpoint_type="news")
        deprecation_warnings = [
            x for x in w if issubclass(x.category, DeprecationWarning)
        ]
        assert any("news" in str(x.message) for x in deprecation_warnings)


def test_deprecated_snippet_endpoint_warning() -> None:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        YouSearchAPIWrapper(ydc_api_key="test", endpoint_type="snippet")
        deprecation_warnings = [
            x for x in w if issubclass(x.category, DeprecationWarning)
        ]
        assert any("snippet" in str(x.message) for x in deprecation_warnings)


def test_deprecated_rag_endpoint_warning() -> None:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        YouSearchAPIWrapper(ydc_api_key="test", endpoint_type="rag")
        deprecation_warnings = [
            x for x in w if issubclass(x.category, DeprecationWarning)
        ]
        assert any("rag" in str(x.message) for x in deprecation_warnings)


def test_deprecated_n_hits_warning() -> None:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        YouSearchAPIWrapper(ydc_api_key="test", n_hits=5)
        deprecation_warnings = [
            x for x in w if issubclass(x.category, DeprecationWarning)
        ]
        assert any("n_hits" in str(x.message) for x in deprecation_warnings)


def test_deprecated_search_lang_warning() -> None:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        YouSearchAPIWrapper(ydc_api_key="test", search_lang="en")
        deprecation_warnings = [
            x for x in w if issubclass(x.category, DeprecationWarning)
        ]
        assert any("search_lang" in str(x.message) for x in deprecation_warnings)


def test_deprecated_ui_lang_warning() -> None:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        YouSearchAPIWrapper(ydc_api_key="test", ui_lang="en")
        deprecation_warnings = [
            x for x in w if issubclass(x.category, DeprecationWarning)
        ]
        assert any("ui_lang" in str(x.message) for x in deprecation_warnings)


def test_deprecated_spellcheck_warning() -> None:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        YouSearchAPIWrapper(ydc_api_key="test", spellcheck=True)
        deprecation_warnings = [
            x for x in w if issubclass(x.category, DeprecationWarning)
        ]
        assert any("spellcheck" in str(x.message) for x in deprecation_warnings)


def test_snippet_endpoint_does_not_mutate_state() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        wrapper = YouSearchAPIWrapper(endpoint_type="snippet", ydc_api_key="test")
    assert wrapper.endpoint_type == "snippet"


async def test_raw_results_async() -> None:
    instance = YouSearchAPIWrapper(ydc_api_key="test_api_key")

    mock_response = AsyncMock()
    mock_response.__aenter__.return_value = mock_response
    mock_response.__aexit__.return_value = None
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value=MOCK_RESPONSE_RAW)
    mock_response.raise_for_status = lambda: None

    with patch("aiohttp.ClientSession.get", return_value=mock_response):
        results = await instance.raw_results_async("test query")
        assert results == MOCK_RESPONSE_RAW


async def test_results_async() -> None:
    instance = YouSearchAPIWrapper(ydc_api_key="test_api_key")

    mock_response = AsyncMock()
    mock_response.__aenter__.return_value = mock_response
    mock_response.__aexit__.return_value = None
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value=MOCK_RESPONSE_RAW)
    mock_response.raise_for_status = lambda: None

    with patch("aiohttp.ClientSession.get", return_value=mock_response):
        results = await instance.results_async("test query")
        assert results == MOCK_PARSED_OUTPUT


async def test_results_news_async() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        instance = YouSearchAPIWrapper(endpoint_type="news", ydc_api_key="test_api_key")

    mock_response = AsyncMock()
    mock_response.__aenter__.return_value = mock_response
    mock_response.__aexit__.return_value = None
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value=NEWS_RESPONSE_RAW)
    mock_response.raise_for_status = lambda: None

    with patch("aiohttp.ClientSession.get", return_value=mock_response):
        results = await instance.results_async("test query")
        assert results == NEWS_RESPONSE_PARSED


async def test_contents_async() -> None:
    instance = YouSearchAPIWrapper(ydc_api_key="test_api_key")

    mock_response = AsyncMock()
    mock_response.__aenter__.return_value = mock_response
    mock_response.__aexit__.return_value = None
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value=MOCK_CONTENTS_RESPONSE)
    mock_response.raise_for_status = lambda: None

    with patch("aiohttp.ClientSession.post", return_value=mock_response):
        results = await instance.contents_async(["https://example.com"])
        assert results == MOCK_CONTENTS_PARSED


@responses.activate
def test_livecrawl_markdown_preferred_over_snippets() -> None:
    responses.add(
        responses.GET,
        f"{TEST_ENDPOINT}/v1/search",
        json=LIVECRAWL_RESPONSE_RAW,
        status=200,
    )
    wrapper = YouSearchAPIWrapper(ydc_api_key="test", livecrawl="web")
    results = wrapper.results("test")
    assert len(results) == 3
    assert results[0].page_content == (
        "# Full Page Content\n\nThis is the livecrawled markdown."
    )
    assert results[0].metadata["url"] == "https://example.com/page.html"
    assert results[1].page_content == "fallback snippet 1"
    assert results[2].page_content == "fallback snippet 2"


@responses.activate
def test_livecrawl_html_fallback() -> None:
    responses.add(
        responses.GET,
        f"{TEST_ENDPOINT}/v1/search",
        json=LIVECRAWL_HTML_RESPONSE_RAW,
        status=200,
    )
    wrapper = YouSearchAPIWrapper(ydc_api_key="test", livecrawl="web")
    results = wrapper.results("test")
    assert len(results) == 1
    assert results[0].page_content == "<h1>Full HTML</h1><p>Content here.</p>"


@responses.activate
def test_livecrawl_empty_contents_falls_back_to_snippets() -> None:
    empty_contents_response: Dict[str, Any] = {
        "results": {
            "web": [
                {
                    "description": "Has empty contents",
                    "snippets": ["real snippet"],
                    "thumbnail_url": None,
                    "title": "Page",
                    "url": "https://example.com/page.html",
                    "favicon_url": None,
                    "page_age": None,
                    "contents": {},
                },
            ],
            "news": [],
        }
    }
    responses.add(
        responses.GET,
        f"{TEST_ENDPOINT}/v1/search",
        json=empty_contents_response,
        status=200,
    )
    wrapper = YouSearchAPIWrapper(ydc_api_key="test", livecrawl="web")
    results = wrapper.results("test")
    assert len(results) == 1
    assert results[0].page_content == "real snippet"


@responses.activate
def test_livecrawl_ignores_n_snippets_per_hit() -> None:
    responses.add(
        responses.GET,
        f"{TEST_ENDPOINT}/v1/search",
        json=LIVECRAWL_RESPONSE_RAW,
        status=200,
    )
    wrapper = YouSearchAPIWrapper(
        ydc_api_key="test", livecrawl="web", n_snippets_per_hit=1
    )
    results = wrapper.results("test")
    assert results[0].page_content == (
        "# Full Page Content\n\nThis is the livecrawled markdown."
    )
    assert results[1].page_content == "fallback snippet 1"
    assert len(results) == 2


@responses.activate
def test_livecrawl_k_limits_total_docs() -> None:
    responses.add(
        responses.GET,
        f"{TEST_ENDPOINT}/v1/search",
        json=LIVECRAWL_RESPONSE_RAW,
        status=200,
    )
    wrapper = YouSearchAPIWrapper(ydc_api_key="test", livecrawl="web", k=1)
    results = wrapper.results("test")
    assert len(results) == 1
    assert results[0].page_content == (
        "# Full Page Content\n\nThis is the livecrawled markdown."
    )


@responses.activate
def test_news_livecrawl_prefers_contents_over_description() -> None:
    responses.add(
        responses.GET,
        f"{TEST_ENDPOINT}/v1/search",
        json=NEWS_LIVECRAWL_RESPONSE_RAW,
        status=200,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        wrapper = YouSearchAPIWrapper(
            ydc_api_key="test", endpoint_type="news", livecrawl="news"
        )
    results = wrapper.results("test")
    assert len(results) == 1
    assert results[0].page_content == ("# Full News Article\n\nDetailed content here.")


@responses.activate
def test_raw_research() -> None:
    responses.add(
        responses.POST,
        f"{TEST_RESEARCH_ENDPOINT}/v1/research",
        json=MOCK_RESEARCH_RESPONSE,
        status=200,
    )
    wrapper = YouSearchAPIWrapper(ydc_api_key="test")
    result = wrapper.raw_research("quantum computing advances")
    assert result == MOCK_RESEARCH_RESPONSE


@responses.activate
def test_research_text() -> None:
    responses.add(
        responses.POST,
        f"{TEST_RESEARCH_ENDPOINT}/v1/research",
        json=MOCK_RESEARCH_RESPONSE,
        status=200,
    )
    wrapper = YouSearchAPIWrapper(ydc_api_key="test")
    result = wrapper.research_text("quantum computing advances")
    assert result == MOCK_RESEARCH_TEXT


@responses.activate
def test_research_effort_passed_through() -> None:
    responses.add(
        responses.POST,
        f"{TEST_RESEARCH_ENDPOINT}/v1/research",
        json=MOCK_RESEARCH_RESPONSE,
        status=200,
    )
    wrapper = YouSearchAPIWrapper(ydc_api_key="test", research_effort="deep")
    wrapper.raw_research("test query")
    body = json.loads(responses.calls[0].request.body)
    assert body["research_effort"] == "deep"


@responses.activate
def test_research_effort_omitted_when_none() -> None:
    responses.add(
        responses.POST,
        f"{TEST_RESEARCH_ENDPOINT}/v1/research",
        json=MOCK_RESEARCH_RESPONSE,
        status=200,
    )
    wrapper = YouSearchAPIWrapper(ydc_api_key="test")
    wrapper.raw_research("test query")
    body = json.loads(responses.calls[0].request.body)
    assert "research_effort" not in body


def test_format_research_response_no_sources() -> None:
    raw: Dict[str, Any] = {
        "output": {"content": "Short answer.", "content_type": "text", "sources": []}
    }
    result = YouSearchAPIWrapper._format_research_response(raw)
    assert result == "Short answer."


async def test_raw_research_async() -> None:
    wrapper = YouSearchAPIWrapper(ydc_api_key="test")

    mock_response = AsyncMock()
    mock_response.__aenter__.return_value = mock_response
    mock_response.__aexit__.return_value = None
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value=MOCK_RESEARCH_RESPONSE)
    mock_response.raise_for_status = lambda: None

    with patch("aiohttp.ClientSession.post", return_value=mock_response):
        result = await wrapper.raw_research_async("quantum computing advances")
        assert result == MOCK_RESEARCH_RESPONSE


async def test_research_text_async() -> None:
    wrapper = YouSearchAPIWrapper(ydc_api_key="test")

    mock_response = AsyncMock()
    mock_response.__aenter__.return_value = mock_response
    mock_response.__aexit__.return_value = None
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value=MOCK_RESEARCH_RESPONSE)
    mock_response.raise_for_status = lambda: None

    with patch("aiohttp.ClientSession.post", return_value=mock_response):
        result = await wrapper.research_text_async("quantum computing advances")
        assert result == MOCK_RESEARCH_TEXT
