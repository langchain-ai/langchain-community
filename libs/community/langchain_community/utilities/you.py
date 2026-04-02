"""Wrapper for You.com Search, Contents, and Research APIs.

For setup instructions and API key, visit:
https://docs.you.com/get-started/quickstart
"""

import warnings
from typing import Any, Dict, List, Literal, Optional

import aiohttp
import requests
from langchain_core.documents import Document
from langchain_core.utils import get_from_dict_or_env
from pydantic import BaseModel, model_validator
from typing_extensions import Self

YOU_SEARCH_API_URL = "https://ydc-index.io"
YOU_RESEARCH_API_URL = "https://api.you.com"

# Sent in User-Agent so You.com can identify langchain-community traffic.
YOU_LANGCHAIN_USER_AGENT = "langchain-community-you"


class YouSearchAPIWrapper(BaseModel):
    """Wrapper for You.com Search, Contents, and Research APIs.

    To connect to the You.com API requires an API key which
    you can get at https://you.com/platform.
    You can check out the docs at https://docs.you.com/api-reference/.

    You need to set the environment variable ``YDC_API_KEY`` for the wrapper
    to operate.

    Attributes:
        ydc_api_key: You.com API key. If not set, reads from ``YDC_API_KEY``
            env var.
        endpoint_type: Determines which results to parse from the unified search
            response. ``"search"`` parses web results, ``"news"`` parses news
            results (deprecated: news is now served through the search endpoint).
        count: Maximum number of results per section (web/news). Defaults to 10.
        safesearch: Content filter level: ``"off"``, ``"moderate"``, or
            ``"strict"``.
        country: Country code for geographic focus (e.g. ``"US"``).
        freshness: Restrict results by recency. One of ``"day"``, ``"week"``,
            ``"month"``, ``"year"``, or a date range
            ``"YYYY-MM-DDtoYYYY-MM-DD"``.
        offset: Pagination offset (0--9). Results offset by ``count * offset``.
        livecrawl: Which sections to livecrawl for full page content:
            ``"web"``, ``"news"``, or ``"all"``.
        livecrawl_formats: Format of livecrawled content: ``"html"`` or
            ``"markdown"``.
        language: Language of results to return (BCP 47 format, e.g. ``"en"``).
            Defaults to ``"en"`` if not set.
        k: Maximum number of Documents to return from ``results()``.
        n_snippets_per_hit: Limit snippets returned per hit.
    """

    ydc_api_key: Optional[str] = None

    endpoint_type: Literal["search", "news", "rag", "snippet"] = "search"

    # v1 search params
    count: Optional[int] = None
    safesearch: Optional[Literal["off", "moderate", "strict"]] = None
    country: Optional[str] = None
    freshness: Optional[str] = None
    offset: Optional[int] = None
    livecrawl: Optional[Literal["web", "news", "all"]] = None
    livecrawl_formats: Optional[Literal["html", "markdown"]] = None
    language: Optional[str] = None

    k: Optional[int] = None
    n_snippets_per_hit: Optional[int] = None
    research_effort: Optional[Literal["lite", "standard", "deep", "exhaustive"]] = None

    # Deprecated fields kept for backwards compat
    num_web_results: Optional[int] = None
    n_hits: Optional[int] = None
    search_lang: Optional[str] = None
    ui_lang: Optional[str] = None
    spellcheck: Optional[bool] = None

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Any:
        """Validate that api key exists in environment."""
        ydc_api_key = get_from_dict_or_env(values, "ydc_api_key", "YDC_API_KEY")
        values["ydc_api_key"] = ydc_api_key

        return values

    @model_validator(mode="after")
    def _warn_deprecated_fields(self) -> Self:
        if self.num_web_results is not None:
            warnings.warn(
                "`num_web_results` is deprecated. Use `count` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if self.count is None:
                self.count = self.num_web_results
        if self.n_hits is not None:
            warnings.warn(
                "`n_hits` is deprecated and has no effect.",
                DeprecationWarning,
                stacklevel=2,
            )
        for field_name in ("search_lang", "ui_lang", "spellcheck"):
            if getattr(self, field_name) is not None:
                warnings.warn(
                    f"`{field_name}` is deprecated and has no effect.",
                    DeprecationWarning,
                    stacklevel=2,
                )
        if self.endpoint_type == "news":
            warnings.warn(
                '`endpoint_type="news"` is deprecated. Use '
                '`endpoint_type="search"` and read news from '
                "`results.news` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        if self.endpoint_type == "rag":
            warnings.warn(
                '`endpoint_type="rag"` is deprecated and returns Forbidden.',
                DeprecationWarning,
                stacklevel=2,
            )
        if self.endpoint_type == "snippet":
            warnings.warn(
                '`endpoint_type="snippet"` is deprecated. '
                'Use `endpoint_type="search"` instead.',
                DeprecationWarning,
                stacklevel=2,
            )
        if self.endpoint_type not in ("search", "snippet") and self.n_snippets_per_hit:
            warnings.warn(
                ("'n_snippets_per_hit' only has effect on `endpoint_type=\"search\"`."),
                UserWarning,
                stacklevel=2,
            )
        return self

    def _get_headers(self) -> Dict[str, str]:
        """Build headers for API requests."""
        return {
            "X-API-Key": self.ydc_api_key or "",
            "User-Agent": YOU_LANGCHAIN_USER_AGENT,
        }

    def _generate_params(self, query: str, **kwargs: Any) -> Dict:
        """Build query parameters for the v1/search endpoint.

        Args:
            query: The search query.

        Returns:
            Dict of non-None query parameters.
        """
        params: Dict[str, Any] = {
            "query": query,
            "count": self.count,
            "safesearch": self.safesearch,
            "country": self.country,
            "freshness": self.freshness,
            "offset": self.offset,
            "livecrawl": self.livecrawl,
            "livecrawl_formats": self.livecrawl_formats,
            "language": self.language,
            **kwargs,
        }
        return {k: v for k, v in params.items() if v is not None}

    def _parse_results(self, raw_search_results: Dict) -> List[Document]:
        """Extract Documents from the v1/search response.

        For ``endpoint_type="news"``, parses ``results.news``.
        For web results, prefers livecrawl ``contents`` (markdown then html)
        when available, falling back to ``snippets``.

        Args:
            raw_search_results: Raw JSON response from the search API.

        Returns:
            List of Documents with page content and metadata.
        """
        endpoint = "search" if self.endpoint_type == "snippet" else self.endpoint_type
        results = raw_search_results.get("results", {})

        if endpoint == "news":
            news_results = results.get("news", [])
            if self.k is not None:
                news_results = news_results[: self.k]
            docs = []
            for result in news_results:
                contents = result.get("contents") or {}
                page_content = (
                    contents.get("markdown")
                    or contents.get("html")
                    or result.get("description", "")
                )
                docs.append(Document(page_content=page_content, metadata=result))
            return docs

        docs = []
        for hit in results.get("web", []):
            meta = {
                "url": hit.get("url"),
                "thumbnail_url": hit.get("thumbnail_url"),
                "title": hit.get("title"),
                "description": hit.get("description"),
                "favicon_url": hit.get("favicon_url"),
                "page_age": hit.get("page_age"),
            }

            contents = hit.get("contents") or {}
            livecrawl_content = contents.get("markdown") or contents.get("html")
            if livecrawl_content:
                docs.append(Document(page_content=livecrawl_content, metadata=meta))
                if self.k is not None and len(docs) >= self.k:
                    return docs
                continue

            n_snippets = self.n_snippets_per_hit or len(hit.get("snippets", []))
            for snippet in hit.get("snippets", [])[:n_snippets]:
                docs.append(Document(page_content=snippet, metadata=dict(meta)))
                if self.k is not None and len(docs) >= self.k:
                    return docs
        return docs

    def raw_results(
        self,
        query: str,
        **kwargs: Any,
    ) -> Dict:
        """Run query through You.com Search and return the raw JSON response.

        Args:
            query: The query to search for.

        Returns:
            Raw API response dict.
        """
        headers = self._get_headers()
        params = self._generate_params(query, **kwargs)
        response = requests.get(
            f"{YOU_SEARCH_API_URL}/v1/search",
            params=params,
            headers=headers,
        )
        response.raise_for_status()
        return response.json()

    def results(
        self,
        query: str,
        **kwargs: Any,
    ) -> List[Document]:
        """Run query through You.com Search and return parsed Documents."""
        raw_search_results = self.raw_results(
            query,
            **{key: value for key, value in kwargs.items() if value is not None},
        )
        return self._parse_results(raw_search_results)

    async def raw_results_async(
        self,
        query: str,
        **kwargs: Any,
    ) -> Dict:
        """Get raw results from the You.com Search API asynchronously."""
        headers = self._get_headers()
        params = self._generate_params(query, **kwargs)

        async with aiohttp.ClientSession() as session:
            async with session.get(
                url=f"{YOU_SEARCH_API_URL}/v1/search",
                params=params,
                headers=headers,
            ) as response:
                response.raise_for_status()
                return await response.json()

    async def results_async(
        self,
        query: str,
        **kwargs: Any,
    ) -> List[Document]:
        """Run query through You.com Search asynchronously and return Documents."""
        raw_search_results = await self.raw_results_async(
            query,
            **{key: value for key, value in kwargs.items() if value is not None},
        )
        return self._parse_results(raw_search_results)

    def contents(
        self,
        urls: List[str],
        formats: Optional[List[Literal["html", "markdown", "metadata"]]] = None,
        crawl_timeout: Optional[float] = None,
    ) -> List[Document]:
        """Fetch clean content from URLs via the You.com Contents API.

        Args:
            urls: URLs to fetch content from.
            formats: Content formats to return (``"html"``, ``"markdown"``,
                ``"metadata"``). Defaults to server default.
            crawl_timeout: Per-URL crawl timeout in seconds (1--60).

        Returns:
            List of Documents with page content and metadata.
        """
        headers = self._get_headers()
        body: Dict[str, Any] = {"urls": urls}
        if formats is not None:
            body["formats"] = formats
        if crawl_timeout is not None:
            body["crawl_timeout"] = crawl_timeout

        response = requests.post(
            f"{YOU_SEARCH_API_URL}/v1/contents",
            json=body,
            headers=headers,
        )
        response.raise_for_status()
        return self._parse_contents_results(response.json())

    async def contents_async(
        self,
        urls: List[str],
        formats: Optional[List[Literal["html", "markdown", "metadata"]]] = None,
        crawl_timeout: Optional[float] = None,
    ) -> List[Document]:
        """Fetch content from URLs asynchronously via the You.com Contents API.

        Args:
            urls: URLs to fetch content from.
            formats: Content formats to return.
            crawl_timeout: Per-URL crawl timeout in seconds (1--60).

        Returns:
            List of Documents with page content and metadata.
        """
        headers = self._get_headers()
        body: Dict[str, Any] = {"urls": urls}
        if formats is not None:
            body["formats"] = formats
        if crawl_timeout is not None:
            body["crawl_timeout"] = crawl_timeout

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url=f"{YOU_SEARCH_API_URL}/v1/contents",
                json=body,
                headers=headers,
            ) as response:
                response.raise_for_status()
                return self._parse_contents_results(await response.json())

    def raw_research(self, query: str) -> Dict:
        """Call the You.com Research API and return the raw JSON response.

        Args:
            query: The research question or complex query.

        Returns:
            Raw API response dict with ``output.content`` and
            ``output.sources``.
        """
        headers = self._get_headers()
        body: Dict[str, Any] = {"input": query}
        if self.research_effort is not None:
            body["research_effort"] = self.research_effort

        response = requests.post(
            f"{YOU_RESEARCH_API_URL}/v1/research",
            json=body,
            headers=headers,
        )
        response.raise_for_status()
        return response.json()

    def research_text(self, query: str) -> str:
        """Research a topic and return a formatted markdown answer with sources.

        Args:
            query: The research question or complex query.

        Returns:
            Markdown-formatted answer followed by a numbered sources section.
        """
        return self._format_research_response(self.raw_research(query))

    async def raw_research_async(self, query: str) -> Dict:
        """Async variant of :meth:`raw_research`.

        Args:
            query: The research question or complex query.

        Returns:
            Raw API response dict with ``output.content`` and
            ``output.sources``.
        """
        headers = self._get_headers()
        body: Dict[str, Any] = {"input": query}
        if self.research_effort is not None:
            body["research_effort"] = self.research_effort

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url=f"{YOU_RESEARCH_API_URL}/v1/research",
                json=body,
                headers=headers,
            ) as response:
                response.raise_for_status()
                return await response.json()

    async def research_text_async(self, query: str) -> str:
        """Async variant of :meth:`research_text`.

        Args:
            query: The research question or complex query.

        Returns:
            Markdown-formatted answer followed by a numbered sources section.
        """
        return self._format_research_response(await self.raw_research_async(query))

    @staticmethod
    def _format_research_response(raw: Dict) -> str:
        """Format a raw research API response as markdown with a sources section.

        Args:
            raw: Raw JSON response dict from ``/v1/research``.

        Returns:
            Markdown answer followed by a numbered ``## Sources`` section.
        """
        output = raw.get("output", {})
        parts: List[str] = [output.get("content", "")]
        sources = output.get("sources", [])
        if sources:
            lines = ["", "## Sources", ""]
            for i, src in enumerate(sources, 1):
                title = src.get("title") or src.get("url", "")
                url = src.get("url", "")
                lines.append(f"{i}. [{title}]({url})")
            parts.append("\n".join(lines))
        return "\n".join(parts)

    @staticmethod
    def _parse_contents_results(raw_results: List[Dict]) -> List[Document]:
        """Convert Contents API response into Documents.

        Uses markdown content as page_content when available, falls back to html.
        """
        docs = []
        for page in raw_results:
            content = page.get("markdown") or page.get("html") or ""
            metadata: Dict[str, Any] = {
                "url": page.get("url"),
                "title": page.get("title"),
            }
            page_metadata = page.get("metadata")
            if page_metadata:
                metadata["site_name"] = page_metadata.get("site_name")
                metadata["favicon_url"] = page_metadata.get("favicon_url")
            docs.append(Document(page_content=content, metadata=metadata))
        return docs
