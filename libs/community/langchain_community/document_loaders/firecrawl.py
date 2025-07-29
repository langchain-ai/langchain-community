from typing import Iterator, Literal, Optional

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from langchain_core.utils import get_from_env


class FireCrawlLoader(BaseLoader):
    """
    FireCrawlLoader document loader integration

    Setup:
        Install ``firecrawl-py``,``langchain_community`` and set environment variable ``FIRECRAWL_API_KEY``.

        .. code-block:: bash

            pip install -U firecrawl-py langchain_community
            export FIRECRAWL_API_KEY="your-api-key"

    Instantiate:
        .. code-block:: python

            from langchain_community.document_loaders import FireCrawlLoader

            loader = FireCrawlLoader(
                url = "https://firecrawl.dev",
                mode = "crawl"
                # other params = ...
            )

    Lazy load:
        .. code-block:: python

            docs = []
            docs_lazy = loader.lazy_load()

            # async variant:
            # docs_lazy = await loader.alazy_load()

            for doc in docs_lazy:
                docs.append(doc)
            print(docs[0].page_content[:100])
            print(docs[0].metadata)

        .. code-block:: python

            Introducing [Smart Crawl!](https://www.firecrawl.dev/smart-crawl)
             Join the waitlist to turn any web
            {'ogUrl': 'https://www.firecrawl.dev/', 'title': 'Home - Firecrawl', 'robots': 'follow, index', 'ogImage': 'https://www.firecrawl.dev/og.png?123', 'ogTitle': 'Firecrawl', 'sitemap': {'lastmod': '2024-08-12T00:28:16.681Z', 'changefreq': 'weekly'}, 'keywords': 'Firecrawl,Markdown,Data,Mendable,Langchain', 'sourceURL': 'https://www.firecrawl.dev/', 'ogSiteName': 'Firecrawl', 'description': 'Firecrawl crawls and converts any website into clean markdown.', 'ogDescription': 'Turn any website into LLM-ready data.', 'pageStatusCode': 200, 'ogLocaleAlternate': []}

    Async load:
        .. code-block:: python

            docs = await loader.aload()
            print(docs[0].page_content[:100])
            print(docs[0].metadata)

        .. code-block:: python

            Introducing [Smart Crawl!](https://www.firecrawl.dev/smart-crawl)
             Join the waitlist to turn any web
            {'ogUrl': 'https://www.firecrawl.dev/', 'title': 'Home - Firecrawl', 'robots': 'follow, index', 'ogImage': 'https://www.firecrawl.dev/og.png?123', 'ogTitle': 'Firecrawl', 'sitemap': {'lastmod': '2024-08-12T00:28:16.681Z', 'changefreq': 'weekly'}, 'keywords': 'Firecrawl,Markdown,Data,Mendable,Langchain', 'sourceURL': 'https://www.firecrawl.dev/', 'ogSiteName': 'Firecrawl', 'description': 'Firecrawl crawls and converts any website into clean markdown.', 'ogDescription': 'Turn any website into LLM-ready data.', 'pageStatusCode': 200, 'ogLocaleAlternate': []}

    """  # noqa: E501

    def _prepare_crawler_options(self, params: dict) -> dict:
        """Prepare crawler options for the new API format."""
        # Transform old parameter names to new ones if present
        if "includes" in params:
            params["includePaths"] = params.pop("includes")
        if "excludes" in params:
            params["excludePaths"] = params.pop("excludes")
        if "allowBackwardCrawling" in params:
            params["allowBackwardLinks"] = params.pop("allowBackwardCrawling")
        if "allowExternalContentLinks" in params:
            params["allowExternalLinks"] = params.pop("allowExternalContentLinks")
        if "pageOptions" in params:
            params["scrapeOptions"] = self._prepare_scrape_options(
                params.pop("pageOptions")
            )

        return params

    def _prepare_scrape_options(self, params: dict) -> dict:
        """Prepare scrape options for the new API format."""
        # Handle extractor options transformation
        if "extractorOptions" in params:
            extractor = params.pop("extractorOptions")
            if "extractionPrompt" in extractor:
                params["prompt"] = extractor["extractionPrompt"]
            if "extractionSchema" in extractor:
                params["schema"] = extractor["extractionSchema"]
            if "userPrompt" in extractor:
                params["prompt"] = extractor["userPrompt"]

        # Transform include/exclude options to formats
        formats = []
        if params.pop("includeMarkdown", True):  # Default to True
            formats.append("markdown")
        if params.pop("includeHtml", False):
            formats.append("html")
        if params.pop("includeRawHtml", False):
            formats.append("rawHtml")
        if params.pop("includeExtract", False):
            formats.append("extract")
        if params.pop("includeLinks", False):
            formats.append("links")
        if params.pop("screenshot", False):
            formats.append("screenshot")
        if params.pop("fullPageScreenshot", False):
            formats.append("screenshot@fullPage")

        # Transform tag options
        if "onlyIncludeTags" in params:
            params["includeTags"] = params.pop("onlyIncludeTags")
        if "removeTags" in params:
            params["excludeTags"] = params.pop("removeTags")

        # Set formats if not already specified
        if "formats" not in params and formats:
            params["formats"] = formats
        elif "formats" not in params:
            params["formats"] = ["markdown"]  # Default

        return params

    def __init__(
        self,
        url: str,
        *,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        mode: Literal["crawl", "scrape", "map", "extract", "search"] = "crawl",
        params: Optional[dict] = None,
    ):
        """Initialize with API key and url.

        Args:
            url: The url to be crawled.
            api_key: The Firecrawl API key. If not specified will be read from env var
                FIRECRAWL_API_KEY. Get an API key
            api_url: The Firecrawl API URL. If not specified will be read from env var
                FIRECRAWL_API_URL or defaults to https://api.firecrawl.dev.
            mode: The mode to run the loader in. Default is "crawl".
                 Options include "scrape" (single url),
                 "crawl" (all accessible sub pages),
                 "map" (returns list of links that are semantically related).
                 "extract" (extracts structured data from a page).
                 "search" (search for data across the web).
            params: The parameters to pass to the Firecrawl API.
                Examples include crawlerOptions.
                For more details, visit: https://github.com/mendableai/firecrawl-py
        """

        try:
            from firecrawl import FirecrawlApp
        except ImportError:
            raise ImportError(
                "`firecrawl` package not found, please run `pip install firecrawl-py`"
            )
        if mode not in ("crawl", "scrape", "search", "map", "extract"):
            raise ValueError(
                f"""Invalid mode '{mode}'.
                Allowed: 'crawl', 'scrape', 'search', 'map', 'extract'."""
            )

        if not url:
            raise ValueError("Url must be provided")

        api_key = api_key or get_from_env("api_key", "FIRECRAWL_API_KEY")
        self.firecrawl = FirecrawlApp(api_key=api_key, api_url=api_url)
        self.url = url
        self.mode = mode
        self.params = params or {}

    def lazy_load(self) -> Iterator[Document]:
        self.params['integration'] = 'langchain'
        if self.mode == "scrape":
            firecrawl_docs = [
                self.firecrawl.scrape_url(
                    self.url, **self._prepare_scrape_options(self.params.copy())
                )
            ]
        elif self.mode == "crawl":
            if not self.url:
                raise ValueError("URL is required for crawl mode")
            crawl_response = self.firecrawl.crawl_url(
                self.url, **self._prepare_crawler_options(self.params.copy())
            )
            firecrawl_docs = crawl_response.get("data", [])
        elif self.mode == "map":
            if not self.url:
                raise ValueError("URL is required for map mode")
            map_response = self.firecrawl.map_url(self.url, **self.params)
            # Map returns a MapResponse object with links attribute
            firecrawl_docs = getattr(map_response, "links", [])
        elif self.mode == "extract":
            if not self.url:
                raise ValueError("URL is required for extract mode")
            firecrawl_docs = [str(self.firecrawl.extract([self.url], **self.params))]
        elif self.mode == "search":
            search_params = self.params.copy()
            query = search_params.pop("query", None)
            if not query:
                raise ValueError("Query parameter is required for search mode")
            firecrawl_docs = self.firecrawl.search(
                query=query,
                **search_params,
            )
        else:
            raise ValueError(
                f"""Invalid mode '{self.mode}'.
                Allowed: 'crawl', 'scrape', 'map', 'extract', 'search'."""
            )

        for doc in firecrawl_docs:
            if self.mode == "map" or self.mode == "extract":
                page_content = doc
                metadata: dict[str, str] = {}
            else:
                # New ScrapeResponse object format only
                page_content = (
                    getattr(doc, "markdown", None)
                    or getattr(doc, "html", None)
                    or getattr(doc, "rawHtml", None)
                    or ""
                )
                metadata = getattr(doc, "metadata", {}) or {}
            if not page_content:
                continue
            yield Document(
                page_content=page_content,
                metadata=metadata,
            )
