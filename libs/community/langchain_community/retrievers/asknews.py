import os
import re
from typing import Any, Dict, List, Literal, Optional

from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


class AskNewsRetriever(BaseRetriever):
    """AskNews retriever."""

    k: int = 10
    offset: int = 0
    start_timestamp: Optional[int] = None
    end_timestamp: Optional[int] = None
    method: Literal["nl", "kw"] = "kw"
    categories: List[
        Literal[
            "All",
            "Business",
            "Crime",
            "Politics",
            "Science",
            "Sports",
            "Technology",
            "Military",
            "Health",
            "Entertainment",
            "Finance",
            "Culture",
            "Climate",
            "Environment",
            "World",
        ]
    ] = ["All"]
    historical: bool = False
    similarity_score_threshold: float = 0.5
    kwargs: Optional[Dict[str, Any]] = {}
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    api_key: Optional[str] = None

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Get documents relevant to a query.
        Args:
            query: String to find relevant documents for
            run_manager: The callbacks handler to use
        Returns:
            List of relevant documents
        """
        try:
            from asknews_sdk import AskNewsSDK
        except ImportError:
            raise ImportError(
                "AskNews python package not found. "
                "Please install it with `pip install asknews`."
            )

        # Determine which authentication method to use
        has_client_credentials = bool(
            self.client_id or os.environ.get("ASKNEWS_CLIENT_ID")
        )
        has_api_key = bool(self.api_key or os.environ.get("ASKNEWS_API_KEY"))

        if has_client_credentials and has_api_key:
            raise ValueError(
                "Both client credentials and api_key provided. Please provide "
                "either client credentials or API key, not both."
            )

        if not has_client_credentials and not has_api_key:
            raise ValueError(
                "No authentication credentials provided. Please provide either "
                "client_id/client_secret or api_key."
            )

        # Initialize SDK with appropriate authentication method
        if has_api_key:
            an_client = AskNewsSDK(
                api_key=self.api_key or os.environ["ASKNEWS_API_KEY"],
                scopes=["news"],
            )
        else:
            an_client = AskNewsSDK(
                client_id=self.client_id or os.environ["ASKNEWS_CLIENT_ID"],
                client_secret=self.client_secret or os.environ["ASKNEWS_CLIENT_SECRET"],
                scopes=["news"],
            )
        response = an_client.news.search_news(
            query=query,
            n_articles=self.k,
            start_timestamp=self.start_timestamp,
            end_timestamp=self.end_timestamp,
            method=self.method,
            categories=self.categories,
            historical=self.historical,
            similarity_score_threshold=self.similarity_score_threshold,
            offset=self.offset,
            doc_start_delimiter="<doc>",
            doc_end_delimiter="</doc>",
            return_type="both",
            **self.kwargs,
        )

        return self._extract_documents(response)

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Asynchronously get documents relevant to a query.
        Args:
            query: String to find relevant documents for
            run_manager: The callbacks handler to use
        Returns:
            List of relevant documents
        """
        try:
            from asknews_sdk import AsyncAskNewsSDK
        except ImportError:
            raise ImportError(
                "AskNews python package not found. "
                "Please install it with `pip install asknews`."
            )

        # Determine which authentication method to use
        has_client_credentials = bool(
            self.client_id or os.environ.get("ASKNEWS_CLIENT_ID")
        )
        has_api_key = bool(self.api_key or os.environ.get("ASKNEWS_API_KEY"))

        if has_client_credentials and has_api_key:
            raise ValueError(
                "Both client credentials and api_key provided. Please provide "
                "either client credentials or API key, not both."
            )

        if not has_client_credentials and not has_api_key:
            raise ValueError(
                "No authentication credentials provided. Please provide either "
                "client_id/client_secret or api_key."
            )

        # Initialize SDK with appropriate authentication method
        if has_api_key:
            an_client = AsyncAskNewsSDK(
                api_key=self.api_key or os.environ["ASKNEWS_API_KEY"],
                scopes=["news"],
            )
        else:
            an_client = AsyncAskNewsSDK(
                client_id=self.client_id or os.environ["ASKNEWS_CLIENT_ID"],
                client_secret=self.client_secret or os.environ["ASKNEWS_CLIENT_SECRET"],
                scopes=["news"],
            )
        response = await an_client.news.search_news(
            query=query,
            n_articles=self.k,
            start_timestamp=self.start_timestamp,
            end_timestamp=self.end_timestamp,
            method=self.method,
            categories=self.categories,
            historical=self.historical,
            similarity_score_threshold=self.similarity_score_threshold,
            offset=self.offset,
            return_type="both",
            doc_start_delimiter="<doc>",
            doc_end_delimiter="</doc>",
            **self.kwargs,
        )

        return self._extract_documents(response)

    def _extract_documents(self, response: Any) -> List[Document]:
        """Extract documents from an api response."""

        from asknews_sdk.dto.news import SearchResponse

        sr: SearchResponse = response
        matches = re.findall(r"<doc>(.*?)</doc>", sr.as_string, re.DOTALL)
        docs = [
            Document(
                page_content=matches[i].strip(),
                metadata={
                    "title": sr.as_dicts[i].title,
                    "source": str(sr.as_dicts[i].article_url)
                    if sr.as_dicts[i].article_url
                    else None,
                    "images": sr.as_dicts[i].image_url,
                },
            )
            for i in range(len(matches))
        ]
        return docs
