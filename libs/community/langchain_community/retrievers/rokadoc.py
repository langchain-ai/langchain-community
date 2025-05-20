from http import HTTPStatus
from typing import List

import requests
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict
from typing_extensions import override


class RokadocSearchRetriever(BaseRetriever):
    """Rokadoc Search API retriever.

    Rokadoc is a Japanese-language specialized generative AI service.
    https://rokadoc.ntt.com/
    It includes features for converting documents into text and provides a simple retriever functionality.

    This class allows using a retriever from LangChain by utilizing a free API key that has usage limitations.
    """  # noqa: E501

    top_k: int = 3
    api_key: str = ""
    base_url: str = "https://beta-api.rokadoc.ntt.com/v1/"
    tags_filter: list["str"] = []

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    @override
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Get documents relevant to a query.

        Args:
            query: String to find relevant documents for.
            run_manager: The callback handler to use.

        Returns:
            List of relevant documents.
        """
        """Request a test."""

        doc_result: List[Document] = []

        req_params = {
            "message": query,
            "search_top_k": self.top_k,
        }

        if self.tags_filter:
            req_params["tags_filter_include"] = (True,)
            req_params["tags_filter"] = self.tags_filter

        res = requests.get(
            url=self.base_url + "api/search",
            headers={"api-key": self.api_key},
            params=req_params,
            timeout=30,
        )

        search_results = res.json()
        if res.status_code != HTTPStatus.OK.value:
            try:
                error_data = res.json()  # レスポンスのJSONを取得
            except ValueError:
                error_data = {"message": "No JSON response"}
            raise ValueError(
                f"Request failed with status {res.status_code}: {error_data}"
            )

        for search_result in search_results["search_result"]:
            document = Document(
                page_content=search_result["context"],
                metadata={
                    "source": self.base_url,
                    "page_number": search_result["page_number"],
                    "pdf_name": search_result["pdf_name"],
                },
            )
            doc_result.append(document)

        return doc_result
