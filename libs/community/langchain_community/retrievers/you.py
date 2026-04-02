from typing import Any, List

from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from langchain_community.utilities import YouSearchAPIWrapper


class YouRetriever(BaseRetriever, YouSearchAPIWrapper):
    """You.com Search API retriever.

    Wraps ``YouSearchAPIWrapper`` to provide a LangChain retriever interface.
    Accepts all ``YouSearchAPIWrapper`` arguments.

    Setup:
        Set the ``YDC_API_KEY`` environment variable or pass ``ydc_api_key``
        directly.

    Example:
        .. code-block:: python

            from langchain_community.retrievers import YouRetriever

            retriever = YouRetriever()
            docs = retriever.invoke("latest AI news")
    """

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> List[Document]:
        return self.results(query, **kwargs)

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> List[Document]:
        return await self.results_async(query, **kwargs)
