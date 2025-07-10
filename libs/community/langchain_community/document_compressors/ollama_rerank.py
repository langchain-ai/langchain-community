from copy import deepcopy
from typing import Self
from typing import TYPE_CHECKING, Optional, Sequence

from langchain_core.callbacks.base import Callbacks
from langchain_core.documents import BaseDocumentCompressor, Document
from pydantic import model_validator

from langchain_community.vectorstores.pgvector import BaseModel

if TYPE_CHECKING:
    from ollama import Client, AsyncClient
else:
    try:
        from ollama import Client, AsyncClient
    except ImportError:
        pass


class RerankRequest(BaseModel):
    model: str
    query: str
    top_n: int
    documents: list[str]
    return_documents: bool = False


class RerankResponseResult(BaseModel):
    index: int
    relevance_score: float


class RerankResponse(BaseModel):
    model: str
    results: list[RerankResponseResult]


class OllamaRerank(BaseDocumentCompressor):
    """Document compressor that uses `Ollama Rerank API`."""

    model: str
    """Model name to use."""

    base_url: Optional[str] = None
    """Base url the model is hosted under."""

    client_kwargs: Optional[dict] = {}
    """Additional kwargs to pass to the httpx clients. 
    These arguments are passed to both synchronous and async clients.
    Use sync_client_kwargs and async_client_kwargs to pass different arguments
    to synchronous and asynchronous clients.
    """

    async_client_kwargs: Optional[dict] = {}
    """Additional kwargs to merge with client_kwargs before
    passing to the httpx AsyncClient.
    For a full list of the params, see [this link](https://www.python-httpx.org/api/#asyncclient)
    """

    sync_client_kwargs: Optional[dict] = {}
    """Additional kwargs to merge with client_kwargs before
    passing to the httpx Client.
    For a full list of the params, see [this link](https://www.python-httpx.org/api/#client)
    """

    _client: Client = PrivateAttr(default=None)  # type: ignore
    """
    The client to use for making requests.
    """

    _async_client: AsyncClient = PrivateAttr(default=None)  # type: ignore
    """
    The async client to use for making requests.
    """

    top_n: Optional[int] = 3
    """Number of documents to return."""

    @model_validator(mode="after")
    def _set_clients(self) -> Self:
        """Set clients to use for ollama."""
        client_kwargs = self.client_kwargs or {}

        sync_client_kwargs = client_kwargs
        if self.sync_client_kwargs:
            sync_client_kwargs = {**sync_client_kwargs, **self.sync_client_kwargs}

        async_client_kwargs = client_kwargs
        if self.async_client_kwargs:
            async_client_kwargs = {**async_client_kwargs, **self.async_client_kwargs}

        self._client = Client(host=self.base_url, **sync_client_kwargs)
        self._async_client = AsyncClient(host=self.base_url, **async_client_kwargs)
        return self

    def compress_documents(
            self,
            documents: Sequence[Document],
            query: str,
            callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """
        Compress documents using Ollama's rerank API.
        Args:
            documents: A sequence of documents to compress.
            query: The query to use for compressing the documents.
            callbacks: Callbacks to run during the compression process.

        Returns:
            A sequence of compressed documents.
        """
        res = self._client._request(
            RerankResponse,
            'POST',
            "/api/rerank",
            json=RerankRequest(
                model=self.model,
                query=query,
                top_n=self.top_n,
                documents=[doc.page_content for doc in documents],
                return_documents=True,
            ).model_dump(),
        )
        compressed = []
        for result in res.results:
            doc = documents[result.index]
            doc_copy = Document(doc.page_content, metadata=deepcopy(doc.metadata))
            doc_copy.metadata["relevance_score"] = result.relevance_score
            compressed.append(doc_copy)
        return compressed

    async def acompress_documents(
            self,
            documents: Sequence[Document],
            query: str,
            callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """
        Asynchronously compress documents using Ollama's rerank API.

        Args:
            documents: A sequence of documents to compress.
            query: The query to use for compressing the documents.
            callbacks: Callbacks to run during the compression process.

        Returns:
            A sequence of compressed documents.
        """
        res = await self._async_client._request(
            RerankResponse,
            'POST',
            "/api/rerank",
            json=RerankRequest(
                model=self.model,
                query=query,
                top_n=self.top_n,
                documents=[doc.page_content for doc in documents],
                return_documents=True,
            ).model_dump(),
        )
        compressed = []
        for result in res.results:
            doc = documents[result.index]
            doc_copy = Document(doc.page_content, metadata=deepcopy(doc.metadata))
            doc_copy.metadata["relevance_score"] = result.relevance_score
            compressed.append(doc_copy)
        return compressed
