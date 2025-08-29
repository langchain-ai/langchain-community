from __future__ import annotations

import logging
import uuid
from typing import (
    Any,
    Iterable,
    List,
    Optional,
    Tuple,
    TypeVar,
)

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from vecx.exceptions import NotFoundException

logger = logging.getLogger(__name__)

VST = TypeVar("VST", bound=VectorStore)


def _import_vectorx() -> Any:
    """Try to import vectorx module.

    If it's not already installed, instruct user how
    to install.

    Returns:
        Any: The imported vecx module.

    Raises:
        ImportError: If the vectorx python package is not installed.
    """
    try:
        import vecx
    except ImportError as e:
        raise ImportError(
            "Could not import vectorx python package. "
            "Please install it with `pip install vecx`."
        ) from e
    return vecx


class VectorXVectorStore(VectorStore):
    """Vector store for VectorX encrypted vector database."""

    def __init__(
        self,
        vectorx_index: Optional[Any] = None,
        embedding: Optional[Embeddings] = None,
        text_key: str = "text",
        api_token: Optional[str] = None,
        encryption_key: Optional[str] = None,
        index_name: Optional[str] = None,
        space_type: str = "cosine",
        dimension: Optional[int] = None,
    ):
        """Initialize with VectorX client.

        Args:
            vectorx_index: VectorX index instance
            embedding: Embedding function to use
            text_key: Key in metadata to store the text
            api_token: VectorX API token
            encryption_key: Encryption key for VectorX
            index_name: Name of the index
            space_type: Distance metric (cosine, l2, ip)
            dimension: Dimension of vectors
        """
        if embedding is None:
            raise ValueError("Embedding must be provided")
        self._embedding = embedding
        self._text_key = text_key

        # If index is not provided, initialize it
        if vectorx_index is None:
            if api_token is None:
                raise ValueError(
                    "API token must be provided if vectorx_index is not provided"
                )
            if index_name is None:
                raise ValueError(
                    "Index name must be provided if vectorx_index is not provided"
                )
            if encryption_key is None:
                raise ValueError(
                    "Encryption key must be provided if vectorx_index is not provided"
                )

            vectorx_index = self._initialize_vectorx_index(
                api_token, encryption_key, index_name, dimension, space_type
            )

        self._vectorx_index = vectorx_index

    @classmethod
    def _initialize_vectorx_index(
        cls,
        api_token: str,
        encryption_key: str,
        index_name: str,
        dimension: Optional[int] = None,
        space_type: str = "cosine",
    ) -> Any:
        """Initialize VectorX index using the current API."""
        _import_vectorx()
        from vecx.vectorx import VectorX

        # Initialize VectorX client
        vx = VectorX(token=api_token)

        try:
            # Try to get existing index
            index = vx.get_index(name=index_name, key=encryption_key)
            logger.info(f"Retrieved existing index: {index_name}")
            return index
        except NotFoundException as e:
            if dimension is None:
                raise ValueError(
                    "Must provide dimension when creating a new index"
                ) from e

            # Create a new index if it doesn't exist
            logger.info(f"Creating new index: {index_name}")
            vx.create_index(
                name=index_name,
                dimension=dimension,
                key=encryption_key,
                space_type=space_type,
            )
            return vx.get_index(name=index_name, key=encryption_key)

    @property
    def embeddings(self) -> Optional[Embeddings]:
        """Access the query embedding object if available."""
        return self._embedding

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        batch_size: int = 100,
        embedding_chunk_size: int = 100,
        *,
        async_req: bool = False,
        **kwargs: Any,
    ) -> List[str]:
        """Add texts to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            ids: Optional list of ids to associate with the texts.
            batch_size: Batch size for insertion.
            embedding_chunk_size: Batch size for embedding generation.
            async_req: Whether to make asynchronous request (not supported yet).

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        texts = list(texts)
        ids = ids or [str(uuid.uuid4()) for _ in texts]
        metadatas = metadatas or [{} for _ in texts]

        for metadata, text in zip(metadatas, texts):
            metadata[self._text_key] = text

        # Process in batches
        for i in range(0, len(texts), batch_size):
            chunk_texts = texts[i : i + batch_size]
            chunk_ids = ids[i : i + batch_size]
            chunk_metadatas = metadatas[i : i + batch_size]

            # Generate embeddings
            embeddings = []
            for embedding_batch_start in range(
                0, len(chunk_texts), embedding_chunk_size
            ):
                sub_texts = chunk_texts[
                    embedding_batch_start : embedding_batch_start + embedding_chunk_size
                ]
                sub_embeddings = self._embedding.embed_documents(sub_texts)
                embeddings.extend(sub_embeddings)

            # Prepare entries for upsert
            entries = []
            for id, embedding, metadata in zip(chunk_ids, embeddings, chunk_metadatas):
                # Extract filter values - keep it simple for efficient filtering
                filter_data = {}

                for key, value in metadata.items():
                    if key not in [self._text_key]:
                        filter_data[key] = value

                entry = {
                    "id": id,
                    "vector": embedding,
                    "meta": metadata,
                    "filter": filter_data,
                }
                entries.append(entry)

            # Insert to VectorX - encryption handled by client
            self._vectorx_index.upsert(entries)

        return ids

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Search with a text query and return documents and scores.

        Args:
            query: Text query.
            k: Number of results to return.
            filter: Optional filter dict.

        Returns:
            List of tuples of (document, similarity score).
        """
        embedding = self._embedding.embed_query(query)
        return self.similarity_search_by_vector_with_score(
            embedding, k=k, filter=filter
        )

    def similarity_search_by_vector_with_score(
        self,
        embedding: List[float],
        *,
        k: int = 4,
        filter: Optional[dict] = None,
    ) -> List[Tuple[Document, float]]:
        """Search by vector and return documents and scores.

        Args:
            embedding: Query embedding.
            k: Number of results to return.
            filter: Optional filter dict.

        Returns:
            List of tuples of (document, similarity score).
        """
        docs = []

        # Execute query - encryption handled by client
        results = self._vectorx_index.query(
            vector=embedding, top_k=k, filter=filter, include_vectors=False
        )

        # Process results
        for res in results:
            metadata = res["meta"]
            if self._text_key in metadata:
                text = metadata.pop(self._text_key)
                score = res["similarity"]
                docs.append((Document(page_content=text, metadata=metadata), score))
            else:
                logger.warning(
                    f"Found document with no `{self._text_key}` key. Skipping."
                )
        return docs

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Search with a text query and return documents.

        Args:
            query: Text query.
            k: Number of results to return.
            filter: Optional filter dict.

        Returns:
            List of documents.
        """
        docs_and_scores = self.similarity_search_with_score(
            query, k=k, filter=filter, **kwargs
        )
        return [doc for doc, _ in docs_and_scores]

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        vectorx_index: Optional[Any] = None,
        ids: Optional[List[str]] = None,
        batch_size: int = 32,
        text_key: str = "text",
        index_name: Optional[str] = None,
        api_token: Optional[str] = None,
        encryption_key: Optional[str] = None,
        space_type: str = "cosine",
        dimension: Optional[int] = None,
        **kwargs: Any,
    ) -> VectorXVectorStore:
        """Create a vector store from texts.

        Args:
            texts: List of texts to add.
            embedding: Embedding function.
            metadatas: Optional list of metadatas.
            vectorx_index: Optional VectorX index. If not provided, one will be created.
            ids: Optional list of ids.
            batch_size: Batch size for addition.
            text_key: Key to store text in metadata.
            index_name: Index name, required if vectorx_index not provided.
            api_token: VectorX API token, required if vectorx_index not provided.
            encryption_key: Encryption key, required if vectorx_index not provided.
            space_type: Distance metric type.
            dimension: Vector dimension, can be inferred if not provided.

        Returns:
            VectorXVectorStore instance.
        """
        # If dimension not provided, infer from embedding
        if dimension is None and vectorx_index is None:
            raise ValueError(
                "Dimension must be explicitly provided when creating a new index."
            )

        vectorx = cls(
            vectorx_index=vectorx_index,
            embedding=embedding,
            text_key=text_key,
            api_token=api_token,
            encryption_key=encryption_key,
            index_name=index_name,
            space_type=space_type,
            dimension=dimension,
        )

        vectorx.add_texts(
            texts,
            metadatas=metadatas,
            ids=ids,
            batch_size=batch_size,
        )
        return vectorx

    def delete(
        self,
        ids: Optional[List[str]] = None,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> None:
        """Delete by either ids or filter.

        Args:
            ids: List of ids to delete.
            filter: Filter to use for deletion.

        Returns:
            None
        """
        if ids is not None:
            # Delete by IDs - one at a time to avoid errors
            for id in ids:
                try:
                    self._vectorx_index.delete_vector(id)
                except Exception as e:
                    logger.warning(f"Error deleting vector with ID {id}: {e}")
        elif filter is not None:
            # Delete by filter
            try:
                self._vectorx_index.delete_with_filter(filter=filter)
            except Exception as e:
                logger.warning(f"Error deleting vectors with filter {filter}: {e}")
        else:
            raise ValueError("Either ids or filter must be provided.")

        return None

    @classmethod
    def from_params(
        cls,
        embedding: Embeddings,
        api_token: str,
        encryption_key: str,
        index_name: str,
        dimension: Optional[int] = None,
        space_type: str = "cosine",
        text_key: str = "text",
    ) -> "VectorXVectorStore":
        """Create VectorXVectorStore from parameters.

        Args:
            embedding: Embedding function
            api_token: VectorX API token
            encryption_key: Encryption key for VectorX
            index_name: Name of the index
            dimension: Dimension of vectors
            space_type: Distance metric (cosine, l2, ip)
            text_key: Key in metadata to store the text

        Returns:
            VectorXVectorStore instance
        """
        # If dimension not provided, infer from embedding
        if dimension is None:
            raise ValueError(
                "Dimension must be explicitly provided when creating a new index."
            )

        vectorx_index = cls._initialize_vectorx_index(
            api_token, encryption_key, index_name, dimension, space_type
        )

        return cls(
            vectorx_index=vectorx_index,
            embedding=embedding,
            text_key=text_key,
        )
