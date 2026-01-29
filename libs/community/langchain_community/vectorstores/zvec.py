from __future__ import annotations

import logging
import os
import shutil
import uuid
from enum import Enum
from typing import (
    Any,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from langchain_community.vectorstores.utils import maximal_marginal_relevance

logger = logging.getLogger(__name__)


def _get_zvec_module() -> Any:
    """Get zvec module to avoid duplicate imports"""
    try:
        import zvec
    except ImportError:
        raise ImportError(
            "Could not import zvec python package. "
            "Please install it with `pip install zvec`."
        )
    return zvec


class IntRange(Enum):
    """Integer range constants for type determination."""

    INT32_MIN = -2147483648
    INT32_MAX = 2147483647
    UINT32_MAX = 4294967295
    INT64_MIN = -9223372036854775808
    INT64_MAX = 9223372036854775807
    UINT64_MAX = 18446744073709551615


class Zvec(VectorStore):
    """`Zvec` vector store.

    To use, you should have the ``zvec`` python package installed.

    Example:
        .. code-block:: python

            from langchain_community.vectorstores import Zvec
            from langchain_community.embeddings.openai import OpenAIEmbeddings
            import zvec

            collection_schema = zvec.CollectionSchema(
                name="langchain",
                fields=[
                    zvec.FieldSchema(
                        name="text",
                        data_type=zvec.DataType.STRING,
                    ),
                ],
                vectors=[
                    zvec.VectorSchema(
                        name="embedding",
                        data_type=zvec.DataType.VECTOR_FP32,
                        dimension=1024,
                        index_param=zvec.HnswIndexParam(metric_type=zvec.MetricType.COSINE),
                    ),
                ],
            )
            collection = zvec.create_and_open(
                    path="/path/to/my/collection",
                    schema=collection_schema,
                )
            collection = client.get("langchain")
            embeddings = OpenAIEmbeddings()
            vectorstore = Zvec(collection, embeddings.embed_query, "text")
    """

    def __init__(
        self,
        collection: Any,
        embedding: Embeddings,
        text_field: str,
    ):
        """Initialize with Zvec collection.
        Args:
            collection: A Zvec collection instance to connect to.
            embedding: Embeddings to use for encoding text.
            text_field: The name of the field in the collection that contains the text.
        """
        zvec = _get_zvec_module()
        if not isinstance(collection, zvec.Collection):
            raise ValueError(
                f"collection should be an instance of zvec.Collection, "
                f"but got {type(collection)}"
            )
        self._collection = collection
        self._embedding = embedding
        self._text_field = text_field

    def _similarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[str] = None,
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query vector, along with scores.

        Args:
            embedding: Query embedding vector.
            k: Number of documents to return. Defaults to 4.
            filter: Optional filter to apply to documents. Defaults to None.

        Returns:
            List of tuples of (document, similarity_score), where similarity_score
            is between 0 and 1, with 1 being most similar.
        """
        zvec = _get_zvec_module()

        query_obj = zvec.VectorQuery(
            field_name="embedding",
            vector=embedding,
        )

        ret = self._collection.query(query_obj, topk=k, filter=filter)
        if not ret:
            error_msg = getattr(self._collection, "message", "Unknown error")
            raise ValueError(f"Fail to query docs by vector, error {error_msg}")

        docs = []
        for doc in ret:
            metadata = doc.fields.copy()  # Create a copy to avoid modifying original
            text = metadata.pop(self._text_field)
            score = doc.score
            document = Document(page_content=text, metadata=metadata)
            docs.append((document, score))
        return docs

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        batch_size: int = 25,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.
        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            ids: Optional list of IDs to associate with the texts.
            batch_size: Optional batch size for embedding operations.
            **kwargs: Additional keyword arguments.
        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        zvec = _get_zvec_module()

        text_list = list(texts)
        ids = ids or [str(uuid.uuid4().hex) for _ in text_list]

        for i in range(0, len(text_list), batch_size):
            # batch end
            end = min(i + batch_size, len(text_list))

            batch_texts = text_list[i:end]
            batch_ids = ids[i:end]
            batch_embeddings = self._embedding.embed_documents(list(batch_texts))

            # batch metadatas
            if metadatas:
                batch_metadatas = metadatas[i:end]
            else:
                batch_metadatas = [{} for _ in range(i, end)]

            # Create zvec.Doc objects
            docs = []
            for doc_id, embedding, metadata, text in zip(
                batch_ids, batch_embeddings, batch_metadatas, batch_texts
            ):
                # Add text to metadata
                metadata_copy = (
                    metadata.copy()
                )  # Make a copy to avoid modifying original
                metadata_copy[self._text_field] = text
                doc = zvec.Doc(
                    id=doc_id, vectors={"embedding": embedding}, fields=metadata_copy
                )
                docs.append(doc)

            ret = self._collection.insert(docs)
            if not ret:
                raise ValueError(
                    f"Fail to insert docs to zvec vector database, "
                    f"Error: {getattr(ret, 'message', 'Unknown error')}"
                )
        return ids

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> bool:
        """Delete by vector ID.

        Args:
            ids: List of ids to delete.
            partition: a partition name in collection. [optional].

        Returns:
            True if deletion is successful,
            False otherwise.
        """
        return bool(self._collection.delete(ids))

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to query.

        Args:
            query: Text to search documents similar to.
            k: Number of documents to return. Default to 4.
            filter: Doc fields filter conditions that meet the SQL where clause
                    specification.
            partition: a partition name in collection. [optional].

        Returns:
            List of Documents most similar to the query text.
        """
        docs_and_scores = self.similarity_search_with_relevance_scores(query, k, filter)
        return [doc for doc, _ in docs_and_scores]

    def similarity_search_with_relevance_scores(
        self,
        query: str,
        k: int = 4,
        filter: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query text , alone with relevance scores.

        Less is more similar, more is more dissimilar.

        Args:
            query: input text
            k: Number of Documents to return. Defaults to 4.
            filter: Doc fields filter conditions that meet the SQL where clause
                    specification.
            partition: a partition name in collection. [optional].

        Returns:
            List of Tuples of (doc, similarity_score)
        """
        embedding = self._embedding.embed_query(query)
        return self._similarity_search_with_score_by_vector(
            embedding, k=k, filter=filter
        )

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Doc fields filter conditions that meet the SQL where clause
                    specification.
            partition: a partition name in collection. [optional].

        Returns:
            List of Documents most similar to the query vector.
        """
        docs_and_scores = self._similarity_search_with_score_by_vector(
            embedding, k, filter
        )
        return [doc for doc, _ in docs_and_scores]

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.
            filter: Doc fields filter conditions that meet the SQL where clause
                    specification.
            partition: a partition name in collection. [optional].

        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        embedding = self._embedding.embed_query(query)
        return self.max_marginal_relevance_search_by_vector(
            embedding, k, fetch_k, lambda_mult, filter
        )

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.
            filter: Doc fields filter conditions that meet the SQL where clause
                    specification.
            partition: a partition name in collection. [optional].

        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        zvec = _get_zvec_module()

        query_obj = zvec.VectorQuery(
            field_name="embedding",
            vector=embedding,
        )
        # query by vector
        ret = self._collection.query(
            query_obj,
            topk=fetch_k,
            filter=filter,
            include_vector=True,
        )
        if not ret:
            error_msg = getattr(self._collection, "message", "Unknown error")
            raise ValueError(f"Fail to query docs by vector, error {error_msg}")

        candidate_embeddings = [doc.vector for doc in ret]
        mmr_selected = maximal_marginal_relevance(
            np.array(embedding), candidate_embeddings, lambda_mult, k
        )

        # Fixed access to metadata fields
        metadatas = [ret[i].fields.copy() for i in mmr_selected]
        return [
            Document(page_content=metadata.pop(self._text_field), metadata=metadata)
            for metadata in metadatas
        ]

    @classmethod
    def get_zvec_datatype(cls, value: Any) -> Any:
        """Get zvec data type from value."""
        zvec = _get_zvec_module()
        if value is None:
            return zvec.DataType.STRING
        elif isinstance(value, bool):
            return zvec.DataType.BOOL
        elif isinstance(value, int):
            return cls._get_integer_datatype(value, zvec)
        elif isinstance(value, float):
            return zvec.DataType.DOUBLE
        elif isinstance(value, (str, bytes)):
            return zvec.DataType.STRING
        elif isinstance(value, (list, tuple)):
            return cls._get_array_datatype(value, zvec)
        else:
            return zvec.DataType.STRING

    @classmethod
    def _get_integer_datatype(cls, integer_value: int, zvec: Any) -> Any:
        """Determine the appropriate integer data type based on value range."""
        if IntRange.INT32_MIN.value <= integer_value <= IntRange.INT32_MAX.value:
            return zvec.DataType.INT32
        elif 0 <= integer_value <= IntRange.UINT32_MAX.value:
            return zvec.DataType.UINT32
        elif IntRange.INT64_MIN.value <= integer_value <= IntRange.INT64_MAX.value:
            return zvec.DataType.INT64
        else:  # integer_value > INT64_MAX (for positive) or out of bounds
            return zvec.DataType.UINT64

    @classmethod
    def _get_array_datatype(
        cls, array_value: Union[List[Any], Tuple[Any, ...]], zvec: Any
    ) -> Any:
        """Determine the appropriate array data type."""
        if not array_value:  # Empty array
            return zvec.DataType.ARRAY_STRING

        # Check if all elements have the same type
        element_types = {type(item).__name__ for item in array_value}
        if len(element_types) > 1:
            return zvec.DataType.ARRAY_STRING

        # All items have the same type - determine appropriate array type
        sample_item = array_value[0]
        if isinstance(sample_item, bool):
            return zvec.DataType.ARRAY_BOOL
        elif isinstance(sample_item, int):
            return zvec.DataType.ARRAY_INT32
        elif isinstance(sample_item, float):
            return zvec.DataType.ARRAY_DOUBLE
        elif isinstance(sample_item, str):
            return zvec.DataType.ARRAY_STRING
        else:
            return zvec.DataType.ARRAY_STRING

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        db_path: str = "zvec.db",
        collection_name: str = "langchain",
        text_field: str = "text",
        batch_size: int = 25,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Zvec:
        """Return Zvec VectorStore initialized from texts and embeddings.

        This is the quick way to get started with zvec vector store.

        Example:
            .. code-block:: python

            from langchain_community.vectorstores import Zvec
            from langchain_community.embeddings import OpenAIEmbeddings
            import zvec

            embeddings = OpenAIEmbeddings()
            zvec = Zvec.from_texts(
                docs,
                embeddings
            )
        """
        zvec = _get_zvec_module()

        if os.path.exists(db_path):
            if os.path.isfile(db_path):
                os.remove(db_path)
            elif os.path.isdir(db_path):
                shutil.rmtree(db_path)

        fields = [
            zvec.FieldSchema(
                name=text_field,
                data_type=zvec.DataType.STRING,
            ),
        ]

        if metadatas:
            for key, value in metadatas[0].items():
                field = zvec.FieldSchema(
                    name=key,
                    data_type=cls.get_zvec_datatype(value),
                )
                fields.append(field)

        # Get embedding dimension from first text
        if not texts:
            raise ValueError("texts list cannot be empty")
        dimension = len(embedding.embed_query(texts[0]))

        collection_schema = zvec.CollectionSchema(
            name=collection_name,
            fields=fields,
            vectors=[
                zvec.VectorSchema(
                    name="embedding",
                    data_type=zvec.DataType.VECTOR_FP32,
                    dimension=dimension,
                    index_param=zvec.HnswIndexParam(metric_type=zvec.MetricType.COSINE),
                ),
            ],
        )

        collection = zvec.create_and_open(db_path, collection_schema)
        if not collection:
            raise ValueError(f"Failed to create collection {collection_name}.")

        zvec_vector_db = cls(collection, embedding, text_field)
        zvec_vector_db.add_texts(texts, metadatas, ids, batch_size)
        return zvec_vector_db
