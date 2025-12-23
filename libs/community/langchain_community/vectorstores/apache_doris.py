from __future__ import annotations

import json
import uuid
from typing import Any, Callable, Dict, Iterable, List, Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.utils import guard_import
from langchain_core.vectorstores import VectorStore

from langchain_community.vectorstores.utils import maximal_marginal_relevance


def guard():
    guard_import("doris_vector_search")

class ApacheDoris(VectorStore):
    """`Apache Doris` vector store.

    To use, you should have the `doris-vector-search` library available.
    You can install it or ensure it's in your PYTHONPATH.

    Args:
        embedding: Embedding to use for the vectorstore.
        host: Host of the Doris database. Defaults to "localhost".
        query_port: Query port of the Doris database. Defaults to 9030.
        http_port: HTTP port of the Doris database. Defaults to 8030.
        user: User for the Doris database. Defaults to "root".
        password: Password for the Doris database. Defaults to "".
        database: Database name. Defaults to "langchain".
        table_name: Name of the table to use. Defaults to "langchain".
        vector_key: Key to use for the vector in the database. Defaults to "langchain".
        id_key: Key to use for the id in the database. Defaults to "id".
        text_key: Key to use for the text in the database. Defaults to "text".

    Example:
        .. code-block:: python
            vectorstore = ApacheDoris(
                embedding_function,
                host="localhost",
                database="my_db",
                table_name="my_table"
            )
            vectorstore.add_texts(['text1', 'text2'])
            result = vectorstore.similarity_search('text1')
    """

    def __init__(
        self,
        embedding: Optional[Embeddings] = None,
        host: str = "localhost",
        query_port: int = 9030,
        http_port: int = 8030,
        user: str = "root",
        password: str = "",
        database: str = "langchain",
        table_name: str = "langchain",
        vector_key: str = "vector",
        id_key: str = "id",
        text_key: str = "text",
        **kwargs: Any,
    ):
        """Initialize with Apache Doris vectorstore"""
        self._embedding = embedding
        self.host = host
        self.query_port = query_port
        self.http_port = http_port
        self.user = user
        self.password = password
        self.database = database
        self.table_name = table_name
        self._vector_key = vector_key
        self._id_key = id_key
        self._text_key = text_key

        guard()
        import doris_vector_search

        # Create client
        if isinstance(kwargs.get("auth_options"), doris_vector_search.AuthOptions):
            auth_options = kwargs.get("auth_options")
        else:
            auth_options = doris_vector_search.AuthOptions(
                host=host,
                query_port=query_port,
                http_port=http_port,
                user=user,
                password=password,
            )
        self._client = doris_vector_search.DorisVectorClient(
            database=database, auth_options=auth_options
        )

        # Try to open table, if not exists, will create when adding data
        try:
            self._table = self._client.open_table(table_name)
        except Exception:
            self._table = None

    @property
    def embeddings(self) -> Optional[Embeddings]:
        return self._embedding

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Turn texts into embedding and add it to the database

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas(in dict) associated with the texts.
            ids: Optional list of ids to associate with the texts.

        Returns:
            List of ids of the added texts.
        """
        ids = ids or [str(uuid.uuid4()) for _ in texts]
        embeddings = self._embedding.embed_documents(list(texts))  # type: ignore[union-attr]

        # Prepare data as list of dicts
        data = []
        for idx, text in enumerate(texts):
            embedding = embeddings[idx]
            metadata = metadatas[idx] if metadatas else {"id": ids[idx]}
            data.append(
                {
                    self._id_key: ids[idx],
                    self._text_key: text,
                    self._vector_key: embedding,
                    "metadata": json.dumps(metadata),
                }
            )

        # Convert to pyarrow table
        import pyarrow as pa
        from doris_vector_search import IndexOptions

        table = pa.Table.from_pylist(data)

        if self._table is None:
            # Create table with data
            self._table = self._client.create_table(
                self.table_name, table, create_index=False
            )
            self._table.add_index(kwargs.get("index_options", IndexOptions()))
        else:
            # Add to existing table
            self._table.add(table)

        return ids

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return documents most similar to the query

        Args:
            query: String to query the vectorstore with.
            k: Number of documents to return.
            filter: Optional filter arguments as dict.

        Returns:
            List of documents most similar to the query.
        """
        embedding = self._embedding.embed_query(query)  # type: ignore[union-attr]
        return self.similarity_search_by_vector(embedding, k=k, filter=filter, **kwargs)

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filters: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """
        Return documents most similar to the query vector.
        """
        if self._table is None:
            raise ValueError("Table not initialized. Add texts first.")

        query = self._table.search(embedding).limit(k)

        # Apply filters if provided
        if filters:
            for condition in filters:
                query = query.where(condition)

        results = query.to_list()

        docs = []
        for row in results:
            docs.append(
                Document(
                    page_content=row[self._text_key],
                    metadata=json.loads(row.get("metadata", "{}")),
                )
            )
        return docs

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[tuple[Document, float]]:
        """Return documents most similar to the query with relevance scores."""
        embedding = self._embedding.embed_query(query)  # type: ignore[union-attr]
        return self.similarity_search_by_vector_with_relevance_scores(
            embedding, k=k, filter=filter, **kwargs
        )

    def similarity_search_by_vector_with_relevance_scores(
        self,
        embedding: List[float],
        k: int = 4,
        filters: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[tuple[Document, float]]:
        """
        Return documents most similar to the query vector with relevance scores.
        """
        if self._table is None:
            raise ValueError("Table not initialized. Add texts first.")

        query = self._table.search(embedding, include_distance=True).limit(k)

        # Apply filters if provided
        if filters:
            for condition in filters:
                query = query.where(condition)

        results = query.to_list()

        docs_and_scores = []
        for row in results:
            doc = Document(
                page_content=row[self._text_key],
                metadata=row.get("metadata", {}),
            )
            score = row.get("distance", 0.0)
            docs_and_scores.append((doc, score))
        return docs_and_scores

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        host: str = "localhost",
        query_port: int = 9030,
        http_port: int = 8030,
        user: str = "root",
        password: str = "",
        database: str = "langchain",
        table_name: str = "langchain",
        **kwargs: Any,
    ) -> ApacheDoris:
        guard()
        import doris_vector_search

        if isinstance(kwargs.get("auth_options"), doris_vector_search.AuthOptions):
            auth_options = kwargs.get("auth_options")
        else:
            auth_options = doris_vector_search.AuthOptions(
                host=host,
                query_port=query_port,
                http_port=http_port,
                user=user,
                password=password,
            )
        instance = cls(
            embedding=embedding,
            auth_options=auth_options,
            database=database,
            table_name=table_name,
            **kwargs,
        )
        instance.add_texts(texts, metadatas=metadatas, **kwargs)
        return instance

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        """
        The 'correct' relevance function
        may differ depending on a few things, including:
        - the distance / similarity metric used by the VectorStore
        - the scale of your embeddings (OpenAI's are unit normed. Many others are not!)
        - embedding dimensionality
        - etc.
        """
        if self._table is None:
            raise ValueError("Table not initialized. Add texts first.")

        metric_type = self._table.index_options.metric_type
        if metric_type == "l2_distance":
            return self._euclidean_relevance_score_fn
        elif metric_type == "inner_product":
            return self._max_inner_product_relevance_score_fn
        else:
            raise ValueError(
                "No supported normalization function"
                f" for distance metric of type: {metric_type}."
            )

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filters: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.
        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm. Defaults to
            20.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.
            filter (Optional[List[str]]): Filter by metadata. Defaults to None.

        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        if self._table is None:
            raise ValueError("Table not initialized. Add texts first.")

        query = self._table.search(embedding, include_distance=True).limit(fetch_k)

        # Apply filters if provided
        if filters:
            for condition in filters:
                query = query.where(condition)

        results = query.to_arrow()

        import numpy as np

        mmr_selected = maximal_marginal_relevance(
            np.array(embedding, dtype=np.float32),
            results["vector"].to_pylist(),
            k=k,
            lambda_mult=lambda_mult,
        )

        candidates = [
            Document(
                page_content=results[self._text_key][idx].as_py(),
                metadata=json.loads(results["metadata"][idx].as_py()),
            )
            for idx in range(len(results))
        ]

        selected_results = [r for i, r in enumerate(candidates) if i in mmr_selected]
        return selected_results

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        """Delete by vector ID or other criteria.

        Args:
            ids: List of ids to delete.
            **kwargs: Other keyword arguments that subclasses might use.

        Returns:
            Optional[bool]: True if deletion is successful,
            False otherwise, None if not implemented.
        """
        if ids is None:
            self._client.drop_table(self.table_name)

        raise NotImplementedError(
            "Deletions by ids are not implemented in doris vector search SDK yet."
        )
