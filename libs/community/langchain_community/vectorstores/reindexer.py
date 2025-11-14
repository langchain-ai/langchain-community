"""Reindexer vector store integration."""

from __future__ import annotations

import json
import math
import shutil
import uuid
from datetime import timedelta
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

try:
    from pyreindexer import RxConnector
    from pyreindexer.index_search_params import IndexSearchParamHnsw
    from pyreindexer.query import CondType
except ImportError:
    raise ImportError(
        "Could not import pyreindexer python package. "
        "Please install it with `pip install pyreindexer`"
    )

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.runnables.config import run_in_executor
from langchain_core.vectorstores import VectorStore
from langchain_core.vectorstores.utils import maximal_marginal_relevance

try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

VST = TypeVar("VST", bound=VectorStore)

DEFAULT_AUTOMATIC_DISK_DIR = "reindexer_storage"
MEMORY_DUMP_FILENAME = "reindexer_memory_dump.json"
DISK_DUMP_SUBDIR = "reindexer_disk_dump"


class ReindexerVectorStore(VectorStore):
    """Reindexer vector store integration.

    Setup:
        Install ``pyreindexer`` package.

        .. code-block:: bash

            pip install pyreindexer

    Key init args — indexing params:
        embedding: Embeddings
            Embedding function to use.
        m: int
            HNSW parameter - number of bi-directional links for each node (default: 16).
        ef_construction: int
            HNSW parameter - size of dynamic candidate list during construction (default: 200).
        multithreading: int
            Number of threads for index construction (default: 1).

    Key init args — connection params:
        rx_connector_config: dict
            Reindexer connector configuration.
            Example: ``{"dsn": "builtin:///tmp/my_db"}``
        rx_namespace: str
            Namespace name in Reindexer (default: "langchain").
        rx_index_definitions: Optional[List[dict]]
            Custom index definitions. If None, default indexes are used.

    Instantiate:
        .. code-block:: python

            from langchain_community.vectorstores import ReindexerVectorStore
            from langchain_openai import OpenAIEmbeddings

            vector_store = ReindexerVectorStore(
                embedding=OpenAIEmbeddings(),
                rx_connector_config={"dsn": "builtin:///tmp/my_db"},
                rx_namespace="my_namespace",
            )

    Add Documents:
        .. code-block:: python

            from langchain_core.documents import Document

            document_1 = Document(page_content="foo", metadata={"baz": "bar"})
            document_2 = Document(page_content="thud", metadata={"bar": "baz"})
            document_3 = Document(page_content="i will be deleted :(")

            documents = [document_1, document_2, document_3]
            ids = vector_store.add_documents(documents=documents)

    Delete Documents:
        .. code-block:: python

            vector_store.delete(ids=["id_3"])

    Search:
        .. code-block:: python

            results = vector_store.similarity_search(query="thud", k=1)
            for doc in results:
                print(f"* {doc.page_content} [{doc.metadata}]")

    Search with score:
        .. code-block:: python

            results = vector_store.similarity_search_with_score(query="qux", k=1)
            for doc, score in results:
                print(f"* [SIM={score:.3f}] {doc.page_content} [{doc.metadata}]")

    Use as Retriever:
        .. code-block:: python

            retriever = vector_store.as_retriever(
                search_kwargs={"k": 1},
            )
            retriever.invoke("thud")

    Search with filter:
        .. code-block:: python

            results = vector_store.similarity_search(
                query="thud", k=1, filter={"bar": "baz"}
            )
            for doc in results:
                print(f"* {doc.page_content} [{doc.metadata}]")

    MMR Search:
        .. code-block:: python

            results = vector_store.max_marginal_relevance_search(
                query="qux", k=2, fetch_k=10, lambda_mult=0.5
            )
            for doc in results:
                print(f"* {doc.page_content} [{doc.metadata}]")

    Async Search:
        .. code-block:: python

            import asyncio

            async def search():
                results = await vector_store.asimilarity_search(query="thud", k=1)
                return results

            docs = asyncio.run(search())

    Save/Load:
        .. code-block:: python

            # Save configuration
            vector_store.save_local("./my_vectorstore")

            # Load configuration
            loaded_store = ReindexerVectorStore.load_local(
                "./my_vectorstore", embedding=OpenAIEmbeddings()
            )
    """  # noqa: E501

    def __init__(
        self,
        embedding: Embeddings,
        m: int = 16,
        ef_construction: int = 200,
        multithreading: int = 1,
        rx_connector_config: Optional[dict] = None,
        rx_namespace: str = "langchain",
        rx_index_definitions: Optional[List[dict]] = None,
    ) -> None:
        """Initialize with the given embedding function.

        Args:
            embedding: Embedding function to use.
            m: HNSW parameter - number of bi-directional links for each node.
            ef_construction: HNSW parameter - size of dynamic candidate list
                during construction.
            multithreading: Number of threads for index construction.
            rx_connector_config: Reindexer connector configuration.
                Default: {"dsn": "builtin:///tmp/pyrx",
                          "max_replication_updates_size": 10 * 1024 * 1024}
            rx_namespace: Namespace name in Reindexer.
            rx_index_definitions: Custom index definitions. If None, default
                indexes are used.
        """
        if multithreading not in (0, 1):
            msg = f"multithreading must be 0 or 1. Got {multithreading}."
            raise ValueError(msg)

        if rx_connector_config is None:
            rx_connector_config = {
                "dsn": "builtin://",  # by default memory load
                "max_replication_updates_size": 10 * 1024 * 1024,
            }
        self._rx_connector_config = dict(rx_connector_config)
        self._rx_namespace = rx_namespace

        self._storage_mode: Literal["memory", "disk", "external"] = "external"
        self._auto_disk_path = False
        self._auto_disk_root_dir = DEFAULT_AUTOMATIC_DISK_DIR
        self._disk_path: Optional[Path] = None
        self._config_dsn_value = self._rx_connector_config.get("dsn", "")
        self._prepare_storage_backend()

        self._multithreading = multithreading

        self.embedding = embedding
        dimension = len(self.embedding.embed_query("dimension"))
        if rx_index_definitions is None:
            self._rx_index_definitions = [
                {
                    "name": "id",
                    "json_paths": ["id"],
                    "field_type": "string",
                    "index_type": "hash",
                    "is_pk": True,
                    "is_array": False,
                    "is_dense": False,
                    "is_sparse": False,
                    "is_no_column": False,
                    "collate_mode": "none",
                    "sort_order_letters": "",
                    "expire_after": 0,
                    "config": {},
                },
                {
                    "name": "vector",
                    "json_paths": ["vector"],
                    "field_type": "float_vector",
                    "index_type": "hnsw",
                    "config": {
                        "dimension": dimension,
                        "metric": "cosine",
                        "start_size": 100,
                        "m": m,
                        "ef_construction": ef_construction,
                        "multithreading": multithreading,
                    },
                },
            ]
        else:
            self._rx_index_definitions = rx_index_definitions

        self.init_rx()

    def _prepare_storage_backend(self) -> None:
        """Configure storage paths based on DSN."""
        dsn = self._rx_connector_config.get("dsn")
        if not dsn:
            return

        if not dsn.startswith("builtin://"):
            self._config_dsn_value = dsn
            self._storage_mode = "external"
            return

        path_part = dsn[len("builtin://") :]
        if path_part == "":
            self._storage_mode = "memory"
            self._rx_connector_config["dsn"] = "builtin://"
            self._config_dsn_value = "builtin://"
            return

        self._storage_mode = "disk"
        if path_part == "/":
            self._auto_disk_path = True
            disk_path = self._automatic_disk_path(
                self._rx_namespace,
                directory_name=self._auto_disk_root_dir,
            )
            disk_path.mkdir(parents=True, exist_ok=True)
        else:
            disk_path = self._resolve_disk_path(path_part)

        self._disk_path = disk_path
        disk_path_str = disk_path.as_posix()
        self._rx_connector_config["dsn"] = f"builtin://{disk_path_str}"
        self._config_dsn_value = (
            "builtin:///" if self._auto_disk_path else f"builtin://{disk_path_str}"
        )

    @staticmethod
    def _automatic_disk_path(
        namespace: str,
        *,
        root: Optional[Path] = None,
        directory_name: str = DEFAULT_AUTOMATIC_DISK_DIR,
    ) -> Path:
        """Return default disk path for automatic persistence without creating it."""
        base_root = root or Path.cwd()
        base_dir = (base_root / directory_name).resolve()
        return (base_dir / namespace).resolve()

    @staticmethod
    def _resolve_disk_path(path_part: str) -> Path:
        """Ensure disk path exists and return it."""
        disk_path = Path(path_part)
        if not disk_path.is_absolute():
            disk_path = (Path.cwd() / disk_path).resolve()
        else:
            disk_path = disk_path.resolve()
        disk_path.mkdir(parents=True, exist_ok=True)
        return disk_path

    @staticmethod
    def _disk_path_from_dsn(dsn: str) -> Path:
        """Extract disk path from DSN without creating directories."""
        if not dsn.startswith("builtin://"):
            msg = f"Unsupported DSN format: {dsn}"
            raise ValueError(msg)
        path_part = dsn[len("builtin://") :]
        if not path_part:
            msg = "Disk DSN must include path segment after 'builtin://'."
            raise ValueError(msg)
        disk_path = Path(path_part)
        if not disk_path.is_absolute():
            disk_path = (Path.cwd() / disk_path).resolve()
        else:
            disk_path = disk_path.resolve()
        return disk_path

    @staticmethod
    def _copy_disk_dump(source: Path, target: Path) -> None:
        """Copy saved disk data into target path."""
        if not source.exists():
            msg = f"Saved disk data not found at {source}"
            raise ValueError(msg)
        if target.exists():
            shutil.rmtree(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, target)

    def init_rx(self) -> None:
        """Initialize Reindexer database connection and create namespace."""
        self._database: RxConnector = RxConnector(**self._rx_connector_config)
        # try:
        #    self._database.namespace_drop(self._rx_namespace)
        # except Exception:
        #    pass
        self._database.namespace_open(self._rx_namespace)
        for index_definitions in self._rx_index_definitions:
            self._database.index_add(self._rx_namespace, index_definitions)

    @classmethod
    def from_texts(
        cls: Type[ReindexerVectorStore],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> ReindexerVectorStore:
        """Create a ReindexerVectorStore from a list of texts.

        Args:
            texts: List of texts to add to the vector store.
            embedding: Embedding function to use.
            metadatas: Optional list of metadata dicts to associate with texts.
            **kwargs: Additional arguments to pass to the constructor.

        Returns:
            ReindexerVectorStore instance.
        """
        store = cls(embedding=embedding, **kwargs)
        store.add_texts(texts=texts, metadatas=metadatas)
        return store

    @property
    def embeddings(self) -> Embeddings:
        """Return the embedding function."""
        return self.embedding

    def add_documents(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add documents to the store.

        Args:
            documents: List of Document objects to add.
            ids: Optional list of IDs for the documents.
            **kwargs: Additional arguments (not used).

        Returns:
            List of IDs of the added documents.
        """
        texts = [doc.page_content for doc in documents]
        vectors = self.embedding.embed_documents(texts)

        if ids and len(ids) != len(texts):
            msg = (
                f"ids must be the same length as texts. "
                f"Got {len(ids)} ids and {len(texts)} texts."
            )
            raise ValueError(msg)

        id_iterator: Iterator[Optional[str]] = (
            iter(ids) if ids else iter(doc.id for doc in documents)
        )

        ids_ = []

        for doc, vector in zip(documents, vectors):
            doc_id = next(id_iterator)
            doc_id_ = doc_id if doc_id else str(uuid.uuid4())
            ids_.append(doc_id_)
            item = {
                "id": doc_id_,
                "vector": vector,
                "text": doc.page_content,
                "metadata": doc.metadata,
            }
            self._database.item_upsert(self._rx_namespace, item)

        return ids_

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> None:
        """Delete documents by their IDs.

        Args:
            ids: List of document IDs to delete.
            **kwargs: Additional arguments (not used).
        """
        if ids:
            for _id in ids:
                (
                    self._database.new_query(self._rx_namespace)
                    .where("id", CondType.CondEq, _id)
                    .delete()
                )

    def get_by_ids(self, ids: Sequence[str], /) -> list[Document]:
        """Get documents by their ids.

        Args:
            ids: The ids of the documents to get.

        Returns:
            A list of Document objects.
        """
        documents = []
        results = (
            self._database.new_query(self._rx_namespace)
            .where("id", CondType.CondSet, ids)
            .select_fields()
            .must_execute()
        )
        for result in results:
            documents.append(
                Document(
                    id=result["id"],
                    page_content=result["text"],
                    metadata=result["metadata"],
                )
            )
        return documents

    def _apply_metadata_filter(
        self,
        query: Any,
        filter: Optional[Union[Dict[str, Any], Callable[[Document], bool]]],
    ) -> Any:
        """Apply metadata filter to Reindexer query.

        Args:
            query: Reindexer query object.
            filter: Filter to apply. Can be a dict of metadata key-value pairs
                or a callable function.

        Returns:
            Query object with filter applied.
        """
        if filter is None:
            return query

        if isinstance(filter, dict):
            # Apply metadata filters using Reindexer where conditions
            for key, value in filter.items():
                if isinstance(value, (list, tuple)):
                    # For list values, use CondSet
                    query = query.where(
                        f"metadata.{key}", CondType.CondSet, list(value)
                    )
                else:
                    # For single values, use CondEq
                    query = query.where(f"metadata.{key}", CondType.CondEq, value)
        elif callable(filter):
            # For callable filters, we'll apply them after fetching results
            # This is less efficient but supports complex filtering logic
            pass

        return query

    def _similarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Union[Dict[str, Any], Callable[[Document], bool]]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float, List[float]]]:
        """Search for similar documents by vector with scores.

        Args:
            embedding: Query embedding vector.
            k: Number of results to return.
            filter: Optional filter. Can be a dict of metadata key-value pairs
                or a callable function that takes a Document and returns bool.
            **kwargs: Additional arguments (not used).

        Returns:
            List of tuples (Document, score, embedding_vector).
        """
        param = IndexSearchParamHnsw(k=k, ef=math.ceil(k * 1.5))

        query = (
            self._database.new_query(self._rx_namespace)
            .where_knn("vector", embedding, param)
            .select_fields("vectors()")
            .with_rank()
            .sort(index="rank()", desc=True)
        )

        # Apply metadata filter if it's a dict
        if isinstance(filter, dict):
            query = self._apply_metadata_filter(query, filter)

        query_results = query.must_execute(timedelta(seconds=2))

        results = [
            (
                # Document
                Document(
                    id=query_result["id"],
                    page_content=query_result["text"],
                    metadata=query_result["metadata"],
                ),
                # Score
                float(query_result["rank()"]),
                # Embedding vector
                query_result["vector"],
            )
            for query_result in query_results
        ]

        # Apply callable filter if provided
        if callable(filter):
            results = [
                (doc, score, vector) for doc, score, vector in results if filter(doc)
            ]

        return results

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """Search for similar documents by query text.

        Args:
            query: Query text.
            k: Number of results to return.
            **kwargs: Additional arguments passed to
                _similarity_search_with_score_by_vector.

        Returns:
            List of Document objects.
        """
        embedding = self.embedding.embed_query(query)
        return [
            doc
            for doc, _, _ in self._similarity_search_with_score_by_vector(
                embedding=embedding, k=k, **kwargs
            )
        ]

    def similarity_search_with_score(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        """Search for similar documents by query text with scores.

        Args:
            query: Query text.
            k: Number of results to return.
            **kwargs: Additional arguments passed to
                _similarity_search_with_score_by_vector.

        Returns:
            List of tuples (Document, score).
        """
        embedding = self.embedding.embed_query(query)
        return [
            (doc, similarity)
            for doc, similarity, _ in self._similarity_search_with_score_by_vector(
                embedding=embedding, k=k, **kwargs
            )
        ]

    def similarity_search_by_vector(
        self, embedding: List[float], k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """Search for similar documents by embedding vector.

        Args:
            embedding: Query embedding vector.
            k: Number of results to return.
            **kwargs: Additional arguments passed to
                _similarity_search_with_score_by_vector.

        Returns:
            List of Document objects.
        """
        docs = self._similarity_search_with_score_by_vector(
            embedding=embedding, k=k, **kwargs
        )
        return [doc for doc, _, _ in docs]

    # Async methods
    async def aadd_documents(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Async add documents to the store.

        Args:
            documents: List of Document objects to add.
            ids: Optional list of IDs for the documents.
            **kwargs: Additional arguments (not used).

        Returns:
            List of IDs of the added documents.
        """
        return await run_in_executor(None, self.add_documents, documents, ids, **kwargs)

    async def adelete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> None:
        """Async delete documents by their IDs.

        Args:
            ids: List of document IDs to delete.
            **kwargs: Additional arguments (not used).
        """
        return await run_in_executor(None, self.delete, ids, **kwargs)

    async def aget_by_ids(self, ids: Sequence[str], /) -> list[Document]:
        """Async get documents by their ids.

        Args:
            ids: The ids of the documents to get.

        Returns:
            A list of Document objects.
        """
        return await run_in_executor(None, self.get_by_ids, ids)

    async def asimilarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """Async search for similar documents by query text.

        Args:
            query: Query text.
            k: Number of results to return.
            **kwargs: Additional arguments passed to similarity_search.

        Returns:
            List of Document objects.
        """
        return await run_in_executor(None, self.similarity_search, query, k, **kwargs)

    async def asimilarity_search_with_score(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        """Async search for similar documents by query text with scores.

        Args:
            query: Query text.
            k: Number of results to return.
            **kwargs: Additional arguments passed to similarity_search_with_score.

        Returns:
            List of tuples (Document, score).
        """
        return await run_in_executor(
            None, self.similarity_search_with_score, query, k, **kwargs
        )

    async def asimilarity_search_by_vector(
        self, embedding: List[float], k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """Async search for similar documents by embedding vector.

        Args:
            embedding: Query embedding vector.
            k: Number of results to return.
            **kwargs: Additional arguments passed to similarity_search_by_vector.

        Returns:
            List of Document objects.
        """
        return await run_in_executor(
            None, self.similarity_search_by_vector, embedding, k, **kwargs
        )

    # MMR methods
    def max_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Union[Dict[str, Any], Callable[[Document], bool]]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                of diversity among the results with 0 corresponding
                to maximum diversity and 1 to minimum diversity.
            filter: Optional filter. Can be a dict of metadata key-value pairs
                or a callable function.
            **kwargs: Additional arguments (not used).

        Returns:
            List of Document objects selected by maximal marginal relevance.
        """
        if not _HAS_NUMPY:
            msg = (
                "numpy must be installed to use max_marginal_relevance_search. "
                "Please install numpy with `pip install numpy`."
            )
            raise ImportError(msg)

        prefetch_hits = self._similarity_search_with_score_by_vector(
            embedding=embedding, k=fetch_k, filter=filter, **kwargs
        )

        if not prefetch_hits:
            return []

        mmr_chosen_indices = maximal_marginal_relevance(
            np.array(embedding, dtype=np.float32),
            [vector for _, _, vector in prefetch_hits],
            k=k,
            lambda_mult=lambda_mult,
        )
        return [prefetch_hits[idx][0] for idx in mmr_chosen_indices]

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Union[Dict[str, Any], Callable[[Document], bool]]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                of diversity among the results with 0 corresponding
                to maximum diversity and 1 to minimum diversity.
            filter: Optional filter. Can be a dict of metadata key-value pairs
                or a callable function.
            **kwargs: Additional arguments (not used).

        Returns:
            List of Document objects selected by maximal marginal relevance.
        """
        embedding_vector = self.embedding.embed_query(query)
        return self.max_marginal_relevance_search_by_vector(
            embedding=embedding_vector,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
            **kwargs,
        )

    async def amax_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Union[Dict[str, Any], Callable[[Document], bool]]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Async return docs selected using the maximal marginal relevance.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                of diversity among the results.
            filter: Optional filter. Can be a dict of metadata key-value pairs
                or a callable function.
            **kwargs: Additional arguments (not used).

        Returns:
            List of Document objects selected by maximal marginal relevance.
        """
        return await run_in_executor(
            None,
            self.max_marginal_relevance_search,
            query,
            k,
            fetch_k,
            lambda_mult,
            filter,
            **kwargs,
        )

    async def amax_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Union[Dict[str, Any], Callable[[Document], bool]]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Async return docs selected using the maximal marginal relevance.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                of diversity among the results.
            filter: Optional filter. Can be a dict of metadata key-value pairs
                or a callable function.
            **kwargs: Additional arguments (not used).

        Returns:
            List of Document objects selected by maximal marginal relevance.
        """
        return await run_in_executor(
            None,
            self.max_marginal_relevance_search_by_vector,
            embedding,
            k,
            fetch_k,
            lambda_mult,
            filter,
            **kwargs,
        )

    # Save/Load methods
    def _export_all_items(self) -> List[Dict[str, Any]]:
        """Export all items from the current namespace."""
        query = self._database.new_query(self._rx_namespace).select_fields(
            "id", "text", "metadata", "vector"
        )
        results = query.must_execute()
        exported: List[Dict[str, Any]] = []
        for item in results:
            exported.append(
                {
                    "id": item["id"],
                    "text": item["text"],
                    "metadata": item.get("metadata"),
                    "vector": item.get("vector"),
                }
            )
        return exported

    def _restore_memory_items(self, items: List[Dict[str, Any]]) -> None:
        """Restore items into the namespace from exported data."""
        for item in items:
            payload = {
                "id": item["id"],
                "text": item["text"],
                "metadata": item.get("metadata"),
                "vector": item.get("vector"),
            }
            self._database.item_upsert(self._rx_namespace, payload)

    def save_local(self, path: Union[str, Path]) -> None:
        """Save the vector store configuration to a local directory.

        Note: This saves the configuration, not the actual data.
        The data remains in the Reindexer database.

        Args:
            path: Path to the directory where the configuration will be saved.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        connector_config = dict(self._rx_connector_config)
        connector_config["dsn"] = self._config_dsn_value
        config = {
            "rx_connector_config": connector_config,
            "rx_namespace": self._rx_namespace,
            "rx_index_definitions": self._rx_index_definitions,
            "storage_mode": self._storage_mode,
            "auto_disk_path": self._auto_disk_path,
            "auto_disk_root_dir": self._auto_disk_root_dir,
        }

        config_path = path / "reindexer_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        items = self._export_all_items()
        data_path = path / MEMORY_DUMP_FILENAME
        with open(data_path, "w") as f:
            json.dump(items, f, indent=2)

        if self._storage_mode == "disk":
            if self._disk_path is None:
                msg = "Disk path is not configured for disk storage mode."
                raise ValueError(msg)
            dump_dir = path / DISK_DUMP_SUBDIR
            if dump_dir.exists():
                shutil.rmtree(dump_dir)
            shutil.copytree(self._disk_path, dump_dir)

    @classmethod
    def load_local(
        cls,
        path: Union[str, Path],
        embedding: Embeddings,
        **kwargs: Any,
    ) -> "ReindexerVectorStore":
        """Load a vector store from a local directory.

        Args:
            path: Path to the directory containing the saved configuration.
            embedding: Embedding function to use.
            **kwargs: Additional arguments to pass to the constructor.

        Returns:
            ReindexerVectorStore instance.
        """
        path = Path(path)
        config_path = path / "reindexer_config.json"

        if not config_path.exists():
            raise ValueError(f"Configuration file not found at {config_path}")

        with open(config_path, "r") as f:
            config = json.load(f)

        storage_mode: str = config.get("storage_mode", "memory")
        auto_disk_path: bool = config.get("auto_disk_path", False)
        auto_disk_root_dir: str = config.get(
            "auto_disk_root_dir", DEFAULT_AUTOMATIC_DISK_DIR
        )

        # Use kwargs values if provided, otherwise use config values
        stored_connector_config = config.get("rx_connector_config", {}) or {}
        rx_connector_config = kwargs.pop("rx_connector_config", None)
        if rx_connector_config is None:
            rx_connector_config = dict(stored_connector_config)
        else:
            rx_connector_config = dict(rx_connector_config)
        rx_namespace = kwargs.pop(
            "rx_namespace", config.get("rx_namespace", "langchain")
        )
        rx_index_definitions = kwargs.pop(
            "rx_index_definitions", config.get("rx_index_definitions")
        )

        memory_dump_file = path / MEMORY_DUMP_FILENAME
        disk_dump_dir = path / DISK_DUMP_SUBDIR

        # Validate disk storage path before initialization
        if storage_mode == "disk":
            if not disk_dump_dir.exists():
                raise ValueError(f"Saved disk data not found at {disk_dump_dir}")

            dsn = rx_connector_config.get("dsn")
            if not dsn and auto_disk_path:
                dsn = "builtin:///"
                rx_connector_config["dsn"] = dsn

            if not dsn:
                dsn = stored_connector_config.get("dsn")
                rx_connector_config["dsn"] = dsn

            if dsn is None:
                raise ValueError("DSN must be provided to restore disk storage.")

            if dsn == "builtin:///":
                target_path = cls._automatic_disk_path(
                    rx_namespace,
                    directory_name=auto_disk_root_dir,
                )
            else:
                target_path = cls._disk_path_from_dsn(dsn)
            cls._copy_disk_dump(disk_dump_dir, target_path)
            rx_connector_config["dsn"] = (
                "builtin:///"
                if dsn == "builtin:///"
                else f"builtin://{target_path.as_posix()}"
            )

        store = cls(
            embedding=embedding,
            rx_connector_config=rx_connector_config,
            rx_namespace=rx_namespace,
            rx_index_definitions=rx_index_definitions,
            **kwargs,
        )

        if not memory_dump_file.exists():
            if storage_mode == "memory":
                raise ValueError(f"Saved memory dump not found at {memory_dump_file}")
        else:
            with open(memory_dump_file, "r") as f:
                items = json.load(f)
            store._restore_memory_items(items)

        if storage_mode == "disk":
            store._auto_disk_root_dir = auto_disk_root_dir

        return store
