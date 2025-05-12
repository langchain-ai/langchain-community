from __future__ import annotations

import asyncio
from dataclasses import KW_ONLY, dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterable,
    Optional,
    Sequence,
    Union,
)

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from surrealdb import (
    AsyncHttpSurrealConnection,
    AsyncWsSurrealConnection,
    BlockingHttpSurrealConnection,
    BlockingWsSurrealConnection,
    RecordID,
)

from langchain_community.vectorstores.utils import maximal_marginal_relevance

if TYPE_CHECKING:
    from surrealdb import AsyncSurreal, Surreal

DEFAULT_K = 4  # Number of Documents to return.

type SurrealConnection = Union[
    BlockingWsSurrealConnection, BlockingHttpSurrealConnection
]
type SurrealAsyncConnection = Union[
    AsyncWsSurrealConnection, AsyncHttpSurrealConnection
]

GET_BY_ID_QUERY = """
    SELECT *
    FROM type::table($table)
    WHERE id IN array::combine([$table], $ids)
        .map(|$v| type::thing($v[0], $v[1]))
"""

# # Development commands:
#
# ```sh
# surreal start -u root -p root -l debug
# make integration_tests TEST_FILE=tests/integration_tests/vectorstores/test_surrealdb.py  # noqa: E501
# make format
# make lint
# ```


@dataclass
class SurrealDocument:
    _: KW_ONLY
    id: RecordID = field(hash=False)
    text: str
    embedding: list[float]
    similarity: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def into(self) -> Document:
        return Document(
            id=self.id.id,
            page_content=self.text,
            metadata=self.metadata,
        )


class SurrealDBStore(VectorStore):
    """
    SurrealDB as Vector Store.

    To use, you should have the ``surrealdb`` python package installed.

    Args:
        embedding: The embedding function or model to use for generating embeddings.
        table: SurrealDB table for the vector store (default: "documents").
        connection: SurrealDB connection

    Example:
        .. code-block:: python

            from langchain_community.vectorstores.surrealdb import SurrealDBStore
            from langchain_community.embeddings import HuggingFaceEmbeddings

            model_name = "sentence-transformers/all-mpnet-base-v2"
            embedding = HuggingFaceEmbeddings(model_name=model_name)

            conn = Surreal("ws://localhost:8000/rpc")
            conn.signin({"username": "root", "password": "root"})
            conn.use("langchain", "test")

            connection = SurrealDBStore.from_texts(
                texts=texts,
                embedding=embedding,
                connection=conn
            )
    """

    def __init__(
        self,
        embedding: Embeddings,
        connection: SurrealConnection | None,
        table: str = "documents",
        index_name: str = "documents_vector_index",
        embedding_dimension: int | None = None,
        async_connection: SurrealAsyncConnection | None = None,
    ) -> None:
        self.embedding = embedding
        self.table = table
        self.index_name = index_name
        self.connection = connection
        self.async_connection = async_connection
        if embedding_dimension is not None:
            self.embedding_dimension = embedding_dimension
        else:
            self.embedding_dimension = len(self.embedding.embed_query("foo"))
        self._ensure_index()

    def _ensure_index(self) -> None:
        if self.async_connection is not None:
            self.async_connection.query(f"""
                DEFINE INDEX IF NOT EXISTS {self.index_name}
                    ON TABLE {self.table}
                    FIELDS embedding
                    MTREE DIMENSION {self.embedding_dimension} DIST COSINE TYPE F32
                    CONCURRENTLY;
            """)
        elif self.connection is not None:
            self.connection.query(f"""
                DEFINE INDEX IF NOT EXISTS {self.index_name}
                    ON TABLE {self.table}
                    FIELDS embedding
                    MTREE DIMENSION {self.embedding_dimension} DIST COSINE TYPE F32
                    CONCURRENTLY;
            """)
        else:
            raise ValueError("No connection provided")

    def _build_text_data(
        self,
        text: str,
        embedding: list[float],
        metadata: dict | None,
        with_id: str | None,
    ) -> tuple[RecordID | None, dict]:
        preferred_id = None
        data = {"text": text, "embedding": embedding, "metadata": {}}
        if metadata is not None:
            data["metadata"] = metadata
            preferred_id = metadata.get("id")
        if with_id is not None:
            preferred_id = with_id
        record_id = (
            RecordID(self.table, preferred_id) if preferred_id is not None else None
        )
        return record_id, data

    @staticmethod
    def _parse_documents(ids: Sequence[str], results: list[dict]) -> list[Document]:
        docs = {}
        for x in results:
            doc = SurrealDocument(**x).into()
            docs[doc.id] = doc
        # sort docs in the same order as the passed in IDs
        result: list[Document] = []
        for key in ids:
            d = docs.get(str(key))
            if d is not None:
                result.append(d)
        return result

    def _build_search_query(
        self,
        embedding: list[float],
        k: int,
        score_threshold: float,
        custom_filter: dict[str, str] | None,
    ) -> tuple[str, dict]:
        args = {
            "table": self.table,
            "embedding": embedding,
            "k": k,
            "score_threshold": score_threshold,
        }

        # build additional filter criteria
        custom_filter_str = ""
        if custom_filter:
            for key in custom_filter:
                # check value type
                if type(custom_filter[key]) in [str, bool]:
                    filter_value = f"'{custom_filter[key]}'"
                else:
                    filter_value = f"{custom_filter[key]}"

                custom_filter_str += f"and metadata.{key} = {filter_value} "

        query = f"""
            SELECT
                id,
                text,
                metadata,
                embedding,
                similarity
            FROM (
                SELECT
                    id,
                    text,
                    metadata,
                    embedding,
                    vector::similarity::cosine(embedding, $embedding) as similarity
                FROM type::table($table)
                WHERE embedding <|{k}|> $embedding
                    {custom_filter_str}
            )
            WHERE similarity >= $score_threshold
            ORDER BY similarity DESC
        """

        return query, args

    @staticmethod
    def _parse_results(
        results: list[dict],
    ) -> list[tuple[Document, float, list[float]]]:
        parsed = []
        for raw in results:
            parsed.append(
                (
                    SurrealDocument(**raw).into(),
                    raw["similarity"],
                    raw["embedding"],
                ),
            )
        return parsed

    @staticmethod
    def _filter_documents_from_result(
        search_result: list[tuple[Document, float, list[float]]],
        embedding: list[float],
        k: int = DEFAULT_K,
        lambda_mult: float = 0.5,
    ) -> list[Document]:
        # extract only document from result
        docs = [sub[0] for sub in search_result]
        # extract only embedding from result
        embeddings = [sub[-1] for sub in search_result]

        mmr_selected = maximal_marginal_relevance(
            np.array(embedding, dtype=np.float32),
            embeddings,
            k=k,
            lambda_mult=lambda_mult,
        )

        return [docs[i] for i in mmr_selected]

    # =========================================================================
    # == Extended methods
    # =========================================================================

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict] | None = None,
        *,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        if self.connection is None:
            raise ValueError("No connection provided")
        embeddings = self.embedding.embed_documents(list(texts))
        result_ids = []
        for idx, text in enumerate(texts):
            record_id, data = self._build_text_data(
                text,
                embeddings[idx],
                metadatas[idx] if metadatas is not None else None,
                ids[idx] if ids is not None else None,
            )
            if record_id is not None:
                inserted = self.connection.upsert(record_id, data)
            else:
                inserted = self.connection.insert(self.table, data)
            if isinstance(inserted, list):
                for record in inserted:
                    result_ids.append(record["id"].id)
            else:
                result_ids.append(inserted["id"].id)
        return result_ids

    @property
    def embeddings(self) -> Embeddings | None:
        return self.embedding if isinstance(self.embedding, Embeddings) else None

    def delete(
        self,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> bool | None:
        if self.connection is None:
            raise ValueError("No connection provided")
        try:
            if ids is not None:
                for id in ids:
                    self.connection.delete(RecordID(self.table, id))
            else:
                self.connection.delete(self.table)
        except Exception as _e:
            return False
        return True

    def get_by_ids(self, ids: Sequence[str], /) -> list[Document]:
        if self.connection is None:
            raise ValueError("No connection provided")
        query_results = self.connection.query(
            GET_BY_ID_QUERY,
            {"table": self.table, "ids": ids},
        )
        return self._parse_documents(ids, query_results)

    async def aget_by_ids(self, ids: Sequence[str], /) -> list[Document]:
        if self.async_connection is None:
            raise ValueError("No async connection provided")
        query_results = await self.async_connection.query(
            GET_BY_ID_QUERY,
            {"table": self.table, "ids": ids},
        )
        return self._parse_documents(ids, query_results)

    async def adelete(
        self, ids: Optional[list[str]] = None, **kwargs: Any
    ) -> Optional[bool]:
        if self.async_connection is None:
            raise ValueError("No async connection provided")
        try:
            if ids is not None:
                coroutines = [
                    self.async_connection.delete(RecordID(self.table, id)) for id in ids
                ]
                await asyncio.gather(*coroutines)
            else:
                await self.async_connection.delete(self.table)
        except Exception as _e:
            return False
        return True

    async def aadd_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict] | None = None,
        *,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        if self.async_connection is None:
            raise ValueError("No async connection provided")
        embeddings = self.embedding.embed_documents(list(texts))
        coroutines = []
        for idx, text in enumerate(texts):
            record_id, data = self._build_text_data(
                text,
                embeddings[idx],
                metadatas[idx] if metadatas is not None else None,
                ids[idx] if ids is not None else None,
            )
            if record_id is not None:
                coroutines.append(self.async_connection.upsert(record_id, data))
            else:
                coroutines.append(self.async_connection.insert(self.table, data))
        results = await asyncio.gather(*coroutines)
        result_ids = [x.get("id") for x in results]
        return result_ids

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        *,
        custom_filter: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        query_embedding = self.embedding.embed_query(query)
        return self.similarity_search_by_vector(
            query_embedding, k, custom_filter=custom_filter, **kwargs
        )

    # TODO: implement
    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        raise NotImplementedError

    def aux(
        self,
        embedding: list[float],
        *,
        k: int = DEFAULT_K,
        score_threshold: float = -1,
        custom_filter: dict[str, str] | None = None,
    ) -> list[tuple[Document, float, list[float]]]:
        if self.connection is None:
            raise ValueError("No connection provided")
        query, args = self._build_search_query(
            embedding, k, score_threshold, custom_filter
        )
        results = self.connection.query(query, args)
        return self._parse_results(results)

    def similarity_search_with_score(
        self,
        query: str,
        *,
        k: int = DEFAULT_K,
        score_threshold: float = -1,
        custom_filter: dict[str, str] | None = None,
    ) -> list[tuple[Document, float]]:
        embedding = self.embedding.embed_query(query)
        return [
            (d, s)
            for d, s, _ in self.aux(
                embedding,
                k=k,
                score_threshold=score_threshold,
                custom_filter=custom_filter,
            )
        ]

    def similarity_search_by_vector(
        self,
        embedding: list[float],
        k: int = DEFAULT_K,
        *,
        custom_filter: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        return [
            document
            for document, _, _ in self.aux(
                embedding=embedding, k=k, custom_filter=custom_filter
            )
        ]

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = DEFAULT_K,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        *,
        custom_filter: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        embedding = self.embedding.embed_query(query)
        docs = self.max_marginal_relevance_search_by_vector(
            embedding, k, fetch_k, lambda_mult, custom_filter=custom_filter, **kwargs
        )
        return docs

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: list[float],
        k: int = DEFAULT_K,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        *,
        custom_filter: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        result = self._similarity_search_by_vector_with_score(
            embedding, fetch_k, custom_filter=custom_filter, **kwargs
        )
        return self._filter_documents_from_result(result, embedding, k, lambda_mult)

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: Optional[list[dict]] = None,
        *,
        ids: Optional[list[str]] = None,
        connection: Surreal = None,
        **kwargs: Any,
    ) -> "SurrealDBStore":
        store = SurrealDBStore(embedding, connection)
        store.add_texts(texts, metadatas)
        return store

    @classmethod
    async def afrom_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: Optional[list[dict]] = None,
        *,
        ids: Optional[list[str]] = None,
        connection: AsyncSurreal = None,
        **kwargs: Any,
    ) -> "SurrealDBStore":
        store = SurrealDBStore(embedding, None, async_connection=connection)
        await store.aadd_texts(texts, metadatas)
        return store

    # =========================================================================
    # =========================================================================
    # =========================================================================

    def _similarity_search_by_vector_with_score(
        self,
        embedding: list[float],
        k: int = DEFAULT_K,
        score_threshold: float = -1,
        custom_filter: dict[str, str] | None = None,
    ) -> list[tuple[Document, float, list[float]]]:
        if self.connection is None:
            raise ValueError("No connection provided")
        query, args = self._build_search_query(
            embedding, k, score_threshold, custom_filter
        )
        results = self.connection.query(query, args)
        return self._parse_results(results)

    async def _asimilarity_search_by_vector_with_score(
        self,
        embedding: list[float],
        k: int = DEFAULT_K,
        *,
        custom_filter: dict[str, str] | None = None,
        score_threshold: float = -1,
    ) -> list[tuple[Document, float, list[float]]]:
        if self.async_connection is None:
            raise ValueError("No async connection provided")
        query, args = self._build_search_query(
            embedding, k, score_threshold, custom_filter
        )
        results = await self.async_connection.query(query, args)
        return self._parse_results(results)

    async def asimilarity_search_with_relevance_scores(
        self,
        query: str,
        k: int = DEFAULT_K,
        *,
        custom_filter: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        query_embedding = self.embedding.embed_query(query)
        # TODO: improve using asyncio.gather
        return [
            (document, similarity)
            for document, similarity, _ in (
                await self._asimilarity_search_by_vector_with_score(
                    query_embedding, k, custom_filter=custom_filter, **kwargs
                )
            )
        ]

    def similarity_search_with_relevance_scores(
        self,
        query: str,
        k: int = DEFAULT_K,
        *,
        custom_filter: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        query_embedding = self.embedding.embed_query(query)
        return [
            (document, similarity)
            for document, similarity, _ in (
                self._similarity_search_by_vector_with_score(
                    query_embedding, k, custom_filter=custom_filter, **kwargs
                )
            )
        ]

    async def asimilarity_search_by_vector(
        self,
        embedding: list[float],
        k: int = DEFAULT_K,
        *,
        custom_filter: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        # TODO: improve using asyncio.gather
        return [
            document
            for document, _, _ in await self._asimilarity_search_by_vector_with_score(
                embedding, k, custom_filter=custom_filter, **kwargs
            )
        ]

    async def asimilarity_search(
        self,
        query: str,
        k: int = DEFAULT_K,
        *,
        custom_filter: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        query_embedding = self.embedding.embed_query(query)
        return await self.asimilarity_search_by_vector(
            query_embedding, k, custom_filter=custom_filter, **kwargs
        )

    async def amax_marginal_relevance_search_by_vector(
        self,
        embedding: list[float],
        k: int = DEFAULT_K,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        *,
        custom_filter: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        result = await self._asimilarity_search_by_vector_with_score(
            embedding, fetch_k, custom_filter=custom_filter, **kwargs
        )
        return self._filter_documents_from_result(result, embedding, k, lambda_mult)

    async def amax_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        *,
        custom_filter: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        embedding = self.embedding.embed_query(query)
        docs = await self.amax_marginal_relevance_search_by_vector(
            embedding, k, fetch_k, lambda_mult, custom_filter=custom_filter, **kwargs
        )
        return docs
