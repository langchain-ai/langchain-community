import asyncio
from typing import Iterable, Any, Sequence

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from surrealdb import RecordID

from langchain_community.vectorstores.utils import maximal_marginal_relevance

DEFAULT_K = 4  # Number of Documents to return.


class SurrealDBStore(VectorStore):
    """
    SurrealDB as Vector Store.

    To use, you should have the ``surrealdb`` python package installed.

    Args:
        embedding_function: Embedding function to use.
        dburl: SurrealDB connection url
        ns: surrealdb namespace for the vector store. (default: "langchain")
        db: surrealdb database for the vector store. (default: "database")
        collection: surrealdb collection for the vector store.
            (default: "documents")

        (optional) db_user and db_pass: surrealdb credentials

    Example:
        .. code-block:: python

            from langchain_community.vectorstores.surrealdb import SurrealDBStore
            from langchain_community.embeddings import HuggingFaceEmbeddings

            model_name = "sentence-transformers/all-mpnet-base-v2"
            embedding_function = HuggingFaceEmbeddings(model_name=model_name)
            dburl = "ws://localhost:8000/rpc"
            ns = "langchain"
            db = "docstore"
            collection = "documents"
            db_user = "root"
            db_pass = "root"

            sdb = SurrealDBStore.from_texts(
                    texts=texts,
                    embedding=embedding_function,
                    dburl,
                    ns, db, collection,
                    db_user=db_user, db_pass=db_pass)
    """

    def __init__(
            self,
            embedding_function: Embeddings,
            **kwargs: Any,
    ) -> None:
        try:
            from surrealdb import Surreal
        except ImportError as e:
            raise ImportError(
                """Cannot import from surrealdb.
                please install with `pip install surrealdb`."""
            ) from e

        self.dburl = kwargs.pop("dburl", "ws://localhost:8000/rpc")

        if self.dburl[0:2] == "ws":
            self.sdb = Surreal(self.dburl)
        else:
            raise ValueError("Only websocket connections are supported at this time.")

        self.ns = kwargs.pop("ns", "langchain")
        self.db = kwargs.pop("db", "database")
        self.collection = kwargs.pop("collection", "documents")
        self.embedding_function = embedding_function
        self.kwargs = kwargs

    def initialize(self) -> None:
        """
        Initialize connection to surrealdb database
        and authenticate if credentials are provided
        """
        # await self.sdb.connect()
        if "db_user" in self.kwargs and "db_pass" in self.kwargs:
            user = self.kwargs.get("db_user")
            password = self.kwargs.get("db_pass")
            self.sdb.signin({"username": user, "password": password})
        self.sdb.use(self.ns, self.db)

    @property
    def embeddings(self) -> Embeddings | None:
        return (
            self.embedding_function
            if isinstance(self.embedding_function, Embeddings)
            else None
        )

    async def aadd_texts(
            self,
            texts: Iterable[str],
            metadatas: list[dict] | None = None,
            **kwargs: Any,
    ) -> list[str]:
        """Add list of text along with embeddings to the vector store asynchronously

        Args:
            texts (Iterable[str]): collection of text to add to the database

        Returns:
            List of ids for the newly inserted documents
        """
        embeddings = self.embedding_function.embed_documents(list(texts))
        ids = []
        for idx, text in enumerate(texts):
            data = {"text": text, "embedding": embeddings[idx], "metadata": {}}
            if metadatas is not None and idx < len(metadatas):
                data["metadata"] = metadatas[idx]  # type: ignore[assignment]
            preferred_id = data["metadata"].get("id")
            kwargs_ids = kwargs.get("ids", [])
            if kwargs_ids and idx < len(kwargs_ids) and kwargs_ids[idx] is not None:
                preferred_id = kwargs_ids[idx]

            # TODO: consider using insert and make a single bulk insert call, but not if we have IDs in the metadata or
            #       in kwargs, because only with `create` we can specify IDs
            record_id = RecordID(self.collection, preferred_id) if preferred_id is not None else self.collection
            record = self.sdb.upsert(record_id, data)
            ids.append(record.get("id").id)
        return ids

    def add_texts(
            self,
            texts: Iterable[str],
            metadatas: list[dict] | None = None,
            **kwargs: Any,
    ) -> list[str]:
        """Add list of text along with embeddings to the vector store

        Args:
            texts (Iterable[str]): collection of text to add to the database

        Returns:
            List of ids for the newly inserted documents
        """

        async def _add_texts(
                texts: Iterable[str],
                metadatas: list[dict] | None = None,
                **kwargs: Any,
        ) -> list[str]:
            self.initialize()
            return await self.aadd_texts(texts, metadatas, **kwargs)

        return asyncio.run(_add_texts(texts, metadatas, **kwargs))

    async def adelete(
            self,
            ids: list[str] | None = None,
            **kwargs: Any,
    ) -> bool | None:
        """Delete by document ID asynchronously.

        Args:
            ids: List of ids to delete.
            **kwargs: Other keyword arguments that subclasses might use.

        Returns:
            bool | None: True if deletion is successful,
            False otherwise.
        """

        if ids is None:
            self.sdb.delete(self.collection)
            return True
        else:
            if isinstance(ids, str):
                self.sdb.delete(ids)
                return True
            else:
                if isinstance(ids, list) and len(ids) > 0:
                    _ = [self.sdb.delete(id) for id in ids]
                    return True
        return False

    def delete(
            self,
            ids: list[str] | None = None,
            **kwargs: Any,
    ) -> bool | None:
        """Delete by document ID.

        Args:
            ids: List of ids to delete.
            **kwargs: Other keyword arguments that subclasses might use.

        Returns:
            bool | None: True if deletion is successful,
            False otherwise.
        """

        async def _delete(ids: list[str] | None, **kwargs: Any) -> bool | None:
            self.initialize()
            return await self.adelete(ids=ids, **kwargs)

        return asyncio.run(_delete(ids, **kwargs))

    def get_by_ids(self, ids: Sequence[str], /) -> list[Document]:
        """Get documents by their IDs.

        The returned documents are expected to have the ID field set to the ID of the
        document in the vector store.

        Fewer documents may be returned than requested if some IDs are not found or
        if there are duplicated IDs.

        Users should not assume that the order of the returned documents matches
        the order of the input IDs. Instead, users should rely on the ID field of the
        returned documents.

        This method should **NOT** raise exceptions if no documents are found for
        some IDs.

        Args:
            ids: List of ids to retrieve.

        Returns:
            List of Documents.
        """
        results = self.sdb.query(
            "SELECT * FROM type::table($collection) WHERE id IN array::combine([$collection], $ids).map(|$v| type::thing($v[0], $v[1]))",
            {"collection": self.collection, "ids": ids})
        return [Document(
            page_content=x.get("text"),
            metadata=x.get("metadata", {}),
            id=x.get("metadata").get("id"),
        ) for x in results]

    async def _asimilarity_search_by_vector_with_score(
            self,
            embedding: list[float],
            k: int = DEFAULT_K,
            *,
            filter: dict[str, str] | None = None,
            **kwargs: Any,
    ) -> list[tuple[Document, float, Any]]:
        """Run similarity search for query embedding asynchronously
        and return documents and scores

        Args:
            embedding (list[float]): Query embedding.
            k (int): Number of results to return. Defaults to 4.
            filter (dict[str, str] | None): Filter by metadata. Defaults to None.

        Returns:
            List of Documents most similar along with scores
        """
        args = {
            "collection": self.collection,
            "embedding": embedding,
            "k": k,
            "score_threshold": kwargs.get("score_threshold", 0),
        }

        # build additional filter criteria
        custom_filter = ""
        if filter:
            for key in filter:
                # check value type
                if type(filter[key]) in [str, bool]:
                    filter_value = f"'{filter[key]}'"
                else:
                    filter_value = f"{filter[key]}"

                custom_filter += f"and metadata.{key} = {filter_value} "

        query = f"""
        select
            id,
            text,
            metadata,
            embedding,
            vector::similarity::cosine(embedding, $embedding) as similarity
        from ⟨{args["collection"]}⟩
        where vector::similarity::cosine(embedding, $embedding) >= $score_threshold
          {custom_filter}
        order by similarity desc LIMIT $k;
        """
        results = self.sdb.query(query, args)

        return [
            (
                Document(
                    page_content=doc["text"],
                    # metadata={"id": doc["id"], **(doc.get("metadata") or {})},
                    metadata=doc.get("metadata", {}),
                    # id=doc["id"].id
                    id=doc.get("metadata").get("id"),
                ),
                doc["similarity"],
                doc["embedding"],
            )
            for doc in results
        ]

    async def asimilarity_search_with_relevance_scores(
            self,
            query: str,
            k: int = DEFAULT_K,
            *,
            filter: dict[str, str] | None = None,
            **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Run similarity search asynchronously and return relevance scores

        Args:
            query (str): Query
            k (int): Number of results to return. Defaults to 4.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List of Documents most similar along with relevance scores
        """
        query_embedding = self.embedding_function.embed_query(query)
        return [
            (document, similarity)
            for document, similarity, _ in (
                await self._asimilarity_search_by_vector_with_score(
                    query_embedding, k, filter=filter, **kwargs
                )
            )
        ]

    def similarity_search_with_relevance_scores(
            self,
            query: str,
            k: int = DEFAULT_K,
            *,
            filter: dict[str, str] | None = None,
            **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Run similarity search synchronously and return relevance scores

        Args:
            query (str): Query
            k (int): Number of results to return. Defaults to 4.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List of Documents most similar along with relevance scores
        """

        async def _similarity_search_with_relevance_scores() -> list[
            tuple[Document, float]
        ]:
            self.initialize()
            return await self.asimilarity_search_with_relevance_scores(
                query, k, filter=filter, **kwargs
            )

        return asyncio.run(_similarity_search_with_relevance_scores())

    async def asimilarity_search_with_score(
            self,
            query: str,
            k: int = DEFAULT_K,
            *,
            filter: dict[str, str] | None = None,
            **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Run similarity search asynchronously and return distance scores

        Args:
            query (str): Query
            k (int): Number of results to return. Defaults to 4.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List of Documents most similar along with relevance distance scores
        """
        query_embedding = self.embedding_function.embed_query(query)
        return [
            (document, similarity)
            for document, similarity, _ in (
                await self._asimilarity_search_by_vector_with_score(
                    query_embedding, k, filter=filter, **kwargs
                )
            )
        ]

    def similarity_search_with_score(
            self,
            query: str,
            k: int = DEFAULT_K,
            *,
            filter: dict[str, str] | None = None,
            **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Run similarity search synchronously and return distance scores

        Args:
            query (str): Query
            k (int): Number of results to return. Defaults to 4.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List of Documents most similar along with relevance distance scores
        """

        async def _similarity_search_with_score() -> list[tuple[Document, float]]:
            self.initialize()
            return await self.asimilarity_search_with_score(
                query, k, filter=filter, **kwargs
            )

        return asyncio.run(_similarity_search_with_score())

    async def asimilarity_search_by_vector(
            self,
            embedding: list[float],
            k: int = DEFAULT_K,
            *,
            filter: dict[str, str] | None = None,
            **kwargs: Any,
    ) -> list[Document]:
        """Run similarity search on query embedding asynchronously

        Args:
            embedding (list[float]): Query embedding
            k (int): Number of results to return. Defaults to 4.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List of Documents most similar to the query
        """
        return [
            document
            for document, _, _ in await self._asimilarity_search_by_vector_with_score(
                embedding, k, filter=filter, **kwargs
            )
        ]

    def similarity_search_by_vector(
            self,
            embedding: list[float],
            k: int = DEFAULT_K,
            *,
            filter: dict[str, str] | None = None,
            **kwargs: Any,
    ) -> list[Document]:
        """Run similarity search on query embedding

        Args:
            embedding (list[float]): Query embedding
            k (int): Number of results to return. Defaults to 4.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List of Documents most similar to the query
        """

        async def _similarity_search_by_vector() -> list[Document]:
            self.initialize()
            return await self.asimilarity_search_by_vector(
                embedding, k, filter=filter, **kwargs
            )

        return asyncio.run(_similarity_search_by_vector())

    async def asimilarity_search(
            self,
            query: str,
            k: int = DEFAULT_K,
            *,
            filter: dict[str, str] | None = None,
            **kwargs: Any,
    ) -> list[Document]:
        """Run similarity search on query asynchronously

        Args:
            query (str): Query
            k (int): Number of results to return. Defaults to 4.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List of Documents most similar to the query
        """
        query_embedding = self.embedding_function.embed_query(query)
        return await self.asimilarity_search_by_vector(
            query_embedding, k, filter=filter, **kwargs
        )

    def similarity_search(
            self,
            query: str,
            k: int = DEFAULT_K,
            *,
            filter: dict[str, str] | None = None,
            **kwargs: Any,
    ) -> list[Document]:
        """Run similarity search on query

        Args:
            query (str): Query
            k (int): Number of results to return. Defaults to 4.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List of Documents most similar to the query
        """

        async def _similarity_search() -> list[Document]:
            self.initialize()
            return await self.asimilarity_search(query, k, filter=filter, **kwargs)

        return asyncio.run(_similarity_search())

    async def amax_marginal_relevance_search_by_vector(
            self,
            embedding: list[float],
            k: int = DEFAULT_K,
            fetch_k: int = 20,
            lambda_mult: float = 0.5,
            *,
            filter: dict[str, str] | None = None,
            **kwargs: Any,
    ) -> list[Document]:
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
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List of Documents selected by maximal marginal relevance.
        """

        result = await self._asimilarity_search_by_vector_with_score(
            embedding, fetch_k, filter=filter, **kwargs
        )

        # extract only document from result
        docs = [sub[0] for sub in result]
        # extract only embedding from result
        embeddings = [sub[-1] for sub in result]

        mmr_selected = maximal_marginal_relevance(
            np.array(embedding, dtype=np.float32),
            embeddings,
            k=k,
            lambda_mult=lambda_mult,
        )

        return [docs[i] for i in mmr_selected]

    def max_marginal_relevance_search_by_vector(
            self,
            embedding: list[float],
            k: int = DEFAULT_K,
            fetch_k: int = 20,
            lambda_mult: float = 0.5,
            *,
            filter: dict[str, str] | None = None,
            **kwargs: Any,
    ) -> list[Document]:
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
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List of Documents selected by maximal marginal relevance.
        """

        async def _max_marginal_relevance_search_by_vector() -> list[Document]:
            self.initialize()
            return await self.amax_marginal_relevance_search_by_vector(
                embedding, k, fetch_k, lambda_mult, filter=filter, **kwargs
            )

        return asyncio.run(_max_marginal_relevance_search_by_vector())

    async def amax_marginal_relevance_search(
            self,
            query: str,
            k: int = 4,
            fetch_k: int = 20,
            lambda_mult: float = 0.5,
            *,
            filter: dict[str, str] | None = None,
            **kwargs: Any,
    ) -> list[Document]:
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
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List of Documents selected by maximal marginal relevance.
        """

        embedding = self.embedding_function.embed_query(query)
        docs = await self.amax_marginal_relevance_search_by_vector(
            embedding, k, fetch_k, lambda_mult, filter=filter, **kwargs
        )
        return docs

    def max_marginal_relevance_search(
            self,
            query: str,
            k: int = DEFAULT_K,
            fetch_k: int = 20,
            lambda_mult: float = 0.5,
            *,
            filter: dict[str, str] | None = None,
            **kwargs: Any,
    ) -> list[Document]:
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
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List of Documents selected by maximal marginal relevance.
        """

        async def _max_marginal_relevance_search() -> list[Document]:
            self.initialize()
            return await self.amax_marginal_relevance_search(
                query, k, fetch_k, lambda_mult, filter=filter, **kwargs
            )

        return asyncio.run(_max_marginal_relevance_search())

    @classmethod
    async def afrom_texts(
            cls,
            texts: list[str],
            embedding: Embeddings,
            metadatas: list[dict] | None = None,
            **kwargs: Any,
    ) -> "SurrealDBStore":
        """Create SurrealDBStore from list of text asynchronously

        Args:
            texts (list[str]): list of text to vectorize and store
            embedding (Optional[Embeddings]): Embedding function.
            dburl (str): SurrealDB connection url
                (default: "ws://localhost:8000/rpc")
            ns (str): surrealdb namespace for the vector store.
                (default: "langchain")
            db (str): surrealdb database for the vector store.
                (default: "database")
            collection (str): surrealdb collection for the vector store.
                (default: "documents")

            (optional) db_user and db_pass: surrealdb credentials

        Returns:
            SurrealDBStore object initialized and ready for use."""

        sdb = cls(embedding, **kwargs)
        sdb.initialize()
        await sdb.aadd_texts(texts, metadatas, **kwargs)
        return sdb

    @classmethod
    def from_texts(
            cls,
            texts: list[str],
            embedding: Embeddings,
            metadatas: list[dict] | None = None,
            **kwargs: Any,
    ) -> "SurrealDBStore":
        """Create SurrealDBStore from list of text

        Args:
            texts (list[str]): list of text to vectorize and store
            embedding (Optional[Embeddings]): Embedding function.
            dburl (str): SurrealDB connection url
            ns (str): surrealdb namespace for the vector store.
                (default: "langchain")
            db (str): surrealdb database for the vector store.
                (default: "database")
            collection (str): surrealdb collection for the vector store.
                (default: "documents")

            (optional) db_user and db_pass: surrealdb credentials

        Returns:
            SurrealDBStore object initialized and ready for use."""
        sdb = asyncio.run(cls.afrom_texts(texts, embedding, metadatas, **kwargs))
        return sdb
