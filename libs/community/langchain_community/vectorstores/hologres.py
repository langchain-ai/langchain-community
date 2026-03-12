from __future__ import annotations

import json
import logging
import shlex
import uuid
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Type

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.utils import get_from_dict_or_env
from langchain_core.vectorstores import VectorStore

ADA_TOKEN_COUNT = 1536
_LANGCHAIN_DEFAULT_TABLE_NAME = "langchain_pg_embedding"
_DEFAULT_SCHEMA = "public"
_ID_COLUMN = "langchain_id"
_DOCUMENT_COLUMN = "document"
_EMBEDDING_COLUMN = "embedding"
_METADATA_COLUMN = "metadata"
_HGRAPH_VERSION = 40000


def _extract_metadata_value(metadata: Dict[str, Any], key: str) -> Any:
    """Return a metadata value and normalize nested containers for storage."""
    value = metadata.get(key)
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)) or value is None or isinstance(value, bool):
        return json.dumps(value, sort_keys=True)
    return value


class Hologres(VectorStore):
    """`Hologres` vector store with sdk selected by instance version.

    For instances with version < r4.0.0, this class uses `hologres-vector`.
    For instances with version >= r4.0.0, this class uses `holo-search-sdk`.

    - `connection_string` is a hologres connection string.
    - `embedding_function` any embedding function implementing
        `langchain.embeddings.base.Embeddings` interface.
    - `ndims` is the number of dimensions of the embedding output.
    - `table_name` is the name of the table to store embeddings and data.
        (default: langchain_pg_embedding)
        - NOTE: The table will be created when initializing the store (if not exists)
            So, make sure the user has the right permissions to create tables.
    - `pre_delete_table` if True, will delete the table if it exists.
        (default: False)
        - Useful for testing.
    """

    def __init__(
        self,
        connection_string: str,
        embedding_function: Embeddings,
        ndims: int = ADA_TOKEN_COUNT,
        table_name: str = _LANGCHAIN_DEFAULT_TABLE_NAME,
        pre_delete_table: bool = False,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.connection_string = connection_string
        self.ndims = ndims
        self.table_name = table_name
        self.embedding_function = embedding_function
        self.pre_delete_table = pre_delete_table
        self.logger = logger or logging.getLogger(__name__)
        self.__post_init__()

    def __post_init__(self) -> None:
        """Initialize the store and ensure the table/index exist."""
        self.holo_version = self.get_holo_version(self.connection_string)
        self._use_holo_search_sdk = self.holo_version >= _HGRAPH_VERSION

        if self._use_holo_search_sdk:
            self._init_holo_search_sdk_store()
        else:
            self._init_hologres_vector_store()

    def _init_holo_search_sdk_store(self) -> None:
        try:
            from holo_search_sdk import connect
        except ImportError as exc:
            raise ImportError(
                "holo-search-sdk is required for Hologres instances with "
                "version >= r4.0.0."
            ) from exc

        config = self._parse_connection_string(self.connection_string)

        self.client = connect(
            host=config["host"],
            port=config["port"],
            database=config["database"],
            access_key_id=config["user"],
            access_key_secret=config["password"],
            schema=config["schema"],
        )

        if self.pre_delete_table:
            self.client.drop_table(self.table_name)

        self._ensure_table()
        self.storage = self.client.open_table(self.table_name)
        self._ensure_vector_index()

    def _init_hologres_vector_store(self) -> None:
        try:
            from hologres_vector import HologresVector
        except ImportError as exc:
            raise ImportError(
                "hologres-vector is required for Hologres instances with "
                "version < r4.0.0."
            ) from exc

        self.storage = HologresVector(
            connection_string=self.connection_string,
            ndims=self.ndims,
            table_name=self.table_name,
            table_schema={_DOCUMENT_COLUMN: "text"},
            pre_delete_table=self.pre_delete_table,
            logger=self.logger,
        )

    @property
    def embeddings(self) -> Embeddings:
        return self.embedding_function

    @classmethod
    def __from(
        cls,
        texts: List[str],
        embeddings: List[List[float]],
        embedding_function: Embeddings,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        ndims: int = ADA_TOKEN_COUNT,
        table_name: str = _LANGCHAIN_DEFAULT_TABLE_NAME,
        pre_delete_table: bool = False,
        **kwargs: Any,
    ) -> Hologres:
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]

        if not metadatas:
            metadatas = [{} for _ in texts]

        connection_string = cls.get_connection_string(kwargs)

        store = cls(
            connection_string=connection_string,
            embedding_function=embedding_function,
            ndims=ndims,
            table_name=table_name,
            pre_delete_table=pre_delete_table,
        )

        store.add_embeddings(
            texts=texts, embeddings=embeddings, metadatas=metadatas, ids=ids, **kwargs
        )

        return store

    def _ensure_table(self) -> None:
        from psycopg import sql as psql

        if self.client.check_table_exist(self.table_name):
            return

        create_table_sql = psql.SQL(
            """
            CREATE TABLE IF NOT EXISTS {} (
                {} TEXT PRIMARY KEY,
                {} TEXT,
                {} FLOAT4[] CHECK (
                    array_ndims({}) = 1 AND array_length({}, 1) = {}
                ),
                {} JSON
            );
            """
        ).format(
            psql.Identifier(self.table_name),
            psql.Identifier(_ID_COLUMN),
            psql.Identifier(_DOCUMENT_COLUMN),
            psql.Identifier(_EMBEDDING_COLUMN),
            psql.Identifier(_EMBEDDING_COLUMN),
            psql.Identifier(_EMBEDDING_COLUMN),
            psql.Literal(self.ndims),
            psql.Identifier(_METADATA_COLUMN),
        )
        self.client.execute(create_table_sql, fetch_result=False)

    def _ensure_vector_index(self) -> None:
        try:
            self.storage.set_vector_index(
                _EMBEDDING_COLUMN,
                "Cosine",
                "rabitq",
                use_reorder=True,
            )
        except Exception as e:
            raise RuntimeError("Failed to create vector index") from e

    def _apply_filter(
        self,
        query_builder: Any,
        filter: Optional[dict],
    ) -> Any:
        from psycopg import sql as psql

        if not filter:
            return query_builder

        for key, value in filter.items():
            query_builder = query_builder.where(
                psql.SQL("{} ->> {} = {}").format(
                    psql.Identifier(_METADATA_COLUMN),
                    psql.Literal(key),
                    psql.Literal(str(_extract_metadata_value({key: value}, key))),
                )
            )
        return query_builder

    def _rows_to_docs_and_scores(
        self, rows: Sequence[Tuple[Any, ...]]
    ) -> List[Tuple[Document, float]]:
        docs_and_scores: List[Tuple[Document, float]] = []
        for row in rows:
            distance = row[0]
            document = row[1]
            metadata_raw = row[2]
            if isinstance(metadata_raw, dict):
                metadata = metadata_raw
            elif isinstance(metadata_raw, str):
                try:
                    metadata = json.loads(metadata_raw)
                except json.JSONDecodeError:
                    metadata = {}
            else:
                metadata = {}
            docs_and_scores.append(
                (
                    Document(page_content=document, metadata=metadata),
                    float(distance),
                )
            )
        return docs_and_scores

    @staticmethod
    def _parse_connection_string(connection_string: str) -> Dict[str, Any]:
        """Parse a libpq-style connection string into sdk connection args."""
        config: Dict[str, str] = {}
        for token in shlex.split(connection_string):
            key, _, value = token.partition("=")
            if not key or not _:
                continue
            config[key] = value

        host = config.get("host")
        database = config.get("dbname") or config.get("database")
        user = config.get("user")
        password = config.get("password")

        if not host or not database or not user or password is None:
            raise ValueError(
                "Connection string must include host, dbname, user, and password."
            )

        port = int(config.get("port", "80"))
        schema = config.get("schema") or _DEFAULT_SCHEMA

        return {
            "host": host,
            "port": port,
            "database": database,
            "user": user,
            "password": password,
            "schema": schema,
        }

    @staticmethod
    def get_holo_version(connection_string: str) -> int:
        """Return the Hologres instance version number."""
        row: Any
        import psycopg

        with psycopg.connect(connection_string) as conn:
            with conn.cursor() as cursor:
                cursor.execute("select hg_version_num();")
                row = cursor.fetchone()

        if row is None:
            raise ValueError("Failed to retrieve Hologres version.")

        return int(row[0])

    def add_embeddings(
        self,
        texts: Iterable[str],
        embeddings: List[List[float]],
        metadatas: List[dict],
        ids: List[str],
        **kwargs: Any,
    ) -> None:
        """Add embeddings to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            embeddings: List of list of embedding vectors.
            metadatas: List of metadatas associated with the texts.
            kwargs: vectorstore specific parameters
        """
        text_list = list(texts)
        if not text_list:
            return

        if self._use_holo_search_sdk:
            self._add_embeddings_with_holo_search_sdk(
                text_list, embeddings, metadatas, ids
            )
        else:
            self._add_embeddings_with_hologres_vector(
                text_list, embeddings, metadatas, ids
            )

    def _add_embeddings_with_holo_search_sdk(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: List[dict],
        ids: List[str],
    ) -> None:
        rows: List[List[Any]] = []
        for doc_id, text, embedding, metadata in zip(ids, texts, embeddings, metadatas):
            rows.append(
                [
                    doc_id,
                    text,
                    embedding,
                    json.dumps(metadata, sort_keys=True),
                ]
            )

        try:
            self.storage.upsert_multi(
                _ID_COLUMN,
                rows,
                [_ID_COLUMN, _DOCUMENT_COLUMN, _EMBEDDING_COLUMN, _METADATA_COLUMN],
                update=True,
                update_columns=[_DOCUMENT_COLUMN, _EMBEDDING_COLUMN, _METADATA_COLUMN],
            )
        except Exception as e:
            self.logger.exception(e)
            raise

    def _add_embeddings_with_hologres_vector(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: List[dict],
        ids: List[str],
    ) -> None:
        try:
            schema_datas = [{"document": t} for t in texts]
            self.storage.upsert_vectors(embeddings, ids, metadatas, schema_datas)
        except Exception as e:
            self.logger.exception(e)

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            kwargs: vectorstore specific parameters

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        text_list = list(texts)
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in text_list]

        embeddings = self.embedding_function.embed_documents(text_list)

        if not metadatas:
            metadatas = [{} for _ in text_list]

        self.add_embeddings(text_list, embeddings, metadatas, ids, **kwargs)

        return ids

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Run similarity search with Hologres with distance.

        Args:
            query (str): Query text to search for.
            k (int): Number of results to return. Defaults to 4.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List of Documents most similar to the query.
        """
        embedding = self.embedding_function.embed_query(text=query)
        return self.similarity_search_by_vector(
            embedding=embedding,
            k=k,
            filter=filter,
        )

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List of Documents most similar to the query vector.
        """
        docs_and_scores = self.similarity_search_with_score_by_vector(
            embedding=embedding, k=k, filter=filter
        )
        return [doc for doc, _ in docs_and_scores]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None,
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List of Documents most similar to the query and score for each
        """
        embedding = self.embedding_function.embed_query(query)
        docs = self.similarity_search_with_score_by_vector(
            embedding=embedding, k=k, filter=filter
        )
        return docs

    def similarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[dict] = None,
    ) -> List[Tuple[Document, float]]:
        if self._use_holo_search_sdk:
            return self._similarity_search_with_holo_search_sdk(
                embedding=embedding,
                k=k,
                filter=filter,
            )
        else:
            return self._similarity_search_with_hologres_vector(
                embedding=embedding,
                k=k,
                filter=filter,
            )

    def _similarity_search_with_holo_search_sdk(
        self,
        embedding: List[float],
        k: int,
        filter: Optional[dict],
    ) -> List[Tuple[Document, float]]:
        query_builder = self.storage.search_vector(
            embedding,
            _EMBEDDING_COLUMN,
            output_name="distance",
            distance_method="Cosine",
        )
        query_builder = query_builder.select(
            [
                (_DOCUMENT_COLUMN, None),
                (_METADATA_COLUMN, None),
            ]
        )
        query_builder = self._apply_filter(query_builder, filter)
        query_builder = query_builder.order_by("distance", order="desc").limit(k)
        rows = query_builder.fetchall()
        return self._rows_to_docs_and_scores(rows)

    def _similarity_search_with_hologres_vector(
        self,
        embedding: List[float],
        k: int,
        filter: Optional[dict],
    ) -> List[Tuple[Document, float]]:
        results = self.storage.search(
            vector=embedding,
            k=k,
            select_columns=[_DOCUMENT_COLUMN],
            metadata_filters=filter,
        )
        return [
            (
                Document(
                    page_content=result.get(_DOCUMENT_COLUMN) or "",
                    metadata=result.get(_METADATA_COLUMN) or {},
                ),
                float(result["distance"]),
            )
            for result in results
        ]

    @classmethod
    def from_texts(
        cls: Type[Hologres],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        ndims: int = ADA_TOKEN_COUNT,
        table_name: str = _LANGCHAIN_DEFAULT_TABLE_NAME,
        ids: Optional[List[str]] = None,
        pre_delete_table: bool = False,
        **kwargs: Any,
    ) -> Hologres:
        """
        Return VectorStore initialized from texts and embeddings.
        Hologres connection string is required
        "Either pass it as a parameter
        or set the HOLOGRES_CONNECTION_STRING environment variable.
        Create the connection string by calling
        Hologres.connection_string_from_db_params
        """
        embeddings = embedding.embed_documents(list(texts))

        return cls.__from(
            texts,
            embeddings,
            embedding,
            metadatas=metadatas,
            ids=ids,
            ndims=ndims,
            table_name=table_name,
            pre_delete_table=pre_delete_table,
            **kwargs,
        )

    @classmethod
    def from_embeddings(
        cls,
        text_embeddings: List[Tuple[str, List[float]]],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        ndims: int = ADA_TOKEN_COUNT,
        table_name: str = _LANGCHAIN_DEFAULT_TABLE_NAME,
        ids: Optional[List[str]] = None,
        pre_delete_table: bool = False,
        **kwargs: Any,
    ) -> Hologres:
        """Construct Hologres wrapper from raw documents and pre-
        generated embeddings.

        Return VectorStore initialized from documents and embeddings.
        Hologres connection string is required
        "Either pass it as a parameter
        or set the HOLOGRES_CONNECTION_STRING environment variable.
        Create the connection string by calling
        HologresVector.connection_string_from_db_params

        Example:
            .. code-block:: python

                from langchain_community.vectorstores import Hologres
                from langchain_community.embeddings import OpenAIEmbeddings
                embeddings = OpenAIEmbeddings()
                text_embeddings = embeddings.embed_documents(texts)
                text_embedding_pairs = list(zip(texts, text_embeddings))
                faiss = Hologres.from_embeddings(text_embedding_pairs, embeddings)
        """
        texts = [t[0] for t in text_embeddings]
        embeddings = [t[1] for t in text_embeddings]

        return cls.__from(
            texts,
            embeddings,
            embedding,
            metadatas=metadatas,
            ids=ids,
            ndims=ndims,
            table_name=table_name,
            pre_delete_table=pre_delete_table,
            **kwargs,
        )

    @classmethod
    def from_existing_index(
        cls: Type[Hologres],
        embedding: Embeddings,
        ndims: int = ADA_TOKEN_COUNT,
        table_name: str = _LANGCHAIN_DEFAULT_TABLE_NAME,
        pre_delete_table: bool = False,
        **kwargs: Any,
    ) -> Hologres:
        """
        Get instance of an existing Hologres store.This method will
        return the instance of the store without inserting any new
        embeddings
        """
        connection_string = cls.get_connection_string(kwargs)

        store = cls(
            connection_string=connection_string,
            ndims=ndims,
            table_name=table_name,
            embedding_function=embedding,
            pre_delete_table=pre_delete_table,
        )

        return store

    @classmethod
    def get_connection_string(cls, kwargs: Dict[str, Any]) -> str:
        connection_string: str = get_from_dict_or_env(
            data=kwargs,
            key="connection_string",
            env_key="HOLOGRES_CONNECTION_STRING",
        )

        if not connection_string:
            raise ValueError(
                "Hologres connection string is required"
                "Either pass it as a parameter"
                "or set the HOLOGRES_CONNECTION_STRING environment variable."
                "Create the connection string by calling"
                "Hologres.connection_string_from_db_params"
            )

        return connection_string

    @classmethod
    def from_documents(
        cls: Type[Hologres],
        documents: List[Document],
        embedding: Embeddings,
        ndims: int = ADA_TOKEN_COUNT,
        table_name: str = _LANGCHAIN_DEFAULT_TABLE_NAME,
        ids: Optional[List[str]] = None,
        pre_delete_collection: bool = False,
        **kwargs: Any,
    ) -> Hologres:
        """
        Return VectorStore initialized from documents and embeddings.
        Hologres connection string is required
        "Either pass it as a parameter
        or set the HOLOGRES_CONNECTION_STRING environment variable.
        Create the connection string by calling
        Hologres.connection_string_from_db_params
        """
        texts = [d.page_content for d in documents]
        metadatas = [d.metadata for d in documents]
        connection_string = cls.get_connection_string(kwargs)

        kwargs["connection_string"] = connection_string

        return cls.from_texts(
            texts=texts,
            pre_delete_table=pre_delete_collection,
            embedding=embedding,
            metadatas=metadatas,
            ids=ids,
            ndims=ndims,
            table_name=table_name,
            **kwargs,
        )

    @classmethod
    def connection_string_from_db_params(
        cls,
        host: str,
        port: int,
        database: str,
        user: str,
        password: str,
    ) -> str:
        """Return connection string from database parameters."""
        return (
            f"dbname={database} user={user} password={password} host={host} port={port}"
        )
