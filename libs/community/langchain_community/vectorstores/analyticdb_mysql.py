from __future__ import annotations

import json
import logging
import os
from hashlib import sha1
from threading import Thread
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from pydantic_settings import BaseSettings

from langchain_community.vectorstores.utils import maximal_marginal_relevance

# logger
logger = logging.getLogger()

# use the .env that is inside the current folder
load_dotenv(dotenv_path=".env", override=False)


class AnalyticDBMySQLSettings(BaseSettings):
    """AnalyticDB MySQL client configuration.

    Attributes:
        host (str) : An URL to connect to frontend. Defaults to 'localhost'.
        port (int) : URL port to connect with HTTP. Defaults to 3306.
        user (str) : Username to login. Defaults to 'admin'.
        password (str) : Password to login. Defaults to None.
        database (str) : Database name to find the table. Defaults to 'vectorstore'.
        table (str) : Table name to operate on. Defaults to 'langchain'.

        column_map (Dict) : Column type map to project column name onto langchain
                            semantics. Must have keys: `text`, `id`, `vector`,
                            must be same size to number of columns. For example:
                            .. code-block:: python

                                {
                                    'id': 'text_id',
                                    'embedding': 'text_embedding',
                                    'document': 'text_plain',
                                    'metadata': 'metadata_dictionary_in_json',
                                }

                            Defaults to identity map.
    """

    host: str = "localhost"
    port: int = 3306
    user: str = "admin"
    password: str = "admin"

    database: str = "vectorstore"
    table: str = "langchain"

    column_map: Dict[str, str] = {
        "id": "id",
        "document": "document",
        "embedding": "embedding",
        "metadata": "metadata",
    }

    def __getitem__(self, item: str) -> Any:
        """Get attribute.

        Args:
            item: The configuration key.
        """
        return getattr(self, item)

    @staticmethod
    def get_env_config() -> AnalyticDBMySQLSettings:
        """Get AnalyticDB MySQL configuration from env."""
        config = AnalyticDBMySQLSettings()
        config.host = os.getenv("ADB_HOST", "localhost")
        config.port = int(os.getenv("ADB_PORT", "3306"))
        config.user = os.getenv("ADB_USER", "admin")
        config.password = os.getenv("ADB_PASSWORD", "admin")
        config.database = os.getenv("ADB_DATABASE", "vectorstore")
        config.table = os.getenv("ADB_TABLE", "langchain")
        return config


class AnalyticDBMySQL(VectorStore):
    """`AnalyticDB MySQL` vector store.

    You need a `pymysql` python package, and a valid account
    to connect to AnalyticDB MySQL.

    For more information, please visit
        [AnalyticDB MySQL official site](https://www.alibabacloud.com/en/product/analyticdb-for-mysql)

    For example:
    .. code-block:: python
        def create_adb_vectorstore(
            documents: list[Document],
            model_name: str = "text-embedding-v4"
        ) -> VectorStore:
            from langchain_community.embeddings import DashScopeEmbeddings
            from langchain_community.vectorstores import AnalyticDBMySQL

            embeddings = DashScopeEmbeddings(
                model=model_name,
                dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
            )

            return AnalyticDBMySQL.from_documents(
                documents=documents,
                embedding=embeddings,
            )
    """

    def __init__(
        self,
        embedding: Embeddings,
        config: Optional[AnalyticDBMySQLSettings] = None,
        **kwargs: Any,
    ) -> None:
        """Constructor for AnalyticDB MySQL.

        Args:
            embedding (Embeddings): Text embedding model.
            config (AnalyticDBMySQLSettings): AnalyticDB MySQL client configuration.
        """
        try:
            import pymysql  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError(
                "Could not import pymysql python package. "
                "Please install it with `pip install pymysql`."
            )

        super().__init__()
        self._embedding = embedding
        if config is not None:
            self.config = config
        else:
            self.config = AnalyticDBMySQLSettings.get_env_config()

        logger.debug(self.config)
        assert self.config
        assert self.config.host and self.config.port
        assert self.config.column_map and self.config.database and self.config.table
        for k in ["id", "embedding", "document", "metadata"]:
            assert k in self.config.column_map

        # Create a connection to AnalyticDB MySQL
        self.connection = pymysql.connect(
            host=self.config.host,
            port=self.config.port,
            user=self.config.user,
            password=self.config.password,
            database=self.config.database,
            **kwargs,
        )

        # Create table if not exists
        self._create_table()

    @property
    def embeddings(self) -> Embeddings:
        """Vector embeddings object."""
        return self._embedding

    @property
    def metadata_column(self) -> str:
        """Column metadata."""
        return self.config.column_map["metadata"]

    def _create_table(self) -> None:
        """Create vector table if not exists."""
        result = self._execute_sql(
            f"SELECT 1 FROM information_schema.kepler_meta_tables "
            f"WHERE table_schema='{self.config.database}' "
            f"AND table_name=lower('{self.config.table}')"
        )
        if len(result) == 0:
            dim = len(self.embeddings.embed_query("adb"))
            schema = f"""\
            CREATE TABLE IF NOT EXISTS {self.config.database}.{self.config.table}(
                {self.config.column_map["id"]} varchar(50),
                {self.config.column_map["document"]} text,
                {self.config.column_map["embedding"]} array<float>({dim}),
                {self.config.column_map["metadata"]} json,
                ANN INDEX idx_vector_embed({self.config.column_map["embedding"]}),
                PRIMARY KEY ({self.config.column_map["id"]})
            )\
            """
            self._execute_sql(schema)
        else:
            logger.info(f"Table {self.config.database}.{self.config.table} exists.")

    def _execute_sql(
        self, sql: str, data: Optional[dict[str, Any] | list[dict[str, Any]]] = None
    ) -> List[dict[str, Any]]:
        """Execute sql.

        Args:
            sql: DML sql.
            data: SQL parameters or List of datas.
        """

        cursor = self.connection.cursor()
        if data is None:
            cursor.execute(sql)
        else:
            if isinstance(data, list):
                cursor.executemany(sql, data)
            else:
                cursor.execute(sql, data)

        columns = cursor.description
        result = []
        for value in cursor.fetchall():
            r = {}
            for idx, datum in enumerate(value):
                k = columns[idx][0]
                r[k] = datum
            result.append(r)
        cursor.close()
        return result

    def _insert(self, values: list[dict[str, Any]]) -> None:
        """Insert data.

        Args:
            values: List of datas to insert.
        """

        col_id = self.config.column_map["id"]
        col_doc = self.config.column_map["document"]
        col_embed = self.config.column_map["embedding"]
        col_meta = self.config.column_map["metadata"]
        sql = f"""
            REPLACE INTO {self.config.database}.{self.config.table} (
                {col_id}, {col_doc}, {col_embed}, {col_meta}
            ) VALUES (
                %({col_id})s, %({col_doc})s, %({col_embed})s, %({col_meta})s
            )
            """
        self._execute_sql(sql, values)

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        batch_size: int = 32,
        ids: Optional[Iterable[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Insert more texts through the embeddings and add to the VectorStore.

        Args:
            texts: Iterable of strings to add to the VectorStore.
            metadatas: Optional column data to be inserted.
            batch_size: Batch size of insertion.
            ids: Optional list of ids to associate with the texts.

        Returns:
            List of ids from adding the texts into the VectorStore.
        """
        # Embed and create the documents
        ids = ids or [sha1(t.encode("utf-8")).hexdigest() for t in texts]
        embeddings = self._embedding.embed_documents(list(texts))
        metas = metadatas or [{} for _ in texts]

        datas = []
        try:
            t = None
            for tid, document, embedding, metadata in zip(
                ids, texts, embeddings, metas
            ):
                v = {
                    self.config.column_map["id"]: tid,
                    self.config.column_map["document"]: document,
                    self.config.column_map["embedding"]: json.dumps(embedding),
                    self.config.column_map["metadata"]: json.dumps(metadata),
                }
                datas.append(v)
                if len(datas) == batch_size:
                    if t:
                        t.join()
                    t = Thread(target=self._insert, args=[datas])
                    t.start()
                    datas = []
            if len(datas) > 0:
                if t:
                    t.join()
                self._insert(datas)
            return [i for i in ids]
        except Exception as e:
            logger.error(f"add_texts error, {e}")
            raise RuntimeError(f"Failed to add texts: {e}") from e

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[Dict[Any, Any]]] = None,
        config: Optional[AnalyticDBMySQLSettings] = None,
        text_ids: Optional[Iterable[str]] = None,
        batch_size: int = 32,
        **kwargs: Any,
    ) -> AnalyticDBMySQL:
        """Create AnalyticDB MySQL wrapper with existing texts

        Args:
            texts: List or tuple of strings to be added.
            embedding: Function to extract text embedding.
            metadatas: metadata to texts. Defaults to None.
            config: AnalyticDB MySQL client configuration.
            text_ids: IDs for the texts. Defaults to None.
            batch_size: Batch size when transmitting data. Defaults to 32.

        Returns:
            AnalyticDB MySQL Index
        """
        ctx = cls(embedding, config=config, **kwargs)
        ctx.add_texts(texts, ids=text_ids, batch_size=batch_size, metadatas=metadatas)
        return ctx

    def _build_query_sql(self, q_emb: List[float], top_k: int, **kwargs: Any) -> str:
        """Get query sql string.

        Args:
            q_emb: query string embedding.
            top_k: Top K neighbors to retrieve.
        """

        q_emb_str = ",".join(map(str, q_emb))
        q_str = f"""
        SELECT
            {self.config.column_map["id"]} as id,
            {self.config.column_map["document"]} as document,
            {self.config.column_map["metadata"]} as metadata,
            l2_distance({self.config.column_map["embedding"]},'[{q_emb_str}]') as dist,
            {self.config.column_map["embedding"]} as embedding
        FROM {self.config.database}.{self.config.table}
        ORDER BY dist
        LIMIT {top_k}
        """

        logger.debug(q_str)
        return q_str

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        **kwargs: Any,
    ) -> List[Document]:
        """Perform a similarity search with AnalyticDB MySQL by vectors

        Args:
            embedding: query string embedding.
            k: Top K neighbors to retrieve. Defaults to 4.

        Returns:
            List[Document]: List of Documents
        """

        q_str = self._build_query_sql(embedding, k, **kwargs)
        try:
            q_r = self._execute_sql(q_str)
            return [
                Document(
                    page_content=r[self.config.column_map["document"]],
                    metadata=json.loads(r[self.config.column_map["metadata"]]),
                )
                for r in q_r
            ]
        except Exception as e:
            logger.error(f"similarity_search_by_vector, {e}")
            return []

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """Perform a similarity search with AnalyticDB MySQL

        Args:
            query: query string.
            k: Top K neighbors to retrieve. Defaults to 4.

        Returns:
            List[Document]: List of Documents
        """
        return self.similarity_search_by_vector(
            self._embedding.embed_query(query), k, **kwargs
        )

    def similarity_search_with_relevance_scores(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        """Perform a similarity search with AnalyticDB MySQL

        Args:
            query: query string.
            k: Top K neighbors to retrieve. Defaults to 4.

        Returns:
            List[Document, float]: List of (Document, similarity)
        """

        q_str = self._build_query_sql(self._embedding.embed_query(query), k, **kwargs)
        try:
            return [
                (
                    Document(
                        page_content=r[self.config.column_map["document"]],
                        metadata=json.loads(r[self.config.column_map["metadata"]]),
                    ),
                    r["dist"],
                )
                for r in self._execute_sql(q_str)
            ]
        except Exception as e:
            logger.error(f"similarity_search_with_relevance_scores, {e}")
            return []

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> list[Document]:
        """Return docs selected using the maximal marginal relevance.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.

        Returns:
            List[Document]: List of Documents
        """

        q_str = self._build_query_sql(embedding, fetch_k, **kwargs)
        q_r = self._execute_sql(q_str)

        r_embeddings = [json.loads(r[self.config.column_map["embedding"]]) for r in q_r]
        mmr_selected = maximal_marginal_relevance(
            np.array(embedding, dtype=np.float32),
            r_embeddings,
            k=k,
            lambda_mult=lambda_mult,
        )

        candidates = [
            Document(
                page_content=r[self.config.column_map["document"]],
                metadata=json.loads(r[self.config.column_map["metadata"]]),
            )
            for r in q_r
        ]

        selected_results = [r for i, r in enumerate(candidates) if i in mmr_selected]
        return selected_results

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 5,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.

        Args:
            query: Query string.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.

        Returns:
            List[Document]: List of Documents
        """

        if self.embeddings is None:
            raise ValueError("For MMR search, you must specify an embedding function.")

        embedding = self.embeddings.embed_query(query)
        return self.max_marginal_relevance_search_by_vector(
            embedding=embedding,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            **kwargs,
        )

    def delete(self, ids: list[str] | None = None, **kwargs: Any) -> bool | None:
        """Delete rows from table.

        Args:
            ids: List of ids.
        """

        if ids:
            params = {"ids": ",".join([f"'{id}'" for id in ids])}
            delete_sql = (
                f"DELETE FROM {self.config.database}.{self.config.table} "
                f"WHERE {self.config.column_map['id']} IN (%(ids)s)"
            )
            self._execute_sql(delete_sql, params)
        else:
            self._execute_sql(
                f"TRUNCATE TABLE {self.config.database}.{self.config.table}",
            )
        return True

    def drop(self) -> None:
        """Drop table."""
        self._execute_sql(
            f"DROP TABLE IF EXISTS {self.config.database}.{self.config.table}",
        )
