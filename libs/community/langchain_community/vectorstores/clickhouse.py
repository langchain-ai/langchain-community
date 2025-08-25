from __future__ import annotations

import json
import logging
from hashlib import sha1
from threading import Thread
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger()


def has_mul_sub_str(s: str, *args: Any) -> bool:
    """
    Check if a string contains multiple substrings.
    Args:
        s: string to check.
        *args: substrings to check.

    Returns:
        True if all substrings are in the string, False otherwise.
    """
    for a in args:
        if a not in s:
            return False
    return True


class ClickhouseSettings(BaseSettings):
    """`ClickHouse` client configuration.

    Attribute:
        host (str) : An URL to connect to MyScale backend.
                             Defaults to 'localhost'.
        port (int) : URL port to connect with HTTP. Defaults to 8123.
        username (str) : Username to login. Defaults to None.
        password (str) : Password to login. Defaults to None.
        secure (bool) : Connect to server over secure connection. Defaults to False.
        index_type (str): index type string. Default is "HNSW", supported values are
                          "HNSW" or "NONE".
        index_params (Dict): index build parameter.
            Defaults to None. For HNSW, following parameters are supported :-
                "quantization" : One of 'f64','f32', 'f16', 'bf16', 'i8', 'b1'. Default is 'bf16'.
                "hnsw_max_connections_per_layer": Default is 32.
                "hnsw_candidate_list_size_for_construction" : Default is 128.
        index_query_params(dict): index query parameters.
        database (str) : Database name to find the table. Defaults to 'default'.
        table (str) : Table name to operate on.
                      Defaults to 'vector_table'.
        metric (str) : Metric to compute distance,
                       supported are ('L2Distance', 'cosineDistance')
                       Defaults to 'cosineDistance'.

        column_map (Dict) : Column type map to project column name onto langchain
                            semantics. Must have keys: `text`, `id`, `vector`,
                            must be same size to number of columns. For example:
                            .. code-block:: python

                                {
                                    'id': 'text_id',
                                    'uuid': 'global_unique_id'
                                    'embedding': 'text_embedding',
                                    'document': 'text_plain',
                                    'metadata': 'metadata_dictionary_in_json',
                                }

                            Defaults to identity map.
    """

    host: str = "localhost"
    port: int = 8123

    username: Optional[str] = None
    password: Optional[str] = None

    secure: bool = False

    index_type: Optional[str] = "HNSW"
    # ClickHouse supports L2Distance and cosineDistance.
    index_params: Optional[Dict] = {"quantization": "bf16", "hnsw_max_connections_per_layer": "64", "hnsw_candidate_list_size_for_construction": "256"}
    index_query_params: Dict[str, str] = {}

    column_map: Dict[str, str] = {
        "id": "id",
        "uuid": "uuid",
        "document": "document",
        "embedding": "embedding",
        "metadata": "metadata",
    }

    database: str = "default"
    table: str = "langchain"
    metric: str = "cosineDistance"

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="clickhouse_",
        extra="ignore",
    )


class Clickhouse(VectorStore):
    """ClickHouse vector store integration.

    Setup:
        Install ``langchain_community`` and ``clickhouse-connect``:

        .. code-block:: bash

            pip install -qU langchain_community clickhouse-connect

    Key init args — indexing params:
        embedding: Embeddings
            Embedding function to use.

    Key init args — client params:
        config: Optional[ClickhouseSettings]
            ClickHouse client configuration.

    Instantiate:
        .. code-block:: python

            from langchain_community.vectorstores import Clickhouse, ClickhouseSettings
            from langchain_openai import OpenAIEmbeddings

            settings = ClickhouseSettings(table="clickhouse_example")
            vector_store = Clickhouse(embedding=OpenAIEmbeddings(), config=settings)

    Add Documents:
        .. code-block:: python

            from langchain_core.documents import Document

            document_1 = Document(page_content="foo", metadata={"baz": "bar"})
            document_2 = Document(page_content="thud", metadata={"bar": "baz"})
            document_3 = Document(page_content="i will be deleted :(")

            documents = [document_1, document_2, document_3]
            ids = ["1", "2", "3"]
            vector_store.add_documents(documents=documents, ids=ids)

    Delete Documents:
        .. code-block:: python

            vector_store.delete(ids=["3"])

    Search:
        .. code-block:: python

            results = vector_store.similarity_search(query="What did I eat for breakfast?",k=1)
            for doc in results:
                print(f"* {doc.page_content} [{doc.metadata}]")

        .. code-block:: python

    Search with filter:
        .. code-block:: python

            results = vector_store.similarity_search(query="What did I eat for breakfast?",k=1,filter="<filter>")
            for doc in results:
                print(f"* {doc.page_content} [{doc.metadata}]")

        .. code-block:: python

    Search with score:
        .. code-block:: python

            results = vector_store.similarity_search_with_relevance_scores(query="Will it be hot tomorrow?",k=1)
            for doc, score in results:
                print(f"* [SIM={score:3f}] {doc.page_content} [{doc.metadata}]")

        .. code-block:: python

    Use as Retriever:
        .. code-block:: python

            retriever = vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"k": 1, "score_threshold": 0.5},
            )
            retriever.invoke("Stealing from the bank is a crime", filter={"source": "news"})

        .. code-block:: python

    """  # noqa: E501

    def __init__(
        self,
        embedding: Embeddings,
        config: Optional[ClickhouseSettings] = None,
        **kwargs: Any,
    ) -> None:
        """ClickHouse Wrapper to LangChain

        Args:
            embedding_function (Embeddings): embedding function to use
            config (ClickHouseSettings): Configuration to ClickHouse Client
            kwargs (any): Other keyword arguments will pass into
                [clickhouse-connect](https://docs.clickhouse.com/)
        """
        try:
            from clickhouse_connect import get_client
        except ImportError:
            raise ImportError(
                "Could not import clickhouse connect python package. "
                "Please install it with `pip install clickhouse-connect`."
            )
        try:
            from tqdm import tqdm

            self.pgbar = tqdm
        except ImportError:
            # Just in case if tqdm is not installed
            self.pgbar = lambda x, **kwargs: x
        super().__init__()
        if config is not None:
            self.config = config
        else:
            self.config = ClickhouseSettings()
        assert self.config
        assert self.config.host and self.config.port
        assert (
            self.config.column_map
            and self.config.database
            and self.config.table
            and self.config.metric
        )
        for k in ["id", "embedding", "document", "metadata", "uuid"]:
            assert k in self.config.column_map
        assert self.config.metric in [
            "L2Distance",
            "cosineDistance",
        ]
        # initialize the schema
        dim = len(embedding.embed_query("test"))
        self.schema = self._schema(dim, self.config.index_params)

        self.dim = dim
        self.BS = "\\"
        self.must_escape = ("\\", "'")
        self.embedding_function = embedding
        self.dist_order = "ASC"

        # Create a connection to clickhouse
        self.client = get_client(
            host=self.config.host,
            port=self.config.port,
            username=self.config.username,
            password=self.config.password,
            secure=self.config.secure,
            **kwargs,
        )
        self.client.command(self.schema)

    def _schema(self, dim: int, index_params: Optional[dict] = None) -> str:
        """Create table schema
        :param dim: dimension of embeddings
        :param index_params: parameters used for index

        This function returns a `CREATE TABLE` statement based on the value of
        `self.config.index_type`.
        If an index type is specified that index will be created, otherwise
        no index will be created.
        In the case of there being no index, a linear scan will be performed
        when the embedding field is queried.
        """

        if self.config.index_type.lower() == "hnsw":
            quantization = "bf16"
            M = 32
            ef_c = 128
            if (
                index_params and "quantization" in index_params
            ):
                quantization = index_params["quantization"]
            if (
                index_params and "hnsw_max_connections_per_layer" in index_params
            ):
                M = int(index_params["hnsw_max_connections_per_layer"])
            if (
                index_params
                and "hnsw_candidate_list_size_for_construction" in index_params
            ):
                ef_c = int(index_params[
                    "hnsw_candidate_list_size_for_construction"])
            return f"""\
        CREATE TABLE IF NOT EXISTS {self.config.database}.{self.config.table}(
            {self.config.column_map["id"]} Nullable(String),
            {self.config.column_map["document"]} Nullable(String),
            {self.config.column_map["embedding"]} Array(Float32),
            {self.config.column_map["metadata"]} JSON,
            {self.config.column_map["uuid"]} UUID DEFAULT generateUUIDv4(),
            CONSTRAINT cons_vec_len CHECK length(
                {self.config.column_map["embedding"]}) = {dim},
            INDEX vec_idx {self.config.column_map["embedding"]} TYPE \
            vector_similarity('{self.config.index_type.lower()}','{self.config.metric}', {dim}, '{quantization}', {M}, {ef_c})
        ) ENGINE = MergeTree ORDER BY uuid SETTINGS index_granularity = 8192\
        """
        else:
            return f"""\
                CREATE TABLE IF NOT EXISTS {self.config.database}.{self.config.table}(
                    {self.config.column_map["id"]} Nullable(String),
                    {self.config.column_map["document"]} Nullable(String),
                    {self.config.column_map["embedding"]} Array(Float32),
                    {self.config.column_map["metadata"]} JSON,
                    {self.config.column_map["uuid"]} UUID DEFAULT generateUUIDv4(),
                    CONSTRAINT cons_vec_len CHECK length({
                self.config.column_map["embedding"]
            }) = {dim}
                ) ENGINE = MergeTree ORDER BY uuid
                """

    @property
    def embeddings(self) -> Embeddings:
        """Provides access to the embedding mechanism used by the Clickhouse instance.

        This property allows direct access to the embedding function or model being
        used by the Clickhouse instance to convert text documents into embedding vectors
        for vector similarity search.

        Returns:
            The `Embeddings` instance associated with this Clickhouse instance.
        """
        return self.embedding_function

    def escape_str(self, value: str) -> str:
        """Escape special characters in a string for Clickhouse SQL queries.

        This method is used internally to prepare strings for safe insertion
        into SQL queries by escaping special characters that might otherwise
        interfere with the query syntax.

        Args:
            value: The string to be escaped.

        Returns:
            The escaped string, safe for insertion into SQL queries.
        """
        return "".join(f"{self.BS}{c}" if c in self.must_escape else c for c in value)

    def _build_insert_sql(self, transac: Iterable, column_names: Iterable[str]) -> str:
        """Construct an SQL query for inserting data into the Clickhouse database.

        This method formats and constructs an SQL `INSERT` query string using the
        provided transaction data and column names. It is utilized internally during
        the process of batch insertion of documents and their embeddings into the
        database.

        Args:
            transac: iterable of tuples, representing a row of data to be inserted.
            column_names: iterable of strings representing the names of the columns
                into which data will be inserted.

        Returns:
            A string containing the constructed SQL `INSERT` query.
        """
        ks = ",".join(column_names)
        _data = []
        for n in transac:
            n = ",".join([f"'{self.escape_str(str(_n))}'" for _n in n])
            _data.append(f"({n})")
        i_str = f"""
                INSERT INTO TABLE 
                    {self.config.database}.{self.config.table}({ks})
                VALUES
                {",".join(_data)}
                """
        return i_str

    def _insert(self, transac: Iterable, column_names: Iterable[str]) -> None:
        """Execute an SQL query to insert data into the Clickhouse database.

        This method performs the actual insertion of data into the database by
        executing the SQL query constructed by `_build_insert_sql`. It's a critical
        step in adding new documents and their associated data into the vector store.

        Args:
            transac:iterable of tuples, representing a row of data to be inserted.
            column_names: An iterable of strings representing the names of the columns
                into which data will be inserted.
        """
        _insert_query = self._build_insert_sql(transac, column_names)
        self.client.command(_insert_query)

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
            ids: Optional list of ids to associate with the texts.
            batch_size: Batch size of insertion
            metadata: Optional column data to be inserted

        Returns:
            List of ids from adding the texts into the VectorStore.

        """
        # Embed and create the documents
        ids = ids or [sha1(t.encode("utf-8")).hexdigest() for t in texts]
        colmap_ = self.config.column_map
        transac = []
        column_names = {
            colmap_["id"]: ids,
            colmap_["document"]: texts,
            colmap_["embedding"]: self.embedding_function.embed_documents(list(texts)),
        }
        metadatas = metadatas or [{} for _ in texts]
        column_names[colmap_["metadata"]] = map(json.dumps, metadatas)
        assert len(set(colmap_) - set(column_names)) >= 0
        keys, values = zip(*column_names.items())
        try:
            t = None
            for v in self.pgbar(
                zip(*values), desc="Inserting data...", total=len(metadatas)
            ):
                assert (
                    len(v[keys.index(self.config.column_map["embedding"])]) == self.dim
                )
                transac.append(v)
                if len(transac) == batch_size:
                    if t:
                        t.join()
                    t = Thread(target=self._insert, args=[transac, keys])
                    t.start()
                    transac = []
            if len(transac) > 0:
                if t:
                    t.join()
                self._insert(transac, keys)
            return [i for i in ids]
        except Exception as e:
            logger.error(f"\033[91m\033[1m{type(e)}\033[0m \033[95m{str(e)}\033[0m")
            return []

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[Dict[Any, Any]]] = None,
        config: Optional[ClickhouseSettings] = None,
        text_ids: Optional[Iterable[str]] = None,
        batch_size: int = 32,
        **kwargs: Any,
    ) -> Clickhouse:
        """Create ClickHouse wrapper with existing texts

        Args:
            embedding_function (Embeddings): Function to extract text embedding
            texts (Iterable[str]): List or tuple of strings to be added
            config (ClickHouseSettings, Optional): ClickHouse configuration
            text_ids (Optional[Iterable], optional): IDs for the texts.
                                                     Defaults to None.
            batch_size (int, optional): Batchsize when transmitting data to ClickHouse.
                                        Defaults to 32.
            metadata (List[dict], optional): metadata to texts. Defaults to None.
            Other keyword arguments will pass into
                [clickhouse-connect](https://clickhouse.com/docs/en/integrations/python#clickhouse-connect-driver-api)
        Returns:
            ClickHouse Index
        """
        ctx = cls(embedding, config, **kwargs)
        ctx.add_texts(texts, ids=text_ids, batch_size=batch_size, metadatas=metadatas)
        return ctx

    def __repr__(self) -> str:
        """Text representation for ClickHouse Vector Store, prints backends, username
            and schemas. Easy to use with `str(ClickHouse())`

        Returns:
            repr: string to show connection info and data schema
        """
        _repr = f"\033[92m\033[1m{self.config.database}.{self.config.table} @ "
        _repr += f"{self.config.host}:{self.config.port}\033[0m\n\n"
        _repr += f"\033[1musername: {self.config.username}\033[0m\n\nTable Schema:\n"
        _repr += "-" * 51 + "\n"
        for r in self.client.query(
            f"DESC {self.config.database}.{self.config.table}"
        ).named_results():
            _repr += (
                f"|\033[94m{r['name']:24s}\033[0m|\033[96m{r['type']:24s}\033[0m|\n"
            )
        _repr += "-" * 51 + "\n"
        return _repr

    def _build_query_sql(
        self, q_emb: List[float], topk: int, where_str: Optional[str] = None
    ) -> str:
        """Construct an SQL query for performing a similarity search.

        This internal method generates an SQL query for finding the top-k most similar
        vectors in the database to a given query vector.It allows for optional filtering
        conditions to be applied via a WHERE clause.

        Args:
            q_emb: The query vector as a list of floats.
            topk: The number of top similar items to retrieve.
            where_str: opt str representing additional WHERE conditions for the query
                Defaults to None.

        Returns:
            A string containing the SQL query for the similarity search.
        """
        q_emb_str = ",".join(map(str, q_emb))
        if where_str:
            where_str = f"PREWHERE {where_str}"
        else:
            where_str = ""

        settings_strs = []
        if self.config.index_query_params:
            for k in self.config.index_query_params:
                settings_strs.append(f"SETTINGS {k}={self.config.index_query_params[k]}")
        q_str = f"""
            SELECT {self.config.column_map["document"]}, 
                {self.config.column_map["metadata"]}, dist
            FROM {self.config.database}.{self.config.table}
            {where_str}
            ORDER BY L2Distance({self.config.column_map["embedding"]}, [{q_emb_str}]) 
                AS dist {self.dist_order}
            LIMIT {topk} {" ".join(settings_strs)}
            """
        return q_str

    def similarity_search(
        self, query: str, k: int = 4, where_str: Optional[str] = None, **kwargs: Any
    ) -> List[Document]:
        """Perform a similarity search with ClickHouse

        Args:
            query (str): query string
            k (int, optional): Top K neighbors to retrieve. Defaults to 4.
            where_str (Optional[str], optional): where condition string.
                                                 Defaults to None.

            NOTE: Please do not let end-user to fill this and always be aware
                  of SQL injection. When dealing with metadatas, remember to
                  use `{self.metadata_column}.attribute` instead of `attribute`
                  alone. The default name for it is `metadata`.

        Returns:
            List[Document]: List of Documents
        """
        return self.similarity_search_by_vector(
            self.embedding_function.embed_query(query), k, where_str, **kwargs
        )

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        where_str: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Perform a similarity search with ClickHouse by vectors

        Args:
            query (str): query string
            k (int, optional): Top K neighbors to retrieve. Defaults to 4.
            where_str (Optional[str], optional): where condition string.
                                                 Defaults to None.

            NOTE: Please do not let end-user to fill this and always be aware
                  of SQL injection. When dealing with metadatas, remember to
                  use `{self.metadata_column}.attribute` instead of `attribute`
                  alone. The default name for it is `metadata`.

        Returns:
            List[Document]: List of documents
        """
        q_str = self._build_query_sql(embedding, k, where_str)
        try:
            return [
                Document(
                    page_content=r[self.config.column_map["document"]],
                    metadata=r[self.config.column_map["metadata"]],
                )
                for r in self.client.query(q_str).named_results()
            ]
        except Exception as e:
            logger.error(f"\033[91m\033[1m{type(e)}\033[0m \033[95m{str(e)}\033[0m")
            return []

    def similarity_search_with_relevance_scores(
        self, query: str, k: int = 4, where_str: Optional[str] = None, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        """Perform a similarity search with ClickHouse

        Args:
            query (str): query string
            k (int, optional): Top K neighbors to retrieve. Defaults to 4.
            where_str (Optional[str], optional): where condition string.
                                                 Defaults to None.

            NOTE: Please do not let end-user to fill this and always be aware
                  of SQL injection. When dealing with metadatas, remember to
                  use `{self.metadata_column}.attribute` instead of `attribute`
                  alone. The default name for it is `metadata`.

        Returns:
            List[Document]: List of (Document, similarity)
        """
        q_str = self._build_query_sql(
            self.embedding_function.embed_query(query), k, where_str
        )
        try:
            return [
                (
                    Document(
                        page_content=r[self.config.column_map["document"]],
                        metadata=r[self.config.column_map["metadata"]],
                    ),
                    r["dist"],
                )
                for r in self.client.query(q_str).named_results()
            ]
        except Exception as e:
            logger.error(f"\033[91m\033[1m{type(e)}\033[0m \033[95m{str(e)}\033[0m")
            return []

    def drop(self) -> None:
        """
        Helper function: Drop data
        """
        self.client.command(
            f"DROP TABLE IF EXISTS {self.config.database}.{self.config.table}"
        )

    @property
    def metadata_column(self) -> str:
        return self.config.column_map["metadata"]
