from __future__ import annotations

import json
import logging
import re
import struct
import warnings
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
)

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

if TYPE_CHECKING:
    import sqlite3

logger = logging.getLogger(__name__)

_VALID_METADATA_KEY_RE = re.compile(r"^[a-zA-Z0-9_.]+$")

_OPERATOR_MAP: Dict[str, str] = {
    "$eq": "=",
    "$ne": "!=",
    "$gt": ">",
    "$gte": ">=",
    "$lt": "<",
    "$lte": "<=",
}


def serialize_f32(vector: List[float]) -> bytes:
    """Serializes a list of floats into a compact "raw bytes" format

    Source: https://github.com/asg017/sqlite-vec/blob/21c5a14fc71c83f135f5b00c84115139fd12c492/examples/simple-python/demo.py#L8-L10
    """
    return struct.pack("%sf" % len(vector), *vector)


class SQLiteVec(VectorStore):
    """SQLite with Vec extension as a vector database.

    To use, you should have the ``sqlite-vec`` python package installed.
    Example:
        .. code-block:: python
            from langchain_community.vectorstores import SQLiteVec
            from langchain_community.embeddings.openai import OpenAIEmbeddings
            ...
    """

    def __init__(
        self,
        table: str,
        connection: Optional[sqlite3.Connection],
        embedding: Embeddings,
        db_file: str = "vec.db",
    ):
        """Initialize with sqlite client with vss extension."""
        try:
            import sqlite_vec  # noqa  # pylint: disable=unused-import
        except ImportError:
            raise ImportError(
                "Could not import sqlite-vec python package. "
                "Please install it with `pip install sqlite-vec`."
            )

        if not connection:
            connection = self.create_connection(db_file)

        if not isinstance(embedding, Embeddings):
            warnings.warn("embeddings input must be Embeddings object.")

        self._connection = connection
        self._table = table
        self._embedding = embedding

        self.create_table_if_not_exists()

    def create_table_if_not_exists(self) -> None:
        self._connection.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self._table}
            (
                rowid INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT,
                metadata BLOB,
                text_embedding BLOB
            )
            ;
            """
        )
        self._connection.execute(
            f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS {self._table}_vec USING vec0(
                rowid INTEGER PRIMARY KEY,
                text_embedding float[{self.get_dimensionality()}]
            )
            ;
            """
        )
        self._connection.execute(
            f"""
                CREATE TRIGGER IF NOT EXISTS {self._table}_embed_text 
                AFTER INSERT ON {self._table}
                BEGIN
                    INSERT INTO {self._table}_vec(rowid, text_embedding)
                    VALUES (new.rowid, new.text_embedding) 
                    ;
                END;
            """
        )
        self._connection.commit()

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add more texts to the vectorstore index.
        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            kwargs: vectorstore specific parameters
        """
        max_id = self._connection.execute(
            f"SELECT max(rowid) as rowid FROM {self._table}"
        ).fetchone()["rowid"]
        if max_id is None:  # no text added yet
            max_id = 0

        embeds = self._embedding.embed_documents(list(texts))
        if not metadatas:
            metadatas = [{} for _ in texts]
        data_input = [
            (text, json.dumps(metadata), serialize_f32(embed))
            for text, metadata, embed in zip(texts, metadatas, embeds)
        ]
        self._connection.executemany(
            f"INSERT INTO {self._table}(text, metadata, text_embedding) VALUES (?,?,?)",
            data_input,
        )
        self._connection.commit()
        # pulling every ids we just inserted
        results = self._connection.execute(
            f"SELECT rowid FROM {self._table} WHERE rowid > {max_id}"
        )
        return [row["rowid"] for row in results]

    @staticmethod
    def _build_metadata_filter(
        filter: Dict[str, Any],
    ) -> Tuple[str, List[Any]]:
        """Convert a metadata filter dict to SQL WHERE clauses.

        Uses ``json_extract`` on the ``metadata`` column (aliased as ``e``) to
        produce parameterized SQL fragments that can be appended to a query.

        Args:
            filter: Mapping of metadata keys to expected values or operator
                dicts.  Supported operators: ``$eq``, ``$ne``, ``$gt``,
                ``$gte``, ``$lt``, ``$lte``, ``$in``, ``$nin``.

        Returns:
            A ``(sql_fragment, params)`` tuple where *sql_fragment* contains
            one or more ``AND``-joined conditions and *params* is the list of
            bind values.

        Raises:
            ValueError: If a key contains unsafe characters or an unsupported
                operator is used.
        """
        clauses: List[str] = []
        params: List[Any] = []

        for key, value in filter.items():
            if not _VALID_METADATA_KEY_RE.match(key):
                msg = (
                    f"Invalid metadata filter key: {key!r}. "
                    "Keys must contain only alphanumeric characters, "
                    "underscores, and dots."
                )
                raise ValueError(msg)

            json_path = f"json_extract(e.metadata, '$.{key}')"

            if isinstance(value, dict):
                for op, val in value.items():
                    if op in _OPERATOR_MAP:
                        clauses.append(f"{json_path} {_OPERATOR_MAP[op]} ?")
                        params.append(val)
                    elif op == "$in":
                        if not isinstance(val, (list, tuple)):
                            msg = (
                                "$in operator requires a list, "
                                f"got {type(val).__name__}"
                            )
                            raise ValueError(msg)
                        placeholders = ", ".join("?" for _ in val)
                        clauses.append(f"{json_path} IN ({placeholders})")
                        params.extend(val)
                    elif op == "$nin":
                        if not isinstance(val, (list, tuple)):
                            msg = (
                                "$nin operator requires a list, "
                                f"got {type(val).__name__}"
                            )
                            raise ValueError(msg)
                        placeholders = ", ".join("?" for _ in val)
                        clauses.append(f"{json_path} NOT IN ({placeholders})")
                        params.extend(val)
                    else:
                        msg = f"Unsupported filter operator: {op!r}"
                        raise ValueError(msg)
            else:
                clauses.append(f"{json_path} = ?")
                params.append(value)

        return " AND ".join(clauses), params

    def similarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        fetch_k: int = 20,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs and scores most similar to the embedding vector.

        Args:
            embedding: Embedding vector to search with.
            k: Number of documents to return.
            filter: Optional metadata filter dict.  Keys are metadata field
                names; values are either literal values (equality check) or
                operator dicts such as ``{"$gt": 5}``.  Supported operators:
                ``$eq``, ``$ne``, ``$gt``, ``$gte``, ``$lt``, ``$lte``,
                ``$in``, ``$nin``.
            fetch_k: Number of candidates to retrieve from the vector index
                before applying the metadata filter.  Only used when *filter*
                is provided.

        Returns:
            List of ``(Document, distance)`` tuples ordered by distance.
        """
        filter_clause = ""
        filter_params: List[Any] = []
        limit_clause = ""

        vec_k = k
        if filter:
            vec_k = fetch_k
            where_fragment, filter_params = self._build_metadata_filter(filter)
            filter_clause = f"AND {where_fragment}"
            limit_clause = "LIMIT ?"

        sql_query = f"""
            SELECT
                text,
                metadata,
                distance
            FROM {self._table} AS e
            INNER JOIN {self._table}_vec AS v on v.rowid = e.rowid
            WHERE
                v.text_embedding MATCH ?
                AND k = ?
                {filter_clause}
            ORDER BY distance
            {limit_clause}
        """

        params: List[Any] = [serialize_f32(embedding), vec_k, *filter_params]
        if filter:
            params.append(k)

        cursor = self._connection.cursor()
        cursor.execute(sql_query, params)
        results = cursor.fetchall()

        documents = []
        for row in results:
            metadata = json.loads(row["metadata"]) or {}
            doc = Document(page_content=row["text"], metadata=metadata)
            documents.append((doc, row["distance"]))

        return documents

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        fetch_k: int = 20,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to query.

        Args:
            query: Text to look up similar documents to.
            k: Number of documents to return.
            filter: Optional metadata filter dict. See
                :meth:`similarity_search_with_score_by_vector` for details.
            fetch_k: Number of candidates to fetch before filtering.

        Returns:
            List of documents most similar to the query.
        """
        embedding = self._embedding.embed_query(query)
        documents = self.similarity_search_with_score_by_vector(
            embedding=embedding, k=k, filter=filter, fetch_k=fetch_k
        )
        return [doc for doc, _ in documents]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        fetch_k: int = 20,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs and distance scores most similar to query.

        Args:
            query: Text to look up similar documents to.
            k: Number of documents to return.
            filter: Optional metadata filter dict. See
                :meth:`similarity_search_with_score_by_vector` for details.
            fetch_k: Number of candidates to fetch before filtering.

        Returns:
            List of ``(Document, distance)`` tuples.
        """
        embedding = self._embedding.embed_query(query)
        documents = self.similarity_search_with_score_by_vector(
            embedding=embedding, k=k, filter=filter, fetch_k=fetch_k
        )
        return documents

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        fetch_k: int = 20,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to the embedding vector.

        Args:
            embedding: Embedding vector to search with.
            k: Number of documents to return.
            filter: Optional metadata filter dict. See
                :meth:`similarity_search_with_score_by_vector` for details.
            fetch_k: Number of candidates to fetch before filtering.

        Returns:
            List of documents most similar to the embedding.
        """
        documents = self.similarity_search_with_score_by_vector(
            embedding=embedding, k=k, filter=filter, fetch_k=fetch_k
        )
        return [doc for doc, _ in documents]

    @classmethod
    def from_texts(
        cls: Type[SQLiteVec],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        table: str = "langchain",
        db_file: str = "vec.db",
        **kwargs: Any,
    ) -> SQLiteVec:
        """Return VectorStore initialized from texts and embeddings."""
        connection = cls.create_connection(db_file)
        vec = cls(
            table=table, connection=connection, db_file=db_file, embedding=embedding
        )
        vec.add_texts(texts=texts, metadatas=metadatas)
        return vec

    @staticmethod
    def create_connection(db_file: str) -> sqlite3.Connection:
        import sqlite3

        import sqlite_vec

        connection = sqlite3.connect(db_file)
        connection.row_factory = sqlite3.Row
        connection.enable_load_extension(True)
        sqlite_vec.load(connection)
        connection.enable_load_extension(False)
        return connection

    def get_dimensionality(self) -> int:
        """
        Function that does a dummy embedding to figure out how many dimensions
        this embedding function returns. Needed for the virtual table DDL.
        """
        dummy_text = "This is a dummy text"
        dummy_embedding = self._embedding.embed_query(dummy_text)
        return len(dummy_embedding)
