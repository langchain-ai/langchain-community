from __future__ import annotations

import logging
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    TypeVar,
    Union,
)

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings
    from pymongo.collection import Collection


# Before Python 3.11 native StrEnum is not available
class DocumentDBSimilarityType(str, Enum):
    """DocumentDB Similarity Type as enumerator."""

    COS = "cosine"
    """Cosine similarity"""
    DOT = "dotProduct"
    """Dot product"""
    EUC = "euclidean"
    """Euclidean distance"""


DocumentDBDocumentType = TypeVar("DocumentDBDocumentType", bound=Dict[str, Any])

logger = logging.getLogger(__name__)

DEFAULT_INSERT_BATCH_SIZE = 128


class DocumentDBVectorSearch(VectorStore):
    """`Amazon DocumentDB (with MongoDB compatibility)` vector store.
    Please refer to the official Vector Search documentation for more details:
    https://docs.aws.amazon.com/documentdb/latest/developerguide/vector-search.html

    To use, you should have both:
    - the ``pymongo`` python package installed
    - a connection string and credentials associated with a DocumentDB cluster

    Example:
        . code-block:: python

            from langchain_community.vectorstores import DocumentDBVectorSearch
            from langchain_community.embeddings.openai import OpenAIEmbeddings
            from pymongo import MongoClient

            mongo_client = MongoClient("<YOUR-CONNECTION-STRING>")
            collection = mongo_client["<db_name>"]["<collection_name>"]
            embeddings = OpenAIEmbeddings()
            vectorstore = DocumentDBVectorSearch(collection, embeddings)
    """

    def __init__(
        self,
        collection: Collection[DocumentDBDocumentType],
        embedding: Embeddings,
        *,
        index_name: str = "vectorSearchIndex",
        text_key: str = "textContent",
        embedding_key: str = "vectorContent",
    ):
        """Constructor for DocumentDBVectorSearch

        Args:
            collection: MongoDB collection to add the texts to.
            embedding: Text embedding model to use.
            index_name: Name of the Vector Search index.
            text_key: MongoDB field that will contain the text
                for each document.
            embedding_key: MongoDB field that will contain the embedding
                for each document.
        """
        self._collection = collection
        self._embedding = embedding
        self._index_name = index_name
        self._text_key = text_key
        self._embedding_key = embedding_key
        self._similarity_type = DocumentDBSimilarityType.COS

    @property
    def embeddings(self) -> Embeddings:
        return self._embedding

    def get_index_name(self) -> str:
        """Returns the index name

        Returns:
            Returns the index name

        """
        return self._index_name

    @classmethod
    def from_connection_string(
        cls,
        connection_string: str,
        namespace: str,
        embedding: Embeddings,
        **kwargs: Any,
    ) -> DocumentDBVectorSearch:
        """Creates an Instance of DocumentDBVectorSearch from a Connection String

        Args:
            connection_string: The DocumentDB cluster endpoint connection string
            namespace: The namespace (database.collection)
            embedding: The embedding utility
            **kwargs: Dynamic keyword arguments

        Returns:
            an instance of the vector store

        """
        try:
            from pymongo import MongoClient
        except ImportError:
            raise ImportError(
                "Could not import pymongo, please install it with "
                "`pip install pymongo`."
            )
        client: MongoClient = MongoClient(connection_string)
        db_name, collection_name = namespace.split(".")
        collection = client[db_name][collection_name]
        return cls(collection, embedding, **kwargs)

    def index_exists(self) -> bool:
        """Verifies if the specified index name during instance
            construction exists on the collection

        Returns:
          Returns True on success and False if no such index exists
            on the collection
        """
        cursor = self._collection.list_indexes()
        index_name = self._index_name

        for res in cursor:
            current_index_name = res.pop("name")
            if current_index_name == index_name:
                return True

        return False

    def delete_index(self) -> None:
        """Deletes the index specified during instance construction if it exists"""
        if self.index_exists():
            self._collection.drop_index(self._index_name)
            # Raises OperationFailure on an error (e.g. trying to drop
            # an index that does not exist)

    def create_index(
        self,
        dimensions: int = 1536,
        similarity: DocumentDBSimilarityType = DocumentDBSimilarityType.COS,
        m: int = 16,
        ef_construction: int = 64,
    ) -> dict[str, Any]:
        """Creates an index using the index name specified at
            instance construction

        Args:
            dimensions: Number of dimensions for vector similarity.
                The maximum number of supported dimensions is 2000

            similarity: Similarity algorithm to use with the HNSW index.
                 Possible options are:
                    - DocumentDBSimilarityType.COS (cosine distance),
                    - DocumentDBSimilarityType.EUC (Euclidean distance), and
                    - DocumentDBSimilarityType.DOT (dot product).

            m: Specifies the max number of connections for an HNSW index.
                Large impact on memory consumption.

            ef_construction: Specifies the size of the dynamic candidate list
                for constructing the graph for HNSW index. Higher values lead
                to more accurate results but slower indexing speed.


        Returns:
            An object describing the created index

        """
        self._similarity_type = similarity

        # prepare the command
        create_index_commands = {
            "createIndexes": self._collection.name,
            "indexes": [
                {
                    "name": self._index_name,
                    "key": {self._embedding_key: "vector"},
                    "vectorOptions": {
                        "type": "hnsw",
                        "similarity": similarity,
                        "dimensions": dimensions,
                        "m": m,
                        "efConstruction": ef_construction,
                    },
                }
            ],
        }

        # retrieve the database object
        current_database = self._collection.database

        # invoke the command from the database object
        create_index_responses: dict[str, Any] = current_database.command(
            create_index_commands
        )

        return create_index_responses

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        # FIX for issue #507: Accept ids parameter to support Indexing API
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List:
        batch_size = kwargs.get("batch_size", DEFAULT_INSERT_BATCH_SIZE)
        _metadatas: Union[List, Generator] = metadatas or ({} for _ in texts)
        texts_batch = []
        metadatas_batch = []
        # FIX for issue #507: Track ids for batching
        ids_batch = []
        result_ids = []

        # FIX for issue #507: Convert texts to list to enable indexing with ids
        texts_list = list(texts)
        _metadatas_list = list(_metadatas)

        # Validate ids length before iteration to prevent IndexError
        if ids and len(ids) != len(texts_list):
            raise ValueError(
                f"Number of ids ({len(ids)}) must match "
                f"number of texts ({len(texts_list)})"
            )

        for i, (text, metadata) in enumerate(zip(texts_list, _metadatas_list)):
            texts_batch.append(text)
            metadatas_batch.append(metadata)
            # FIX for issue #507: Add corresponding id to batch if ids provided
            if ids:
                ids_batch.append(ids[i])

            if (i + 1) % batch_size == 0:
                # FIX for issue #507: Pass ids_batch to _insert_texts
                result_ids.extend(
                    self._insert_texts(
                        texts_batch, metadatas_batch, ids_batch if ids else None
                    )
                )
                texts_batch = []
                metadatas_batch = []
                ids_batch = []

        if texts_batch:
            # FIX for issue #507: Pass ids_batch to _insert_texts for remaining items
            result_ids.extend(
                self._insert_texts(
                    texts_batch, metadatas_batch, ids_batch if ids else None
                )
            )
        return result_ids

    # FIX for issue #507: Updated signature to accept ids parameter
    def _insert_texts(
        self,
        texts: List[str],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
    ) -> List:
        """Used to Load Documents into the collection

        Args:
            texts: The list of documents strings to load
            metadatas: The list of metadata objects associated with each document

        Returns:

        """
        # If the text is empty, then exit early
        if not texts:
            return []

        # Embed and create the documents
        embeddings = self._embedding.embed_documents(texts)
        to_insert = [
            {self._text_key: t, self._embedding_key: embedding, **m}
            for t, m, embedding in zip(texts, metadatas, embeddings)
        ]

        # FIX for issue #507: Set custom _id if ids are provided
        if ids:
            for i, doc in enumerate(to_insert):
                doc["_id"] = ids[i]

        # insert the documents in DocumentDB
        insert_result = self._collection.insert_many(to_insert)

        # FIX for issue #507: Return provided ids if available,
        # else return generated ids
        if ids:
            return ids
        else:
            return insert_result.inserted_ids

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        collection: Optional[Collection[DocumentDBDocumentType]] = None,
        **kwargs: Any,
    ) -> DocumentDBVectorSearch:
        if collection is None:
            raise ValueError("Must provide 'collection' named parameter.")
        vectorstore = cls(collection, embedding, **kwargs)
        vectorstore.add_texts(texts, metadatas=metadatas)
        return vectorstore

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        if ids is None:
            raise ValueError("No document ids provided to delete.")

        for document_id in ids:
            self.delete_document_by_id(document_id)
        return True

    # FIX for issue #507: Updated to handle both custom string IDs and ObjectId strings
    def delete_document_by_id(self, document_id: Optional[str] = None) -> None:
        """Removes a Specific Document by Id

        Args:
            document_id: The document identifier
        """
        try:
            from bson.objectid import ObjectId
        except ImportError as e:
            raise ImportError(
                "Unable to import bson, please install with `pip install bson`."
            ) from e
        if document_id is None:
            raise ValueError("No document id provided to delete.")

        # FIX for issue #507: Handle both ObjectId strings & custom IDs (e.g., SHA256)
        # Try to use ObjectId for 24-character hex strings (backward compatibility)
        # Otherwise use the ID as-is for custom IDs from Indexing API
        try:
            if len(document_id) == 24:
                # Attempt to convert to ObjectId for backward compatibility
                self._collection.delete_one({"_id": ObjectId(document_id)})
            else:
                # Use custom ID directly (e.g., SHA256 hash from Indexing API)
                self._collection.delete_one({"_id": document_id})
        except Exception:
            # If ObjectId conversion fails, try using the ID as-is
            self._collection.delete_one({"_id": document_id})

    def _similarity_search_without_score(
        self,
        embeddings: List[float],
        k: int = 4,
        ef_search: int = 40,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """Returns a list of documents.

        Args:
            embeddings: The query vector
            k: the number of documents to return
            ef_search: Specifies the size of the dynamic candidate list
                that HNSW index uses during search. A higher value of
                efSearch provides better recall at cost of speed.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.
        Returns:
            A list of documents closest to the query vector
        """
        # $match can't be null, so initializes to {} when None to avoid
        # "the match filter must be an expression in an object"
        if not filter:
            filter = {}
        pipeline: List[dict[str, Any]] = [
            {"$match": filter},
            {
                "$search": {
                    "vectorSearch": {
                        "vector": embeddings,
                        "path": self._embedding_key,
                        "similarity": self._similarity_type,
                        "k": k,
                        "efSearch": ef_search,
                    }
                },
            },
        ]

        cursor = self._collection.aggregate(pipeline)

        docs = []

        for res in cursor:
            text = res.pop(self._text_key)
            docs.append(Document(page_content=text, metadata=res))

        return docs

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        ef_search: int = 40,
        *,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        embeddings = self._embedding.embed_query(query)
        docs = self._similarity_search_without_score(
            embeddings=embeddings, k=k, ef_search=ef_search, filter=filter
        )
        return [doc for doc in docs]
