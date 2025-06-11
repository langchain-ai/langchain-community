from __future__ import annotations

import pickle
import random
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.utils import guard_import
from langchain_core.vectorstores import VectorStore

from langchain_community.vectorstores.utils import maximal_marginal_relevance

try:
    # Available from TileDB-Vector-Search ≥ 0.14
    from tiledb.vector_search.module import vspy  # pytype: disable=import-error
except Exception:  # pragma: no cover – TileDB may be missing in CI
    vspy = None  # type: ignore

_METRIC_STR_TO_ENUM = {
    "euclidean": vspy.DistanceMetric.L2 if vspy else None,
    "l2": vspy.DistanceMetric.L2 if vspy else None,
    "squared_l2": vspy.DistanceMetric.SUM_OF_SQUARES if vspy else None,
    "sum_of_squares": vspy.DistanceMetric.SUM_OF_SQUARES if vspy else None,
    "cosine": vspy.DistanceMetric.COSINE if vspy else None,
}

INDEX_METRICS = frozenset(_METRIC_STR_TO_ENUM.keys())
DEFAULT_METRIC = "euclidean"

DOCUMENTS_ARRAY_NAME = "documents"
VECTOR_INDEX_NAME = "vectors"
MAX_UINT64 = np.iinfo(np.dtype("uint64")).max
MAX_FLOAT_32 = np.finfo(np.dtype("float32")).max
MAX_FLOAT = sys.float_info.max


def dependable_tiledb_import() -> Any:
    """Import tiledb-vector-search if available, otherwise raise error."""
    return (
        guard_import("tiledb.vector_search"),
        guard_import("tiledb"),
    )


def get_vector_index_uri_from_group(group: Any) -> str:
    """Get the URI of the vector index."""
    return group[VECTOR_INDEX_NAME].uri


def get_documents_array_uri_from_group(group: Any) -> str:
    """Get the URI of the documents array from group."""
    return group[DOCUMENTS_ARRAY_NAME].uri


def get_vector_index_uri(uri: str) -> str:
    """Get the URI of the vector index."""
    return f"{uri}/{VECTOR_INDEX_NAME}"


def get_documents_array_uri(uri: str) -> str:
    """Get the URI of the documents array."""
    return f"{uri}/{DOCUMENTS_ARRAY_NAME}"


class TileDB(VectorStore):
    """TileDB vector store (LangChain wrapper)."""

    @staticmethod
    def _metric_to_enum(metric: str):
        """Translate user-friendly metric string to TileDB enum."""
        metric_lc = metric.lower()
        if metric_lc not in _METRIC_STR_TO_ENUM or _METRIC_STR_TO_ENUM[metric_lc] is None:
            raise ValueError(
                f"Unsupported distance metric '{metric}'. "
                f"Expected one of {sorted(INDEX_METRICS)}"
            )
        return _METRIC_STR_TO_ENUM[metric_lc]

    @staticmethod
    def _vector_dtype(arr_like) -> np.dtype:
        """Return the dtype we’ll store in the index."""
        if isinstance(arr_like, np.ndarray):
            return arr_like.dtype
        # if it’s a Python list, look at first element
        first = arr_like[0]
        if isinstance(first, (np.int8, int)):
            return np.int8
        if isinstance(first, (np.uint8,)):
            return np.uint8
        return np.float32

    def __init__(
        self,
        embedding: Embeddings,
        index_uri: str,
        metric: str = DEFAULT_METRIC,
        *,
        vector_index_uri: str = "",
        docs_array_uri: str = "",
        config: Optional[Mapping[str, Any]] = None,
        timestamp: Any = None,
        allow_dangerous_deserialization: bool = False,
        **kwargs: Any,
    ):
        if not allow_dangerous_deserialization:
            raise ValueError(
                "TileDB relies on pickle for serialization / deserialization. "
                "Set allow_dangerous_deserialization=True if you trust the data source."
            )

        self.embedding = embedding
        self.embedding_function = embedding.embed_query
        self.index_uri = index_uri
        self.metric = metric.lower()
        self.config = config

        tiledb_vs, tiledb = dependable_tiledb_import()
        with tiledb.scope_ctx(ctx_or_config=config):
            index_group = tiledb.Group(index_uri, "r")
            self.vector_index_uri = (
                vector_index_uri or get_vector_index_uri_from_group(index_group)
            )
            self.docs_array_uri = (
                docs_array_uri or get_documents_array_uri_from_group(index_group)
            )
            index_group.close()

            # Open the vector index; _dtype is exposed on the object
            self.vector_index = tiledb_vs.open_index(
                uri=self.vector_index_uri,
                timestamp=timestamp,
                config=config,
                **kwargs,
            )
            # TileDB >=0.15 exposes .dtype; fall back to float32 if older
            self._vec_dtype = getattr(self.vector_index, "dtype", np.float32)

            self.timestamp = timestamp

    @property
    def embeddings(self) -> Optional[Embeddings]:
        return self.embedding

    def _cast_query_vector(self, vec: List[float]) -> np.ndarray:
        """Cast a single query vector (list) to the index dtype."""
        return np.array([np.asarray(vec, dtype=self._vec_dtype)])

    def process_index_results(
        self,
        ids: List[int],
        scores: List[float],
        *,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        score_threshold: float = MAX_FLOAT,
    ) -> List[Tuple[Document, float]]:
        """Convert TileDB results into LangChain Documents with scores."""
        tiledb = guard_import("tiledb")

        docs, docs_array = [], tiledb.open(
            self.docs_array_uri, "r", timestamp=self.timestamp, config=self.config
        )

        for idx, score in zip(ids, scores):
            if (idx == 0 and score == 0) or (idx == MAX_UINT64 and score == MAX_FLOAT_32):
                continue

            doc = docs_array[idx]
            if doc is None or len(doc["text"]) == 0:
                raise ValueError(f"Could not find document for id {idx}")

            result_doc = Document(page_content=str(doc["text"][0]))
            pickled_md = doc.get("metadata")
            if pickled_md is not None:
                result_doc.metadata = pickle.loads(  # noqa: S301
                    np.asarray(pickled_md.tolist(), dtype=np.uint8).tobytes()
                )

            if filter:
                filter = {k: v if isinstance(v, list) else [v] for k, v in filter.items()}
                if all(result_doc.metadata.get(k) in v for k, v in filter.items()):
                    docs.append((result_doc, score))
            else:
                docs.append((result_doc, score))

        docs_array.close()
        docs = [(doc, score) for doc, score in docs if score <= score_threshold]
        return docs[:k]

    def similarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        *,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        fetch_k: int = 20,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        score_threshold = kwargs.pop("score_threshold", MAX_FLOAT)

        d, i = self.vector_index.query(
            self._cast_query_vector(embedding),
            k=k if filter is None else fetch_k,
            **kwargs,
        )
        return self.process_index_results(
            ids=i[0], scores=d[0], filter=filter, k=k, score_threshold=score_threshold
        )

    def similarity_search_with_score(
        self,
        query: str,
        *,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        fetch_k: int = 20,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        embedding = self.embedding_function(query)
        return self.similarity_search_with_score_by_vector(
            embedding, k=k, filter=filter, fetch_k=fetch_k, **kwargs
        )

    def max_marginal_relevance_search_with_score_by_vector(
        self,
        embedding: List[float],
        *,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        score_threshold = kwargs.pop("score_threshold", MAX_FLOAT)
        scores, indices = self.vector_index.query(
            self._cast_query_vector(embedding),
            k=fetch_k if filter is None else fetch_k * 2,
            **kwargs,
        )
        results = self.process_index_results(
            ids=indices[0],
            scores=scores[0],
            filter=filter,
            k=fetch_k if filter is None else fetch_k * 2,
            score_threshold=score_threshold,
        )

        # Re-embed retrieved docs for MMR
        embeddings = [
            self.embedding.embed_documents([doc.page_content])[0] for doc, _ in results
        ]
        mmr_idxs = maximal_marginal_relevance(
            np.asarray([embedding], dtype=self._vec_dtype),
            embeddings,
            k=k,
            lambda_mult=lambda_mult,
        )
        return [results[i] for i in mmr_idxs]

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        timestamp: int = 0,
        **kwargs: Any,
    ) -> List[str]:
        tiledb = guard_import("tiledb")

        embeddings = self.embedding.embed_documents(list(texts))
        if ids is None:
            ids = [str(random.randint(0, MAX_UINT64 - 1)) for _ in texts]

        external_ids = np.asarray(ids, dtype=np.uint64)
        vecs = np.empty(len(embeddings), dtype="O")
        for i, e in enumerate(embeddings):
            vecs[i] = np.asarray(e, dtype=self._vec_dtype)

        self.vector_index.update_batch(
            vectors=vecs,
            external_ids=external_ids,
            timestamp=None if timestamp == 0 else timestamp,
        )

        docs_data = {"text": np.asarray(texts)}
        if metadatas is not None:
            md_arr = np.empty(len(metadatas), dtype=object)
            for i, md in enumerate(metadatas):
                md_arr[i] = np.frombuffer(pickle.dumps(md), dtype=np.uint8)
            docs_data["metadata"] = md_arr

        with tiledb.open(
            self.docs_array_uri,
            "w",
            timestamp=None if timestamp == 0 else timestamp,
            config=self.config,
        ) as A:
            A[external_ids] = docs_data

        return ids

    def delete(
        self,
        ids: Optional[List[str]] = None,
        timestamp: int = 0,
        **kwargs: Any,
    ) -> Optional[bool]:
        external_ids = np.asarray(ids, dtype=np.uint64)
        self.vector_index.delete_batch(
            external_ids=external_ids,
            timestamp=None if timestamp == 0 else timestamp,
        )
        return True

    @classmethod
    def create(
        cls,
        index_uri: str,
        index_type: str,
        dimensions: int,
        vector_type: np.dtype,
        *,
        metric: str = DEFAULT_METRIC,
        metadatas: bool = True,
        config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        tiledb_vs, tiledb = dependable_tiledb_import()
        distance_metric_enum = cls._metric_to_enum(metric)

        with tiledb.scope_ctx(ctx_or_config=config):
            try:
                tiledb.group_create(index_uri)
            except tiledb.TileDBError as err:
                raise err

            group = tiledb.Group(index_uri, "w")
            vector_index_uri = get_vector_index_uri(group.uri)
            docs_uri = get_documents_array_uri(group.uri)

            if index_type == "FLAT":
                tiledb_vs.flat_index.create(
                    uri=vector_index_uri,
                    dimensions=dimensions,
                    vector_type=vector_type,
                    distance_metric=distance_metric_enum,
                    config=config,
                )
            elif index_type == "IVF_FLAT":
                tiledb_vs.ivf_flat_index.create(
                    uri=vector_index_uri,
                    dimensions=dimensions,
                    vector_type=vector_type,
                    distance_metric=distance_metric_enum,
                    config=config,
                )
            group.add(vector_index_uri, name=VECTOR_INDEX_NAME)

            dim = tiledb.Dim(
                name="id",
                domain=(0, MAX_UINT64 - 1),
                dtype=np.uint64,
            )
            dom = tiledb.Domain(dim)

            attrs = [tiledb.Attr(name="text", dtype="U1", var=True)]
            if metadatas:
                attrs.append(tiledb.Attr(name="metadata", dtype=np.uint8, var=True))

            tiledb.Array.create(
                docs_uri,
                tiledb.ArraySchema(
                    domain=dom,
                    sparse=True,
                    allows_duplicates=False,
                    attrs=attrs,
                ),
            )
            group.add(docs_uri, name=DOCUMENTS_ARRAY_NAME)
            group.close()

    @classmethod
    def __from(
        cls,
        texts: List[str],
        embeddings: List[List[float]],
        embedding: Embeddings,
        index_uri: str,
        *,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        metric: str = DEFAULT_METRIC,
        index_type: str = "FLAT",
        config: Optional[Mapping[str, Any]] = None,
        index_timestamp: int = 0,
        **kwargs: Any,
    ) -> "TileDB":
        if metric.lower() not in INDEX_METRICS:
            raise ValueError(
                f"Unsupported distance metric '{metric}'. "
                f"Expected one of {sorted(INDEX_METRICS)}"
            )

        input_vectors = np.asarray(embeddings)
        vector_type = input_vectors.dtype
        cls.create(
            index_uri=index_uri,
            index_type=index_type,
            dimensions=input_vectors.shape[1],
            vector_type=vector_type,
            metric=metric,
            metadatas=metadatas is not None,
            config=config,
        )

        tiledb_vs, tiledb = dependable_tiledb_import()
        with tiledb.scope_ctx(ctx_or_config=config):
            if not embeddings:
                raise ValueError("embeddings must be provided to build a TileDB index")

            vector_index_uri = get_vector_index_uri(index_uri)
            docs_uri = get_documents_array_uri(index_uri)

            if ids is None:
                ids = [str(random.randint(0, MAX_UINT64 - 1)) for _ in texts]
            external_ids = np.asarray(ids, dtype=np.uint64)

            tiledb_vs.ingestion.ingest(
                index_type=index_type,
                index_uri=vector_index_uri,
                input_vectors=input_vectors,
                external_ids=external_ids,
                index_timestamp=None if index_timestamp == 0 else index_timestamp,
                config=config,
                **kwargs,
            )

            with tiledb.open(docs_uri, "w") as A:
                data = {"text": np.asarray(texts)}
                if metadatas is not None:
                    md_attr = np.empty(len(metadatas), dtype=object)
                    for i, md in enumerate(metadatas):
                        md_attr[i] = np.frombuffer(pickle.dumps(md), dtype=np.uint8)
                    data["metadata"] = md_attr
                A[external_ids] = data

        return cls(
            embedding=embedding,
            index_uri=index_uri,
            metric=metric,
            config=config,
            **kwargs,
        )

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        *,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        metric: str = DEFAULT_METRIC,
        index_uri: str = "/tmp/tiledb_array",
        index_type: str = "FLAT",
        config: Optional[Mapping[str, Any]] = None,
        index_timestamp: int = 0,
        **kwargs: Any,
    ) -> "TileDB":
        embeddings = embedding.embed_documents(texts)
        return cls.__from(
            texts=texts,
            embeddings=embeddings,
            embedding=embedding,
            metadatas=metadatas,
            ids=ids,
            metric=metric,
            index_uri=index_uri,
            index_type=index_type,
            config=config,
            index_timestamp=index_timestamp,
            **kwargs,
        )

    @classmethod
    def from_embeddings(
        cls,
        text_embeddings: List[Tuple[str, List[float]]],
        embedding: Embeddings,
        index_uri: str,
        *,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        metric: str = DEFAULT_METRIC,
        index_type: str = "FLAT",
        config: Optional[Mapping[str, Any]] = None,
        index_timestamp: int = 0,
        **kwargs: Any,
    ) -> "TileDB":
        texts = [t[0] for t in text_embeddings]
        embeddings = [t[1] for t in text_embeddings]
        return cls.__from(
            texts=texts,
            embeddings=embeddings,
            embedding=embedding,
            metadatas=metadatas,
            ids=ids,
            metric=metric,
            index_uri=index_uri,
            index_type=index_type,
            config=config,
            index_timestamp=index_timestamp,
            **kwargs,
        )

    @classmethod
    def load(
        cls,
        index_uri: str,
        embedding: Embeddings,
        *,
        metric: str = DEFAULT_METRIC,
        config: Optional[Mapping[str, Any]] = None,
        timestamp: Any = None,
        **kwargs: Any,
    ) -> "TileDB":
        return cls(
            embedding=embedding,
            index_uri=index_uri,
            metric=metric,
            config=config,
            timestamp=timestamp,
            **kwargs,
        )

    def consolidate_updates(self, **kwargs: Any) -> None:
        self.vector_index = self.vector_index.consolidate_updates(**kwargs)
