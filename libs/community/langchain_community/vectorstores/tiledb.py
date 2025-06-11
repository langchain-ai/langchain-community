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

INDEX_METRICS = frozenset(["euclidean", "squared_l2", "cosine"])
DEFAULT_METRIC = "euclidean"

def _metric_to_enum(metric: str):
    tiledb_vs = guard_import("tiledb.vector_search")
    vspy = tiledb_vs.vspy
    return {
        "euclidean": vspy.DistanceMetric.L2,
        "squared_l2": vspy.DistanceMetric.SUM_OF_SQUARES,
        "cosine": vspy.DistanceMetric.COSINE,
    }[metric]

def _normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return (v / norm).astype(v.dtype)

_SUPPORTED_DTYPES = (np.float32, np.int8, np.uint8)
try:
    _HALF_DTYPES = (np.float16, np.dtype("bfloat16"))
except TypeError:
    _HALF_DTYPES = (np.float16,)

def _resolve_vector_dtype(requested: Optional[np.dtype], sample: np.ndarray) -> np.dtype:
    if requested is not None:
        if requested not in _SUPPORTED_DTYPES:
            raise ValueError
        return requested
    src = sample.dtype
    if src in _SUPPORTED_DTYPES:
        return src
    if src in _HALF_DTYPES:
        return np.float32
    raise ValueError

DOCUMENTS_ARRAY_NAME = "documents"
VECTOR_INDEX_NAME = "vectors"
MAX_UINT64 = np.iinfo(np.dtype("uint64")).max
MAX_FLOAT_32 = np.finfo(np.dtype("float32")).max
MAX_FLOAT = sys.float_info.max

def dependable_tiledb_import() -> Any:
    return (
        guard_import("tiledb.vector_search"),
        guard_import("tiledb"),
    )

def get_vector_index_uri_from_group(group: Any) -> str:
    return group[VECTOR_INDEX_NAME].uri

def get_documents_array_uri_from_group(group: Any) -> str:
    return group[DOCUMENTS_ARRAY_NAME].uri

def get_vector_index_uri(uri: str) -> str:
    return f"{uri}/{VECTOR_INDEX_NAME}"

def get_documents_array_uri(uri: str) -> str:
    return f"{uri}/{DOCUMENTS_ARRAY_NAME}"

class TileDB(VectorStore):
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
            raise ValueError
        if metric not in INDEX_METRICS:
            raise ValueError
        self.embedding = embedding
        self.embedding_function = embedding.embed_query
        self.index_uri = index_uri
        self.metric = metric
        self.config = config
        tiledb_vs, tiledb = dependable_tiledb_import()
        with tiledb.scope_ctx(ctx_or_config=config):
            index_group = tiledb.Group(self.index_uri, "r")
            self.vector_index_uri = (
                vector_index_uri or get_vector_index_uri_from_group(index_group)
            )
            self.docs_array_uri = (
                docs_array_uri or get_documents_array_uri_from_group(index_group)
            )
            index_group.close()
            group = tiledb.Group(self.vector_index_uri, "r")
            self.index_type = group.meta.get("index_type")
            group.close()
            self.timestamp = timestamp
            if self.index_type == "FLAT":
                self.vector_index = tiledb_vs.flat_index.FlatIndex(
                    uri=self.vector_index_uri,
                    config=self.config,
                    timestamp=self.timestamp,
                    **kwargs,
                )
            elif self.index_type == "IVF_FLAT":
                self.vector_index = tiledb_vs.ivf_flat_index.IVFFlatIndex(
                    uri=self.vector_index_uri,
                    config=self.config,
                    timestamp=self.timestamp,
                    **kwargs,
                )
        self._index_dtype = getattr(
            self.vector_index,
            "dtype",
            getattr(self.vector_index, "vector_type", np.float32),
        )

    @property
    def embeddings(self) -> Optional[Embeddings]:
        return self.embedding

    def process_index_results(
        self,
        ids: List[int],
        scores: List[float],
        *,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        score_threshold: float = MAX_FLOAT,
    ) -> List[Tuple[Document, float]]:
        tiledb = guard_import("tiledb")
        docs = []
        docs_array = tiledb.open(
            self.docs_array_uri, "r", timestamp=self.timestamp, config=self.config
        )
        for idx, score in zip(ids, scores):
            if idx == 0 and score == 0:
                continue
            if idx == MAX_UINT64 and score == MAX_FLOAT_32:
                continue
            doc = docs_array[idx]
            if doc is None or len(doc["text"]) == 0:
                raise ValueError
            pickled_metadata = doc.get("metadata")
            result_doc = Document(page_content=str(doc["text"][0]))
            if pickled_metadata is not None:
                metadata = pickle.loads(
                    np.array(pickled_metadata.tolist()).astype(np.uint8).tobytes()
                )
                result_doc.metadata = metadata
            if filter is not None:
                filter = {
                    key: [value] if not isinstance(value, list) else value
                    for key, value in filter.items()
                }
                if all(
                    result_doc.metadata.get(key) in value
                    for key, value in filter.items()
                ):
                    docs.append((result_doc, score))
            else:
                docs.append((result_doc, score))
        docs_array.close()
        docs = [(doc, score) for doc, score in docs if score <= score_threshold]
        return docs[:k]

    def _prepare_query_vector(self, v: List[float]) -> np.ndarray:
        vec = np.asarray(v)
        if vec.dtype in _HALF_DTYPES and self._index_dtype == np.float32:
            vec = vec.astype(np.float32)
        return vec.reshape(1, -1).astype(self._index_dtype, copy=False)

    def similarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        *,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        fetch_k: int = 20,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        if "score_threshold" in kwargs:
            score_threshold = kwargs.pop("score_threshold")
        else:
            score_threshold = MAX_FLOAT
        q = self._prepare_query_vector(embedding)
        d, i = self.vector_index.query(
            q,
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
            embedding,
            k=k,
            filter=filter,
            fetch_k=fetch_k,
            **kwargs,
        )

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        fetch_k: int = 20,
        **kwargs: Any,
    ) -> List[Document]:
        docs_and_scores = self.similarity_search_with_score_by_vector(
            embedding,
            k=k,
            filter=filter,
            fetch_k=fetch_k,
            **kwargs,
        )
        return [doc for doc, _ in docs_and_scores]

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        fetch_k: int = 20,
        **kwargs: Any,
    ) -> List[Document]:
        docs_and_scores = self.similarity_search_with_score(
            query, k=k, filter=filter, fetch_k=fetch_k, **kwargs
        )
        return [doc for doc, _ in docs_and_scores]

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
        if "score_threshold" in kwargs:
            score_threshold = kwargs.pop("score_threshold")
        else:
            score_threshold = MAX_FLOAT
        q = self._prepare_query_vector(embedding)
        scores, indices = self.vector_index.query(
            q,
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
        embeddings = [
            self.embedding.embed_documents([doc.page_content])[0] for doc, _ in results
        ]
        if self.metric == "cosine" and embeddings:
            embeddings = _normalize(np.vstack(embeddings))
        mmr_selected = maximal_marginal_relevance(
            np.array([embedding], dtype=np.float32),
            embeddings,
            k=k,
            lambda_mult=lambda_mult,
        )
        return [results[i] for i in mmr_selected]

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        docs_and_scores = self.max_marginal_relevance_search_with_score_by_vector(
            embedding,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
            **kwargs,
        )
        return [doc for doc, _ in docs_and_scores]

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        embedding = self.embedding_function(query)
        return self.max_marginal_relevance_search_by_vector(
            embedding,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
            **kwargs,
        )

    @classmethod
    def create(
        cls,
        index_uri: str,
        index_type: str,
        dimensions: int,
        vector_type: np.dtype,
        *,
        metadatas: bool = True,
        config: Optional[Mapping[str, Any]] = None,
        metric: str = DEFAULT_METRIC,
    ) -> None:
        tiledb_vs, tiledb = dependable_tiledb_import()
        distance_enum = _metric_to_enum(metric)
        with tiledb.scope_ctx(ctx_or_config=config):
            try:
                tiledb.group_create(index_uri)
            except tiledb.TileDBError as err:
                raise err
            group = tiledb.Group(index_uri, "w")
            vector_index_uri = get_vector_index_uri(group.uri)
            docs_uri = get_documents_array_uri(group.uri)
            create_kwargs = dict(
                uri=vector_index_uri,
                dimensions=dimensions,
                vector_type=vector_type,
                config=config,
                distance_metric=distance_enum,
            )
            if index_type == "FLAT":
                tiledb_vs.flat_index.create(**create_kwargs)
            elif index_type == "IVF_FLAT":
                tiledb_vs.ivf_flat_index.create(**create_kwargs)
            group.add(vector_index_uri, name=VECTOR_INDEX_NAME)
            dim = tiledb.Dim(
                name="id",
                domain=(0, MAX_UINT64 - 1),
                dtype=np.dtype(np.uint64),
            )
            dom = tiledb.Domain(dim)
            text_attr = tiledb.Attr(name="text", dtype=np.dtype("U1"), var=True)
            attrs = [text_attr]
            if metadatas:
                metadata_attr = tiledb.Attr(name="metadata", dtype=np.uint8, var=True)
                attrs.append(metadata_attr)
            schema = tiledb.ArraySchema(
                domain=dom,
                sparse=True,
                allows_duplicates=False,
                attrs=attrs,
            )
            tiledb.Array.create(docs_uri, schema)
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
        vector_dtype: Optional[np.dtype] = None,
        config: Optional[Mapping[str, Any]] = None,
        index_timestamp: int = 0,
        **kwargs: Any,
    ) -> "TileDB":
        if metric not in INDEX_METRICS:
            raise ValueError
        vector_dtype = _resolve_vector_dtype(vector_dtype, np.asarray(embeddings[0]))
        input_vectors = np.asarray(embeddings, dtype=vector_dtype)
        cls.create(
            index_uri=index_uri,
            index_type=index_type,
            dimensions=input_vectors.shape[1],
            vector_type=vector_dtype,
            metadatas=metadatas is not None,
            config=config,
            metric=metric,
        )
        tiledb_vs, tiledb = dependable_tiledb_import()
        with tiledb.scope_ctx(ctx_or_config=config):
            if not embeddings:
                raise ValueError
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
                index_timestamp=index_timestamp if index_timestamp != 0 else None,
                config=config,
                distance_metric=_metric_to_enum(metric),
                **kwargs,
            )
            with tiledb.open(docs_uri, "w") as A:
                data = {"text": np.asarray(texts)}
                if metadatas is not None:
                    metadata_attr = np.empty([len(metadatas)], dtype=object)
                    for i, md in enumerate(metadatas):
                        metadata_attr[i] = np.frombuffer(
                            pickle.dumps(md), dtype=np.uint8
                        )
                    data["metadata"] = metadata_attr
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
        vector_dtype: Optional[np.dtype] = None,
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
            vector_dtype=vector_dtype,
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
        vector_dtype: Optional[np.dtype] = None,
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
            vector_dtype=vector_dtype,
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
        embeddings_np = np.asarray(embeddings)
        target_dtype = self._index_dtype
        if embeddings_np.dtype in _HALF_DTYPES and target_dtype == np.float32:
            embeddings_np = embeddings_np.astype(np.float32)
        vectors = embeddings_np.astype(target_dtype)
        if ids is None:
            ids = [str(random.randint(0, MAX_UINT64 - 1)) for _ in texts]
        external_ids = np.asarray(ids, dtype=np.uint64)
        vectors_object = np.empty((len(vectors)), dtype="O")
        vectors_object[:] = [v for v in vectors]
        self.vector_index.update_batch(
            vectors=vectors_object,
            external_ids=external_ids,
            timestamp=timestamp if timestamp != 0 else None,
        )
        docs = {"text": np.asarray(texts)}
        if metadatas is not None:
            metadata_attr = np.empty([len(metadatas)], dtype=object)
            for i, md in enumerate(metadatas):
                metadata_attr[i] = np.frombuffer(pickle.dumps(md), dtype=np.uint8)
            docs["metadata"] = metadata_attr
        docs_array = tiledb.open(
            self.docs_array_uri,
            "w",
            timestamp=timestamp if timestamp != 0 else None,
            config=self.config,
        )
        docs_array[external_ids] = docs
        docs_array.close()
        return ids

    def delete(
        self, ids: Optional[List[str]] = None, timestamp: int = 0, **kwargs: Any
    ) -> Optional[bool]:
        external_ids = np.asarray(ids, dtype=np.uint64)
        self.vector_index.delete_batch(
            external_ids=external_ids,
            timestamp=timestamp if timestamp != 0 else None,
        )
        return True

    def consolidate_updates(self, **kwargs: Any) -> None:
        self.vector_index = self.vector_index.consolidate_updates(**kwargs)
