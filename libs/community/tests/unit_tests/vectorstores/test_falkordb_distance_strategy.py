"""
Unit tests for FalkorDBVector distance strategy handling.

These tests verify that the FalkorDB LangChain wrapper respects the
configured distance strategy when creating relationship indexes,
propagates a custom distance strategy when instantiating from
documents, and builds the correct distance function into the metadata
filter search query.

The tests use unittest.mock to avoid requiring a live FalkorDB
instance.  They focus on the behaviour of the wrapper itself.
"""

import pytest

pytest.importorskip("falkordb")

from typing import Any, List
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from langchain_community.vectorstores.falkordb_vector import (
    FalkorDBVector,
    IndexType,
)
from langchain_community.vectorstores.utils import DistanceStrategy


class DummyEmbeddings(Embeddings):
    """A minimal embeddings implementation for testing.

    This class implements the methods expected by FalkorDBVector
    but returns trivial fixed-size vectors so that tests can run
    without access to external embedding models.
    """

    def __init__(self, size: int = 2) -> None:
        self.size = size

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Return a distinct vector for each document; dimension is ``self.size``.
        return [[float(i + 1) for _ in range(self.size)] for i in range(len(texts))]

    def embed_query(self, text: str) -> List[float]:
        # Return a simple vector of the correct dimension for any query.
        return [1.0 for _ in range(self.size)]


def test_create_new_index_on_relationship_respects_strategy() -> None:
    """Ensure that create_new_index_on_relationship uses the configured metric."""
    # Mock graph and database; create_edge_vector_index should record its kwargs.
    fake_db = MagicMock()
    fake_graph = MagicMock()
    fake_graph._graph = fake_db
    fake_graph._driver = MagicMock()

    # Instantiate a FalkorDBVector with cosine distance
    store = FalkorDBVector(
        embedding=DummyEmbeddings(),
        graph=fake_graph,
        relation_type="REL",
        embedding_node_property="embedding",
        embedding_dimension=2,
        distance_strategy=DistanceStrategy.COSINE,
    )

    store.create_new_index_on_relationship()
    # Verify that the underlying DB method was called with similarity_function="cosine"
    assert fake_db.create_edge_vector_index.call_count == 1
    _, kwargs = fake_db.create_edge_vector_index.call_args
    assert kwargs["similarity_function"] == "cosine"


def test_from_documents_propagates_distance_strategy() -> None:
    """Ensure that from_documents forwards distance_strategy to the store."""
    fake_db = MagicMock()
    fake_graph = MagicMock()
    fake_graph._graph = fake_db
    fake_graph._driver = MagicMock()

    docs = [Document(page_content="alpha"), Document(page_content="beta")]
    store = FalkorDBVector.from_documents(
        documents=docs,
        embedding=DummyEmbeddings(),
        graph=fake_graph,
        embedding_dimension=2,
        node_label="Test",
        distance_strategy=DistanceStrategy.COSINE,
    )

    assert store._distance_strategy == DistanceStrategy.COSINE


def test_similarity_search_with_score_by_vector_uses_correct_distance() -> None:
    """Ensure metadata-filtered vector search uses the correct distance function."""
    # Prepare a store with cosine distance
    fake_db = MagicMock()
    fake_graph = MagicMock()
    fake_graph._graph = fake_db
    fake_graph._driver = MagicMock()

    store = FalkorDBVector(
        embedding=DummyEmbeddings(),
        graph=fake_graph,
        node_label="Chunk",
        embedding_node_property="embedding",
        embedding_dimension=2,
        distance_strategy=DistanceStrategy.COSINE,
    )
    # Manually set index type for query construction
    store._index_type = IndexType.NODE

    captured: dict[str, Any] = {}

    def fake_query(query: str, params: Any = None) -> List[Any]:
        captured["query"] = query
        return []

    # Patch the _query method to capture the query string
    store._query = fake_query  # type: ignore[assignment]

    # Perform a similarity search with a metadata filter;
    # query should contain cosine distance
    store.similarity_search_with_score_by_vector(
        embedding=[0.1, 0.2], k=1, filter={"lang": "en"}
    )
    assert "vec.cosineDistance" in captured["query"]

    # Repeat for Euclidean strategy
    store2 = FalkorDBVector(
        embedding=DummyEmbeddings(),
        graph=fake_graph,
        node_label="Chunk",
        embedding_node_property="embedding",
        embedding_dimension=2,
        distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
    )
    store2._index_type = IndexType.NODE

    captured2: dict[str, Any] = {}

    def fake_query2(query: str, params: Any = None) -> List[Any]:
        captured2["query"] = query
        return []

    store2._query = fake_query2  # type: ignore[assignment]

    store2.similarity_search_with_score_by_vector(
        embedding=[0.3, 0.4], k=1, filter={"lang": "en"}
    )
    assert "vec.euclideanDistance" in captured2["query"]
