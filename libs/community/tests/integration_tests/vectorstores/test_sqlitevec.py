from typing import List, Optional

import pytest
from langchain_core.documents import Document

from langchain_community.vectorstores import SQLiteVec
from tests.integration_tests.vectorstores.fake_embeddings import (
    FakeEmbeddings,
    fake_texts,
)


def _sqlite_vec_from_texts(
    metadatas: Optional[List[dict]] = None, drop: bool = True
) -> SQLiteVec:
    return SQLiteVec.from_texts(
        fake_texts,
        FakeEmbeddings(),
        metadatas=metadatas,
        table="test",
        db_file=":memory:",
    )


@pytest.mark.requires("sqlite-vec")
def test_sqlitevec() -> None:
    """Test end to end construction and search."""
    docsearch = _sqlite_vec_from_texts()
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo", metadata={})]


@pytest.mark.requires("sqlite-vec")
def test_sqlitevec_with_score() -> None:
    """Test end to end construction and search with scores and IDs."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = _sqlite_vec_from_texts(metadatas=metadatas)
    output = docsearch.similarity_search_with_score("foo", k=3)
    docs = [o[0] for o in output]
    distances = [o[1] for o in output]
    assert docs == [
        Document(page_content="foo", metadata={"page": 0}),
        Document(page_content="bar", metadata={"page": 1}),
        Document(page_content="baz", metadata={"page": 2}),
    ]
    assert distances[0] < distances[1] < distances[2]


@pytest.mark.requires("sqlite-vec")
def test_sqlitevec_add_extra() -> None:
    """Test end to end construction and MRR search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = _sqlite_vec_from_texts(metadatas=metadatas)
    docsearch.add_texts(texts, metadatas)
    output = docsearch.similarity_search("foo", k=10)
    assert len(output) == 6


@pytest.mark.requires("sqlite-vec")
def test_sqlitevec_filter_equality() -> None:
    """Test similarity search with a simple equality metadata filter."""
    metadatas = [
        {"category": "a", "page": 0},
        {"category": "b", "page": 1},
        {"category": "a", "page": 2},
    ]
    docsearch = _sqlite_vec_from_texts(metadatas=metadatas)
    output = docsearch.similarity_search("foo", k=3, filter={"category": "a"})
    assert len(output) == 2
    assert all(doc.metadata["category"] == "a" for doc in output)


@pytest.mark.requires("sqlite-vec")
def test_sqlitevec_filter_with_score() -> None:
    """Test similarity_search_with_score respects metadata filter."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = _sqlite_vec_from_texts(metadatas=metadatas)
    output = docsearch.similarity_search_with_score("foo", k=3, filter={"page": 0})
    assert len(output) == 1
    doc, _distance = output[0]
    assert doc.page_content == "foo"
    assert doc.metadata == {"page": 0}


@pytest.mark.requires("sqlite-vec")
def test_sqlitevec_filter_operator_gt() -> None:
    """Test similarity search with a $gt operator filter."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"score": i} for i in range(len(texts))]
    docsearch = _sqlite_vec_from_texts(metadatas=metadatas)
    output = docsearch.similarity_search("foo", k=3, filter={"score": {"$gt": 0}})
    assert len(output) == 2
    assert all(doc.metadata["score"] > 0 for doc in output)


@pytest.mark.requires("sqlite-vec")
def test_sqlitevec_filter_in_operator() -> None:
    """Test similarity search with the $in operator."""
    metadatas = [
        {"color": "red"},
        {"color": "green"},
        {"color": "blue"},
    ]
    docsearch = _sqlite_vec_from_texts(metadatas=metadatas)
    output = docsearch.similarity_search(
        "foo", k=3, filter={"color": {"$in": ["red", "blue"]}}
    )
    assert len(output) == 2
    colors = {doc.metadata["color"] for doc in output}
    assert colors == {"red", "blue"}


@pytest.mark.requires("sqlite-vec")
def test_sqlitevec_filter_excludes_all() -> None:
    """Test that a filter matching nothing returns an empty list."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = _sqlite_vec_from_texts(metadatas=metadatas)
    output = docsearch.similarity_search("foo", k=3, filter={"page": 999})
    assert output == []


@pytest.mark.requires("sqlite-vec")
def test_sqlitevec_filter_no_filter_unchanged() -> None:
    """Test that omitting filter still works the same as before."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = _sqlite_vec_from_texts(metadatas=metadatas)
    output = docsearch.similarity_search("foo", k=3)
    assert len(output) == 3


@pytest.mark.requires("sqlite-vec")
def test_sqlitevec_search_multiple_tables() -> None:
    """Test end to end construction and search with multiple tables."""
    docsearch_1 = SQLiteVec.from_texts(
        fake_texts,
        FakeEmbeddings(),
        table="table_1",
        db_file=":memory:",  ## change to local storage for testing
    )

    docsearch_2 = SQLiteVec.from_texts(
        fake_texts,
        FakeEmbeddings(),
        table="table_2",
        db_file=":memory:",
    )

    output_1 = docsearch_1.similarity_search("foo", k=1)
    output_2 = docsearch_2.similarity_search("foo", k=1)

    assert output_1 == [Document(page_content="foo", metadata={})]
    assert output_2 == [Document(page_content="foo", metadata={})]
