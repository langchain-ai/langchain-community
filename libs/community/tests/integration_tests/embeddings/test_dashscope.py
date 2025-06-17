"""Test dashscope embeddings."""

import numpy as np

from langchain_community.embeddings.dashscope import DashScopeEmbeddings

MODELS = [
    ("text-embedding-v1", 1536),
    ("text-embedding-v2", 1536),
    ("text-embedding-v3", 1024),
    ("text-embedding-v4", 1024),
]

def test_dashscope_embedding_documents(model: str, dimensions: int) -> None:
    """Test dashscope embeddings."""
    documents = ["foo bar"]
    embedding = DashScopeEmbeddings(model=model)
    output = embedding.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) == dimensions


def test_dashscope_embedding_documents_multiple(model: str, dimensions: int) -> None:
    """Test dashscope embeddings."""
    documents = [
        "foo bar",
        "bar foo",
        "foo",
        "foo0",
        "foo1",
        "foo2",
        "foo3",
        "foo4",
        "foo5",
        "foo6",
        "foo7",
        "foo8",
        "foo9",
        "foo10",
        "foo11",
        "foo12",
        "foo13",
        "foo14",
        "foo15",
        "foo16",
        "foo17",
        "foo18",
        "foo19",
        "foo20",
        "foo21",
        "foo22",
        "foo23",
        "foo24",
    ]
    embedding = DashScopeEmbeddings(model=model)
    output = embedding.embed_documents(documents)
    assert len(output) == 28
    assert len(output[0]) == dimensions
    assert len(output[1]) == dimensions
    assert len(output[2]) == dimensions


def test_dashscope_embedding_query(model: str, dimensions: int) -> None:
    """Test dashscope embeddings."""
    document = "foo bar"
    embedding = DashScopeEmbeddings(model=model)
    output = embedding.embed_query(document)
    assert len(output) == dimensions


def test_dashscope_embedding_with_empty_string(model: str, dimensions: int) -> None:
    """Test dashscope embeddings with empty string."""
    import dashscope

    document = ["", "abc"]
    embedding = DashScopeEmbeddings(model=model)
    output = embedding.embed_documents(document)
    assert len(output) == 2
    assert len(output[0]) == dimensions
    expected_output = dashscope.TextEmbedding.call(
        input="", model=model, text_type="document"
    ).output["embeddings"][0]["embedding"]
    assert np.allclose(output[0], expected_output)
    assert len(output[1]) == dimensions


if __name__ == "__main__":
    for model, dimensions in MODELS:
        test_dashscope_embedding_documents(model, dimensions)
        test_dashscope_embedding_documents_multiple(model, dimensions)
        test_dashscope_embedding_query(model, dimensions)
        test_dashscope_embedding_with_empty_string(model, dimensions)
