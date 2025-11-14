"""Test Reindexer vector store."""

import tempfile
import uuid
from pathlib import Path

import pytest

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from langchain_community.vectorstores.reindexer import (
    DEFAULT_AUTOMATIC_DISK_DIR,
    MEMORY_DUMP_FILENAME,
)
from langchain_community.vectorstores import ReindexerVectorStore



class FakeEmbeddings(Embeddings):
    """Fake embeddings functionality for testing."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return simple embeddings.

        Embeddings encode each text as its index.
        """
        return [[float(1.0)] * 9 + [float(i)] for i in range(len(texts))]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return simple embeddings."""
        return self.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        """Return constant query embeddings.

        Embeddings are identical to embed_documents(texts)[0].
        Distance to each text will be that text's index,
        as it was passed to embed_documents.
        """
        return [float(1.0)] * 9 + [float(0.0)]

    async def aembed_query(self, text: str) -> list[float]:
        """Return constant query embeddings."""
        return self.embed_query(text)


@pytest.fixture
def temp_db_path():
    """Create a temporary database path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test_db"


@pytest.fixture
def embedding():
    """Create a fake embedding instance."""
    return FakeEmbeddings()


@pytest.fixture
def vector_store(embedding, temp_db_path):
    """Create a ReindexerVectorStore instance."""
    return ReindexerVectorStore(
        embedding=embedding,
        rx_connector_config={"dsn": f"builtin://{temp_db_path}"},
        rx_namespace="test_namespace",
    )


def test_add_documents(vector_store: ReindexerVectorStore) -> None:
    """Test adding documents."""
    documents = [
        Document(page_content="foo", metadata={"baz": "bar"}),
        Document(page_content="thud", metadata={"bar": "baz"}),
    ]
    ids = vector_store.add_documents(documents)
    assert len(ids) == 2
    assert all(isinstance(id_, str) for id_ in ids)


def test_add_documents_with_ids(vector_store: ReindexerVectorStore) -> None:
    """Test adding documents with custom IDs."""
    documents = [
        Document(page_content="foo", metadata={"baz": "bar"}),
        Document(page_content="thud", metadata={"bar": "baz"}),
    ]
    custom_ids = ["custom_id_1", "custom_id_2"]
    ids = vector_store.add_documents(documents, ids=custom_ids)
    assert ids == custom_ids


def test_similarity_search(vector_store: ReindexerVectorStore) -> None:
    """Test similarity search."""
    documents = [
        Document(page_content="foo", metadata={"baz": "bar"}),
        Document(page_content="thud", metadata={"bar": "baz"}),
        Document(page_content="qux", metadata={"qux": "quux"}),
    ]
    vector_store.add_documents(documents)
    results = vector_store.similarity_search("query", k=2)
    assert len(results) == 2
    assert all(isinstance(doc, Document) for doc in results)


def test_similarity_search_with_score(vector_store: ReindexerVectorStore) -> None:
    """Test similarity search with scores."""
    documents = [
        Document(page_content="foo", metadata={"baz": "bar"}),
        Document(page_content="thud", metadata={"bar": "baz"}),
    ]
    vector_store.add_documents(documents)
    results = vector_store.similarity_search_with_score("query", k=2)
    assert len(results) == 2
    assert all(isinstance(result, tuple) for result in results)
    assert all(isinstance(doc, Document) for doc, _ in results)
    assert all(isinstance(score, float) for _, score in results)


def test_similarity_search_by_vector(vector_store: ReindexerVectorStore) -> None:
    """Test similarity search by vector."""
    documents = [
        Document(page_content="foo", metadata={"baz": "bar"}),
        Document(page_content="thud", metadata={"bar": "baz"}),
    ]
    vector_store.add_documents(documents)
    embedding = vector_store.embedding.embed_query("query")
    results = vector_store.similarity_search_by_vector(embedding, k=2)
    assert len(results) == 2
    assert all(isinstance(doc, Document) for doc in results)


def test_delete(vector_store: ReindexerVectorStore) -> None:
    """Test deleting documents."""
    documents = [
        Document(page_content="foo", metadata={"baz": "bar"}),
        Document(page_content="thud", metadata={"bar": "baz"}),
        Document(page_content="i will be deleted :("),
    ]
    ids = vector_store.add_documents(documents)
    vector_store.delete(ids=[ids[2]])
    results = vector_store.similarity_search("query", k=10)
    assert len(results) == 2


def test_get_by_ids(vector_store: ReindexerVectorStore) -> None:
    """Test getting documents by IDs."""
    documents = [
        Document(page_content="foo", metadata={"baz": "bar"}),
        Document(page_content="thud", metadata={"bar": "baz"}),
    ]
    ids = vector_store.add_documents(documents)
    retrieved_docs = vector_store.get_by_ids(ids)
    assert len(retrieved_docs) == 2
    assert all(isinstance(doc, Document) for doc in retrieved_docs)
    assert retrieved_docs[0].page_content == "foo"
    assert retrieved_docs[1].page_content == "thud"


def test_from_texts(embedding, temp_db_path) -> None:
    """Test creating vector store from texts."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"meta": i} for i in range(len(texts))]
    vector_store = ReindexerVectorStore.from_texts(
        texts=texts,
        embedding=embedding,
        metadatas=metadatas,
        rx_connector_config={"dsn": f"builtin://{temp_db_path}"},
        rx_namespace="test_from_texts",
    )
    results = vector_store.similarity_search("query", k=3)
    assert len(results) == 3


def test_embeddings_property(vector_store: ReindexerVectorStore) -> None:
    """Test embeddings property."""
    assert vector_store.embeddings is not None
    assert isinstance(vector_store.embeddings, Embeddings)


def test_similarity_search_with_filter(vector_store: ReindexerVectorStore) -> None:
    """Test similarity search with metadata filter."""
    documents = [
        Document(page_content="foo", metadata={"category": "A", "value": 1}),
        Document(page_content="bar", metadata={"category": "B", "value": 2}),
        Document(page_content="baz", metadata={"category": "A", "value": 3}),
    ]
    vector_store.add_documents(documents)
    results = vector_store.similarity_search("query", k=10, filter={"category": "A"})
    assert len(results) == 2
    assert all(doc.metadata.get("category") == "A" for doc in results)


def test_similarity_search_with_callable_filter(
    vector_store: ReindexerVectorStore,
) -> None:
    """Test similarity search with callable filter."""
    documents = [
        Document(page_content="foo", metadata={"value": 1}),
        Document(page_content="bar", metadata={"value": 2}),
        Document(page_content="baz", metadata={"value": 3}),
    ]
    vector_store.add_documents(documents)

    def filter_func(doc: Document) -> bool:
        return doc.metadata.get("value", 0) > 1

    results = vector_store.similarity_search("query", k=10, filter=filter_func)
    assert len(results) == 2
    assert all(doc.metadata.get("value", 0) > 1 for doc in results)


def test_max_marginal_relevance_search(vector_store: ReindexerVectorStore) -> None:
    """Test max marginal relevance search."""
    documents = [
        Document(page_content="foo", metadata={"id": 1}),
        Document(page_content="bar", metadata={"id": 2}),
        Document(page_content="baz", metadata={"id": 3}),
    ]
    vector_store.add_documents(documents)
    results = vector_store.max_marginal_relevance_search("query", k=2, fetch_k=3)
    assert len(results) == 2
    assert all(isinstance(doc, Document) for doc in results)


def test_max_marginal_relevance_search_by_vector(
    vector_store: ReindexerVectorStore,
) -> None:
    """Test max marginal relevance search by vector."""
    documents = [
        Document(page_content="foo", metadata={"id": 1}),
        Document(page_content="bar", metadata={"id": 2}),
    ]
    vector_store.add_documents(documents)
    embedding = [float(1.0)] * 9 + [float(0.0)]
    results = vector_store.max_marginal_relevance_search_by_vector(
        embedding, k=2, fetch_k=2
    )
    assert len(results) == 2
    assert all(isinstance(doc, Document) for doc in results)


@pytest.mark.asyncio
async def test_aadd_documents(vector_store: ReindexerVectorStore) -> None:
    """Test async adding documents."""
    documents = [
        Document(page_content="foo", metadata={"baz": "bar"}),
        Document(page_content="thud", metadata={"bar": "baz"}),
    ]
    ids = await vector_store.aadd_documents(documents)
    assert len(ids) == 2
    assert all(isinstance(id_, str) for id_ in ids)


@pytest.mark.asyncio
async def test_adelete(vector_store: ReindexerVectorStore) -> None:
    """Test async deleting documents."""
    documents = [
        Document(page_content="foo", metadata={"baz": "bar"}),
        Document(page_content="thud", metadata={"bar": "baz"}),
        Document(page_content="i will be deleted :("),
    ]
    ids = vector_store.add_documents(documents)
    await vector_store.adelete(ids=[ids[2]])
    results = vector_store.similarity_search("query", k=10)
    assert len(results) == 2


@pytest.mark.asyncio
async def test_aget_by_ids(vector_store: ReindexerVectorStore) -> None:
    """Test async getting documents by IDs."""
    documents = [
        Document(page_content="foo", metadata={"baz": "bar"}),
        Document(page_content="thud", metadata={"bar": "baz"}),
    ]
    ids = vector_store.add_documents(documents)
    retrieved_docs = await vector_store.aget_by_ids(ids)
    assert len(retrieved_docs) == 2
    assert all(isinstance(doc, Document) for doc in retrieved_docs)


@pytest.mark.asyncio
async def test_asimilarity_search(vector_store: ReindexerVectorStore) -> None:
    """Test async similarity search."""
    documents = [
        Document(page_content="foo", metadata={"baz": "bar"}),
        Document(page_content="thud", metadata={"bar": "baz"}),
    ]
    vector_store.add_documents(documents)
    results = await vector_store.asimilarity_search("query", k=2)
    assert len(results) == 2
    assert all(isinstance(doc, Document) for doc in results)


@pytest.mark.asyncio
async def test_asimilarity_search_with_score(
    vector_store: ReindexerVectorStore,
) -> None:
    """Test async similarity search with scores."""
    documents = [
        Document(page_content="foo", metadata={"baz": "bar"}),
        Document(page_content="thud", metadata={"bar": "baz"}),
    ]
    vector_store.add_documents(documents)
    results = await vector_store.asimilarity_search_with_score("query", k=2)
    assert len(results) == 2
    assert all(isinstance(result, tuple) for result in results)
    assert all(isinstance(doc, Document) for doc, _ in results)


@pytest.mark.asyncio
async def test_asimilarity_search_by_vector(
    vector_store: ReindexerVectorStore,
) -> None:
    """Test async similarity search by vector."""
    documents = [
        Document(page_content="foo", metadata={"baz": "bar"}),
        Document(page_content="thud", metadata={"bar": "baz"}),
    ]
    vector_store.add_documents(documents)
    embedding = [float(1.0)] * 9 + [float(0.0)]
    results = await vector_store.asimilarity_search_by_vector(embedding, k=2)
    assert len(results) == 2
    assert all(isinstance(doc, Document) for doc in results)


@pytest.mark.asyncio
async def test_amax_marginal_relevance_search(
    vector_store: ReindexerVectorStore,
) -> None:
    """Test async max marginal relevance search."""
    documents = [
        Document(page_content="foo", metadata={"id": 1}),
        Document(page_content="bar", metadata={"id": 2}),
        Document(page_content="baz", metadata={"id": 3}),
    ]
    vector_store.add_documents(documents)
    results = await vector_store.amax_marginal_relevance_search(
        "query", k=2, fetch_k=3
    )
    assert len(results) == 2
    assert all(isinstance(doc, Document) for doc in results)


def test_save_local(vector_store: ReindexerVectorStore) -> None:
    """Test saving vector store configuration."""
    documents = [
        Document(page_content="foo", metadata={"baz": "bar"}),
    ]
    vector_store.add_documents(documents)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "saved_store"
        vector_store.save_local(save_path)
        assert (save_path / "reindexer_config.json").exists()


def test_load_local(embedding, temp_db_path) -> None:
    """Test loading vector store configuration."""
    # Create and save a vector store
    original_store = ReindexerVectorStore(
        embedding=embedding,
        rx_connector_config={"dsn": f"builtin://{temp_db_path}_original"},
        rx_namespace="test_save",
    )
    documents = [
        Document(page_content="foo", metadata={"baz": "bar"}),
    ]
    original_store.add_documents(documents)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "saved_store"
        original_store.save_local(save_path)
        original_store._database.close()

        # Load the vector store
        loaded_store = ReindexerVectorStore.load_local(
            save_path,
            embedding=embedding,
            rx_connector_config={"dsn": f"builtin://{temp_db_path}_loaded"},
        )
        assert loaded_store._rx_namespace == "test_save"
        assert loaded_store.embeddings is not None


def test_multithreading_validation(embedding) -> None:
    """Ensure multithreading parameter is validated."""
    with pytest.raises(ValueError):
        ReindexerVectorStore(
            embedding=embedding,
            multithreading=2,
            rx_connector_config={"dsn": "builtin://"},
            rx_namespace=f"invalid_mt_{uuid.uuid4().hex}",
        )


def test_save_load_memory_storage(embedding, tmp_path) -> None:
    """Save and load in-memory storage via file dump."""
    namespace = f"memory_ns_{uuid.uuid4().hex}"
    store = ReindexerVectorStore(
        embedding=embedding,
        rx_connector_config={"dsn": "builtin://"},
        rx_namespace=namespace,
    )
    documents = [
        Document(page_content="alpha", metadata={"idx": 1}),
        Document(page_content="beta", metadata={"idx": 2}),
    ]
    store.add_documents(documents)

    save_dir = tmp_path / "memory_store"
    store.save_local(save_dir)
    dump_file = save_dir / MEMORY_DUMP_FILENAME
    assert dump_file.exists()

    loaded_store = ReindexerVectorStore.load_local(
        save_dir, embedding=embedding
    )
    results = loaded_store.similarity_search("alpha", k=2)
    assert len(results) == 2
    assert {doc.page_content for doc in results} == {"alpha", "beta"}


def test_save_load_disk_auto_path(embedding, tmp_path, monkeypatch) -> None:
    """Save configuration for disk storage and reload using auto path."""
    namespace = f"disk_ns_{uuid.uuid4().hex}"
    monkeypatch.chdir(tmp_path)
    store = ReindexerVectorStore(
        embedding=embedding,
        rx_connector_config={"dsn": "builtin:///"},
        rx_namespace=namespace,
    )
    store.add_documents([Document(page_content="gamma", metadata={"idx": 3})])

    expected_path = (
        Path.cwd() / DEFAULT_AUTOMATIC_DISK_DIR / namespace
    )
    assert expected_path.exists()
    # Expect Reindexer to create internal files/directories for disk storage.
    assert any(expected_path.iterdir())

    save_dir = tmp_path / "disk_store"
    store.save_local(save_dir)
    store._database.close()

    loaded_store = ReindexerVectorStore.load_local(
        save_dir,
        embedding=embedding,
    )
    results = loaded_store.similarity_search("gamma", k=1)
    assert len(results) == 1
    assert results[0].page_content == "gamma"

