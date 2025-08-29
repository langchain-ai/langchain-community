import os
import time
import pytest
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.vectorx import VectorXVectorStore


# ---- Skip tests if no API key ----
VECTORX_API_KEY = os.getenv("VECTORX_API_KEY")
pytestmark = pytest.mark.skipif(
    not VECTORX_API_KEY, reason="Missing VECTORX_API_KEY environment variable."
)


@pytest.fixture(scope="module")
def vectorx_store ():
    """Fixture to create and clean up a VectorXVectorStore."""
    from vecx.vectorx import VectorX

    vx = VectorX(token=VECTORX_API_KEY)
    encryption_key = vx.generate_key()

    # Create unique test index
    timestamp = int(time.time())
    test_index_name = f"test_langchain_index_{timestamp}"
    dimension = 384
    space_type = "cosine"

    embed_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )

    store = VectorXVectorStore.from_params(
        embedding=embed_model,
        api_token=VECTORX_API_KEY,
        index_name=test_index_name,
        encryption_key=encryption_key,
        dimension=dimension,
        space_type=space_type,
    )

    # Load test data
    test_texts = [
        "Python is a high-level, interpreted programming language known for its readability and simplicity.",
        "JavaScript is a scripting language that enables interactive web pages and is an essential part of web applications.",
        "Rust provides memory safety without garbage collection using ownership system.",
        "Machine learning is a subset of artificial intelligence that provides systems the ability to automatically learn and improve from experience.",
        "Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning.",
        "Vector databases are specialized database systems designed to store and query high-dimensional vectors for similarity search.",
        "Time-series databases are optimized for sequential temporal data storage.",
        "Building a real-time ML pipeline with Python and Vector DB.",
        "Implementing secure encryption in distributed databases.",
    ]

    test_metadatas = [
        {"category": "programming", "language": "python", "difficulty": "beginner", "doc_id": "doc1"},
        {"category": "programming", "language": "javascript", "difficulty": "intermediate", "doc_id": "doc2"},
        {"category": "programming", "language": "rust", "difficulty": "advanced", "doc_id": "doc3"},
        {"category": "ai", "field": "machine_learning", "difficulty": "intermediate", "doc_id": "doc4"},
        {"category": "ai", "field": "deep_learning", "difficulty": "advanced", "doc_id": "doc5"},
        {"category": "database", "type": "vector", "feature": "similarity_search", "doc_id": "doc6"},
        {"category": "database", "type": "time_series", "feature": "temporal_storage", "doc_id": "doc7"},
        {"category": ["programming", "ai", "database"], "languages": ["python"], "technologies": ["ml", "vector_db"], "difficulty": "advanced", "doc_id": "doc8"},
        {"category": ["programming", "database", "security"], "field": "cryptography", "difficulty": "advanced", "feature": "encryption", "doc_id": "doc9"},
    ]

    store.add_texts(texts=test_texts, metadatas=test_metadatas)

    yield store

    # Cleanup
    try:
        vx.delete_index(name=test_index_name)
        print(f"Deleted test index: {test_index_name}")
    except Exception as e:
        if "not found" not in str(e).lower():
            print(f"Error deleting test index {test_index_name}: {e}")


# ---- Tests ----
def test_create_vector_store_from_params(vectorx_store):
    assert vectorx_store is not None
    assert vectorx_store._vectorx_index.name.startswith("test_langchain_index_")


def test_basic_query(vectorx_store):
    results = vectorx_store.similarity_search("What is Python?", k=2)
    assert len(results) > 0


def test_similarity_search_with_score(vectorx_store):
    results = vectorx_store.similarity_search_with_score("What is Rust?", k=2)
    assert all(isinstance(score, float) for _, score in results)


def test_single_filter(vectorx_store):
    results = vectorx_store.similarity_search(
        "Programming languages", k=3,
        filter={"category": {"$eq": "programming"}},
    )
    assert len(results) > 0
    for doc in results:
        assert "programming" in str(doc.metadata.get("category", "")).lower()


def test_invalid_filter(vectorx_store):
    results = vectorx_store.similarity_search(
        "Random query", k=2,
        filter={"non_existent": {"$eq": "something"}},
    )
    assert len(results) == 0


def test_delete_by_filter_and_verify(vectorx_store):
    filter_to_delete = {"category": {"$in": ["programming"]}}
    vectorx_store.delete(filter=filter_to_delete)
    results = vectorx_store.similarity_search("JavaScript programming", k=2)
    for doc in results:
        assert "programming" not in str(doc.metadata.get("category", "")).lower()


def test_from_texts():
    """Test the from_texts classmethod directly."""
    from vecx.vectorx import VectorX

    vx = VectorX(token=VECTORX_API_KEY)
    encryption_key = vx.generate_key()

    timestamp = int(time.time())
    test_index_name = f"test_from_texts_index_{timestamp}"

    embed_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )

    texts = ["Quick test doc 1", "Quick test doc 2"]
    metadatas = [{"category": "temp1"}, {"category": "temp2"}]

    store = VectorXVectorStore.from_texts(
        texts=texts,
        embedding=embed_model,
        api_token=VECTORX_API_KEY,
        index_name=test_index_name,
        encryption_key=encryption_key,
        metadatas=metadatas,
        dimension=384,
    )

    results = store.similarity_search("Quick test doc", k=2)
    assert len(results) == 2

    # cleanup
    try:
        vx.delete_index(name=test_index_name)
    except Exception:
        pass
