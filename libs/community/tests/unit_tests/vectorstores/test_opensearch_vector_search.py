from typing import Any, Callable, Dict, List, Tuple

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from pytest import MonkeyPatch

from langchain_community.vectorstores.opensearch_vector_search import (
    HYBRID_SEARCH,
    OpenSearchVectorSearch,
)


class DummyEmbeddings(Embeddings):
    def __init__(self, dim: int = 3) -> None:
        self.dim = dim

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Return a constant vector per document to keep tests deterministic
        return [[0.0] * self.dim for _ in texts]

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return [1.0] * self.dim

    async def aembed_query(self, text: str) -> List[float]:
        return self.embed_query(text)


class ClientMock:
    def __init__(self) -> None:
        self.indices = type(
            "Indices",
            (),
            {
                "delete": lambda self, index: True,
                "exists": lambda self, index: False,
                "create": lambda self, index, body: True,
                "get": lambda self, index: True,
                "refresh": lambda self, index: True,
            },
        )()
        self.transport = type(
            "Transport",
            (),
            {
                "perform_request": lambda self, method, url, body=None, **kwargs: {
                    "hits": {
                        "hits": [
                            {
                                "_id": "1",
                                "_score": 0.9,
                                "_source": {
                                    "text": "doc",
                                    "metadata": {"a": 1},
                                    "vector_field": [0.1, 0.2, 0.3],
                                },
                            }
                        ]
                    }
                }
            },
        )()

    def search(self, **kwargs: Any) -> Dict[str, Any]:
        return self.transport.perform_request(
            method="GET", url="/_search", body=kwargs.get("body")
        )

    def bulk(self, **kwargs: Any) -> Dict[str, Any]:
        return {"items": [{"delete": {}}]}


class AsyncClientMock:
    def __init__(self) -> None:
        # independent async-compatible mock, not inheriting ClientMock
        # to avoid type override mismatches
        class ATransport:
            async def perform_request(
                self,
                method: str,
                url: str,
                body: Any | None = None,
                **kwargs: Any,
            ) -> Dict[str, Any]:
                return {
                    "hits": {
                        "hits": [
                            {
                                "_id": "1",
                                "_score": 0.9,
                                "_source": {
                                    "text": "doc",
                                    "metadata": {"a": 1},
                                    "vector_field": [0.1, 0.2, 0.3],
                                },
                            }
                        ]
                    }
                }

        self.transport = ATransport()

        class AIndices:
            async def get(self, index: str, **kwargs: Any) -> bool:
                return True

            async def create(
                self,
                index: str,
                body: Any | None = None,
                **kwargs: Any,
            ) -> bool:
                return True

            async def refresh(self, index: str, **kwargs: Any) -> bool:
                return True

        self.indices = AIndices()

    async def search(self, **kwargs: Any) -> Dict[str, Any]:
        return await self.transport.perform_request(
            method="GET", url="/_search", body=kwargs.get("body")
        )


@pytest.fixture(autouse=True)
def fake_opensearchpy(monkeypatch: MonkeyPatch) -> None:
    """Provide a minimal fake opensearchpy to satisfy imports in the vectorstore."""
    import sys
    import types

    opensearchpy = types.ModuleType("opensearchpy")
    exceptions = types.ModuleType("opensearchpy.exceptions")

    class NotFoundError(Exception):
        pass

    exceptions.NotFoundError = NotFoundError  # type: ignore[attr-defined]
    helpers = types.ModuleType("opensearchpy.helpers")

    def bulk(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        return {"items": []}

    async def async_bulk(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        return {"items": []}

    helpers.bulk = bulk  # type: ignore[attr-defined]
    helpers.async_bulk = async_bulk  # type: ignore[attr-defined]
    opensearchpy.exceptions = exceptions  # type: ignore[attr-defined]
    opensearchpy.helpers = helpers  # type: ignore[attr-defined]
    sys.modules["opensearchpy"] = opensearchpy
    sys.modules["opensearchpy.exceptions"] = exceptions
    sys.modules["opensearchpy.helpers"] = helpers


@pytest.fixture()
def patch_clients(
    monkeypatch: MonkeyPatch,
) -> Callable[[], Tuple[ClientMock, AsyncClientMock]]:
    def _patch(
        url: str = "http://localhost:9200",
    ) -> Tuple[ClientMock, AsyncClientMock]:
        client = ClientMock()
        async_client = AsyncClientMock()
        monkeypatch.setattr(
            "langchain_community.vectorstores.opensearch_vector_search._get_opensearch_client",
            lambda opensearch_url, **kwargs: client,
        )
        monkeypatch.setattr(
            "langchain_community.vectorstores.opensearch_vector_search._get_async_opensearch_client",
            lambda opensearch_url, **kwargs: async_client,
        )
        return client, async_client

    return _patch


def test_add_texts_and_similarity_search_basic(
    patch_clients: Callable[[], Tuple[ClientMock, AsyncClientMock]],
) -> None:
    patch_clients()
    embed = DummyEmbeddings(dim=3)
    vs = OpenSearchVectorSearch(
        opensearch_url="http://localhost:9200",
        index_name="test-index",
        embedding_function=embed,
        engine="nmslib",
    )

    ids = vs.add_texts(
        ["hola", "adios"], metadatas=[{"k": "v"}, {"x": 1}], ids=["id1", "id2"]
    )
    assert len(ids) == 2

    docs = vs.similarity_search("hola", k=1)
    assert isinstance(docs[0], Document)
    assert docs[0].page_content == "doc"


def test_similarity_search_with_score_by_vector_builds_query(
    patch_clients: Callable[[], Tuple[ClientMock, AsyncClientMock]],
) -> None:
    patch_clients()
    embed = DummyEmbeddings(dim=3)
    vs = OpenSearchVectorSearch(
        opensearch_url="http://localhost:9200",
        index_name="test-index",
        embedding_function=embed,
        engine="nmslib",
    )

    vec = [0.1, 0.2, 0.3]
    hits = vs._raw_similarity_search_with_score_by_vector(
        vec, k=2, search_type="approximate_search"
    )
    # ensure search invoked and returns hits
    assert isinstance(hits, list)
    assert hits and "_id" in hits[0]


def test_boolean_vs_efficient_filter_exclusive(
    patch_clients: Callable[[], Tuple[ClientMock, AsyncClientMock]],
) -> None:
    patch_clients()
    embed = DummyEmbeddings(dim=3)
    vs = OpenSearchVectorSearch("http://localhost:9200", "idx", embed)
    vec = [0.1, 0.2, 0.3]
    with pytest.raises(ValueError):
        vs._raw_similarity_search_with_score_by_vector(
            vec,
            k=2,
            search_type="approximate_search",
            boolean_filter={"term": {"x": 1}},
            efficient_filter={"term": {"y": 2}},
        )


def test_delete_requires_ids(
    patch_clients: Callable[[], Tuple[ClientMock, AsyncClientMock]],
) -> None:
    patch_clients()
    vs = OpenSearchVectorSearch("http://localhost:9200", "idx", DummyEmbeddings())
    with pytest.raises(ValueError):
        vs.delete(ids=None)


def test_index_exists_false(
    patch_clients: Callable[[], Tuple[ClientMock, AsyncClientMock]],
) -> None:
    client, _ = patch_clients()
    vs = OpenSearchVectorSearch("http://localhost:9200", "idx", DummyEmbeddings())
    assert vs.index_exists("idx") is False


def test_create_index_appx_mapping_and_duplicate(
    patch_clients: Callable[[], Tuple[ClientMock, AsyncClientMock]],
    monkeypatch: MonkeyPatch,
) -> None:
    client, _ = patch_clients()
    # make indices.exists return True to trigger error
    monkeypatch.setattr(client.indices, "exists", lambda index: True)
    vs = OpenSearchVectorSearch(
        "http://localhost:9200", "idx", DummyEmbeddings(), engine="nmslib"
    )
    with pytest.raises(RuntimeError):
        vs.create_index(dimension=3, index_name="idx")


def test_hybrid_search_requires_pipeline_and_text(
    patch_clients: Callable[[], Tuple[ClientMock, AsyncClientMock]],
) -> None:
    patch_clients()
    vs = OpenSearchVectorSearch("http://localhost:9200", "idx", DummyEmbeddings())
    with pytest.raises(ValueError):
        vs._raw_similarity_search_with_score_by_vector(
            [0.1, 0.2, 0.3], k=2, search_type=HYBRID_SEARCH, search_pipeline="pipe"
        )
    with pytest.raises(ValueError):
        vs._raw_similarity_search_with_score_by_vector(
            [0.1, 0.2, 0.3], k=2, search_type=HYBRID_SEARCH, query_text="hola"
        )


@pytest.mark.asyncio
async def test_async_similarity_search_with_score(
    patch_clients: Callable[[], Tuple[ClientMock, AsyncClientMock]],
) -> None:
    patch_clients()
    vs = OpenSearchVectorSearch("http://localhost:9200", "idx", DummyEmbeddings())
    docs_scores = await vs.asimilarity_search_with_score("hola", k=1)
    assert len(docs_scores) == 1
    doc, score = docs_scores[0]
    assert isinstance(doc, Document)
    assert isinstance(score, float)


@pytest.mark.asyncio
async def test_adelete_requires_ids(
    patch_clients: Callable[[], Tuple[ClientMock, AsyncClientMock]],
) -> None:
    patch_clients()
    vs = OpenSearchVectorSearch("http://localhost:9200", "idx", DummyEmbeddings())
    with pytest.raises(ValueError):
        await vs.adelete(ids=None)


def test_hits_to_documents_metadata_variants(
    patch_clients: Callable[[], Tuple[ClientMock, AsyncClientMock]],
) -> None:
    patch_clients()
    vs = OpenSearchVectorSearch("http://localhost:9200", "idx", DummyEmbeddings())
    hits = [
        {
            "_id": "x",
            "_score": 0.5,
            "_source": {"text": "t", "metadata": {"a": 1}, "other": {"b": 2}},
        }
    ]
    docs_scores = vs._hits_to_documents(hits, text_field="text", metadata_field="*")
    assert docs_scores[0][0].metadata.get("metadata") == {"a": 1}
    docs_scores = vs._hits_to_documents(
        hits, text_field="text", metadata_field="metadata"
    )
    assert docs_scores[0][0].metadata == {"a": 1}


def test_from_texts_uses_env(
    monkeypatch: MonkeyPatch,
    patch_clients: Callable[[], Tuple[ClientMock, AsyncClientMock]],
) -> None:
    patch_clients()
    monkeypatch.setenv("OPENSEARCH_URL", "http://localhost:9200")
    monkeypatch.setenv("OPENSEARCH_INDEX_NAME", "env-index")
    embed = DummyEmbeddings(dim=3)
    vs = OpenSearchVectorSearch.from_texts(["a", "b"], embed)
    assert isinstance(vs, OpenSearchVectorSearch)
    assert vs.index_name == "env-index"


@pytest.mark.asyncio
async def test_afrom_texts(
    monkeypatch: MonkeyPatch,
    patch_clients: Callable[[], Tuple[ClientMock, AsyncClientMock]],
) -> None:
    patch_clients()
    monkeypatch.setenv("OPENSEARCH_URL", "http://localhost:9200")
    monkeypatch.setenv("OPENSEARCH_INDEX_NAME", "env-index")
    embed = DummyEmbeddings(dim=3)
    vs = await OpenSearchVectorSearch.afrom_texts(["a", "b"], embed)
    assert isinstance(vs, OpenSearchVectorSearch)
    assert vs.index_name == "env-index"
