from typing import Generator

import pytest

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from langchain_community.vectorstores.surrealdb import SurrealDBStore
from langchain_tests.integration_tests.vectorstores import VectorStoreIntegrationTests


class TestSurrealDB(VectorStoreIntegrationTests):
    @pytest.fixture
    def vectorstore(self) -> Generator[SurrealDBStore, None, None]:
        store = SurrealDBStore(embedding_function=self.get_embeddings(), db_user="root", db_pass="root", db="test",
                               ns="test")
        store.delete()
        try:
            yield store
        finally:
            store.delete()


def test_from_documents(embedding_openai: Embeddings) -> None:
    """Test end to end construction and search."""
    documents = [
        Document(page_content="Dogs are tough.", metadata={"a": 1}),
        Document(page_content="Cats have fluff.", metadata={"b": 1}),
        Document(page_content="What is a sandwich?", metadata={"c": 1}),
        Document(page_content="That fence is purple.", metadata={"d": 1, "e": 2}),
    ]
    vectorstore = SurrealDBStore.from_documents(
        documents,
        embedding_openai,
    )
    output = vectorstore.similarity_search("Sandwich", k=1)
    assert output[0].page_content == "What is a sandwich?"
    assert output[0].metadata["c"] == 1
