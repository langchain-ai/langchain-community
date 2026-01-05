"""Test AnalyticDB MySQL functionality.
1. create a AnalyticDB MySQL instance on alibabacloud.
2. connect to instance by mysql client (like DBeaver...), run sql only once:
    'create database vectorstore;'
3. shell input:
    export ADB_HOST=<...>
    export ADB_PORT=<...>
    export ADB_USER=<...>
    export ADB_PASSWORD=<...>
"""

import os
from typing import List

from langchain_core.documents import Document
from langchain_community.vectorstores.alibabacloud_opensearch import (
    AnalyticDBMySQL,
    AnalyticDBMySQLSettings,
)
from tests.integration_tests.vectorstores.fake_embeddings import FakeEmbeddings

ADB_TOKEN_COUNT = 1024

settings = AnalyticDBMySQLSettings()
settings.host = os.getenv("ADB_HOST", "localhost")
settings.port = int(os.getenv("ADB_PORT", "3306"))
settings.user = os.getenv("ADB_USER")
settings.password = os.getenv("ADB_PASSWORD")


class FakeEmbeddingsWithADB(FakeEmbeddings):
    """Fake embeddings functionality for testing."""

    def embed_documents(self, embedding_texts: List[str]) -> List[List[float]]:
        """Return simple embeddings."""
        return [
            [float(1.0)] * (ADB_TOKEN_COUNT - 1) + [float(i)]
            for i in range(len(embedding_texts))
        ]

    def embed_query(self, text: str) -> List[float]:
        """Return simple embeddings."""
        return [float(1.0)] * (ADB_TOKEN_COUNT - 1) + [float(texts.index(text))]


def test_analyticdb_mysql() -> None:
    texts = ["foo", "bar", "baz"]
    ids = ["id_foo", "id_bar", "id_baz"]
    vectorstore = AnalyticDBMySQL.from_texts(
        texts=texts,
        embedding=FakeEmbeddingsWithADB(),
        config=settings,
        text_ids=ids,
    )

    # similarity_search
    output = vectorstore.similarity_search(query="foo", k=1)
    assert output == [Document(page_content="foo")]

    # similarity_search_with_relevance_scores
    output = vectorstore.similarity_search_with_relevance_scores(query="foo", k=1)
    assert output == [(Document(page_content="foo"), 0.0)]

    # max_marginal_relevance_search
    output = vectorstore.max_marginal_relevance_search(query="foo", k=1, fetch_k=2)
    assert output == [Document(page_content="foo")]


def test_analyticdb_mysql_delete() -> None:
    texts = ["foo", "bar", "baz"]
    ids = ["id_foo", "id_bar", "id_baz"]
    vectorstore = AnalyticDBMySQL.from_texts(
        texts=texts,
        embedding=FakeEmbeddingsWithADB(),
        config=settings,
        text_ids=ids,
    )

    vectorstore.delete(ids=ids)
    output = vectorstore.similarity_search("foo", k=1)
    assert output == []
