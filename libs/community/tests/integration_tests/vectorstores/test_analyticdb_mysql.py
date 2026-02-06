"""Test AlibabaCloud AnalyticDB MySQL functionality.
1. create a AlibabaCloud AnalyticDB MySQL instance (https://adb.console.aliyun.com/).
2. connect to instance by mysql client, running:
    'create database vectorstore;'
3. shell running:
    export ADB_HOST=<...>
    export ADB_PORT=<...>
    export ADB_USER=<...>
    export ADB_PASSWORD=<...>
"""

import os
from typing import List

from langchain_core.documents import Document

from langchain_community.vectorstores.analyticdb_mysql import (
    AnalyticDBMySQL,
    AnalyticDBMySQLSettings,
)
from tests.integration_tests.vectorstores.fake_embeddings import FakeEmbeddings

ADB_TOKEN_COUNT = 1024


class FakeEmbeddingsWithADB(FakeEmbeddings):
    """Fake embeddings functionality for testing."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Return simple embeddings."""
        return [
            [float(1.0)] * (ADB_TOKEN_COUNT - 1) + [float(i)] for i in range(len(texts))
        ]

    def embed_query(self, text: str) -> List[float]:
        """Return simple embeddings."""
        return [float(1.0)] * (ADB_TOKEN_COUNT - 1) + [float(0.0)]


def test_analyticdb_mysql() -> None:
    texts = ["foo", "bar", "baz"]
    ids = ["id_foo", "id_bar", "id_baz"]

    # configure settings
    settings = AnalyticDBMySQLSettings()
    settings.host = os.getenv("ADB_HOST", "localhost")
    settings.port = int(os.getenv("ADB_PORT", "3306"))
    settings.user = os.getenv("ADB_USER", "admin")
    settings.password = os.getenv("ADB_PASSWORD", "admin")

    # create AnalyticDB MySQL store
    vectorstore = AnalyticDBMySQL.from_texts(
        texts=texts,
        embedding=FakeEmbeddingsWithADB(),
        config=settings,
        text_ids=ids,
    )

    # similarity_search
    output = vectorstore.similarity_search(query="foo", k=1)
    assert output == [Document(page_content="foo")]

    # max_marginal_relevance_search
    output = vectorstore.max_marginal_relevance_search(query="foo", k=1, fetch_k=2)
    assert output == [Document(page_content="foo")]

    # similarity_search_with_relevance_scores
    result = vectorstore.similarity_search_with_relevance_scores(query="foo", k=1)
    assert result == [(Document(page_content="foo"), 0.0)]

    # delete
    vectorstore.delete(ids=ids)
    output = vectorstore.similarity_search("foo", k=1)
    assert output == []
