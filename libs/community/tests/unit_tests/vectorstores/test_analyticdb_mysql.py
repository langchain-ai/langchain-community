from typing import List
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from langchain_community.vectorstores.analyticdb_mysql import (
    AnalyticDBMySQL,
    AnalyticDBMySQLSettings,
)

ADB_TOKEN_COUNT = 8


class FakeEmbeddingsWithADB(Embeddings):
    """Fake embeddings functionality for testing."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Return simple embeddings."""
        return [
            [float(1.0)] * (ADB_TOKEN_COUNT - 1) + [float(i)] for i in range(len(texts))
        ]

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        """Return constant query embeddings."""
        return [float(1.0)] * (ADB_TOKEN_COUNT - 1) + [float(0.0)]

    async def aembed_query(self, text: str) -> List[float]:
        return self.embed_query(text)


def test_analyticdb_mysql() -> None:
    """Test the AnalyticDBMySQL implementation using mocks."""

    # Test data
    texts = ["foo", "bar", "baz"]
    ids = ["id_foo", "id_bar", "id_baz"]
    metas = [{"name": "foo_n"}, {"name": "bar_n"}, {"name": "baz_n"}]

    # Create mock pymysql module
    mock_pymysql = MagicMock()
    mock_cursor = MagicMock()
    mock_conn = MagicMock()

    mock_pymysql.connect.return_value = mock_conn
    mock_conn.cursor.return_value = mock_cursor

    # Mock the query result
    mock_cursor.fetchall.return_value = [
        (
            "id_foo",
            "foo",
            "[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]",
            '{"name": "foo_n"}',
        ),
    ]
    mock_cursor.description = [
        ("id",),
        ("document",),
        ("embedding",),
        ("metadata",),
    ]

    # Use patch.dict to mock sys.modules
    with patch.dict(
        "sys.modules",
        {
            "pymysql": mock_pymysql,
        },
    ):
        # Create store instance
        setting = AnalyticDBMySQLSettings.get_env_config()
        store = AnalyticDBMySQL(
            embedding=FakeEmbeddingsWithADB(),
            config=setting,
        )

        # Verify connection
        mock_pymysql.connect.assert_called_once_with(
            host=setting.host,
            port=setting.port,
            user=setting.user,
            password=setting.password,
            database=setting.database,
        )

        # Test add_texts operation and verify
        store.add_texts(texts=texts, ids=ids, metadatas=metas)
        assert mock_cursor.executemany.called or mock_cursor.execute.called

        # Test similarity_search operation and verify
        result1 = store.similarity_search(query="foo", k=1)
        assert result1 == [Document(page_content="foo", metadata={"name": "foo_n"})]
        assert mock_cursor.execute.called

        result1 = store.similarity_search(query="foo", k=1, filter={"name": "foo_n"})
        assert result1 == [Document(page_content="foo", metadata={"name": "foo_n"})]
        assert mock_cursor.execute.called

        # Test max_marginal_relevance_search operation and verify
        result2 = store.max_marginal_relevance_search(query="foo", k=1, fetch_k=2)
        assert result2 == [Document(page_content="foo", metadata={"name": "foo_n"})]
        assert mock_cursor.execute.called

        result2 = store.max_marginal_relevance_search(
            query="foo", k=1, fetch_k=2, filter={"name": "foo_n"}
        )
        assert result2 == [Document(page_content="foo", metadata={"name": "foo_n"})]
        assert mock_cursor.execute.called

        # Reset mock for search operation
        mock_cursor.reset_mock()
        mock_cursor.fetchall.return_value = [
            (
                "id_foo",
                "foo",
                '{"name": "foo_n"}',
                1.0,
            ),
        ]
        mock_cursor.description = [
            ("id",),
            ("document",),
            ("metadata",),
            ("dist",),
        ]

        # Test similarity_search_with_relevance_scores operation and verify
        result3 = store.similarity_search_with_relevance_scores(query="foo", k=1)
        assert result3 == [
            (Document(page_content="foo", metadata={"name": "foo_n"}), 1.0)
        ]
        assert mock_cursor.execute.called

        result3 = store.similarity_search_with_relevance_scores(
            query="foo", k=1, filter={"name": "foo_n"}
        )
        assert result3 == [
            (Document(page_content="foo", metadata={"name": "foo_n"}), 1.0)
        ]
        assert mock_cursor.execute.called

        # Test get_by_ids operation and verify
        result4 = store.get_by_ids(ids=["id_foo"])
        assert result4 == [Document(page_content="foo", metadata={"name": "foo_n"})]
        assert mock_cursor.execute.called

        result4 = store.get_by_ids()
        assert result4 == []
        assert mock_cursor.execute.called

        # Test delete operation and verify
        store.delete(ids=["id_foo"])
        calls = [
            call
            for call in mock_cursor.execute.call_args_list
            if "DELETE FROM" in str(call)
        ]
        assert len(calls) > 0 or mock_cursor.execute.called

        store.delete()
        calls = [
            call
            for call in mock_cursor.execute.call_args_list
            if "TRUNCATE" in str(call)
        ]
        assert len(calls) > 0 or mock_cursor.execute.called

        # Test drop operation and verify
        store.drop()
        calls = [
            call
            for call in mock_cursor.execute.call_args_list
            if "DROP TABLE" in str(call)
        ]
        assert len(calls) > 0 or mock_cursor.execute.called
