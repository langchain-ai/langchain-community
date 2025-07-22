import json
from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_community.vectorstores.clickhouse import Clickhouse


class DummyClient:
    def query(self, sql: str):
        # Simulate query result with document, metadata, and distance
        class DummyResult:
            def named_results(self):
                return [
                    {
                        "doc": "Sample document 1",
                        "metadata": {"source": "file1", "id": 101},
                        "distance": 0.12,
                    },
                    {
                        "doc": "Sample document 2",
                        "metadata": {"source": "file2", "id": 202},
                        "distance": 0.34,
                    },
                ]
        return DummyResult()


class DummyClickhouse(Clickhouse):
    def __init__(self):
        self.client = DummyClient()
        self.config = type("Config", (), {
            "column_map": {
                "document": "doc",
                "metadata": "metadata"
            }
        })()

    def _build_query_sql(self, embedding: List[float], k: int, where_str: str = None) -> str:
        return "SELECT * FROM dummy_table"


def test_similarity_search_by_vector_returns_distance():
    ch = DummyClickhouse()
    results = ch.similarity_search_by_vector([0.1, 0.2, 0.3], k=2)

    assert isinstance(results, list)
    assert len(results) == 2
    assert all(isinstance(doc, Document) for doc in results)
    assert results[0].metadata["source"] == "file1"
    assert "distance" in results[0].metadata
    assert isinstance(results[0].metadata["distance"], float)
    print("✅ Test passed: Metadata and distance are returned correctly.")


if __name__ == "__main__":
    test_similarity_search_by_vector_returns_distance()
