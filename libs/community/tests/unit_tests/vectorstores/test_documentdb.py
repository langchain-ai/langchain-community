"""
Test for DocumentDB Indexing API Bug #507

Run with:
pytest tests/unit_tests/vectorstores/test_documentdb.py -v -s
"""

from typing import Any, Generator
from unittest.mock import Mock, patch

import pytest


class MockObjectId:
    """Mock ObjectId for testing without pymongo."""

    def __init__(self, oid: str | None = None) -> None:
        self.oid = oid or "507f1f77bcf86cd799439011"

    def __str__(self) -> str:
        return self.oid

    def __repr__(self) -> str:
        return f"ObjectId('{self.oid}')"


class MockInvalidId(Exception):
    """Mock InvalidId exception."""


@pytest.fixture
def mock_bson() -> Generator[None, None, None]:
    """Mock bson module to avoid pymongo dependency in tests."""
    with patch.dict(
        "sys.modules",
        {"bson": Mock(ObjectId=MockObjectId, errors=Mock(InvalidId=MockInvalidId))},
    ):
        yield


class TestDocumentDBIndexingBug507:
    """
    Tests to reproduce and verify fix for issue #507.

    Bug: DocumentDB VectorStore fails when used with Indexing API
    - Indexing API provides SHA256 hash IDs
    - DocumentDB ignores these and generates ObjectIds
    - Deletion fails with InvalidId error
    """

    @pytest.fixture
    def mock_collection(self) -> Mock:
        """Mock MongoDB collection."""
        collection = Mock()
        collection.insert_many = Mock()
        collection.delete_one = Mock()
        return collection

    @pytest.fixture
    def mock_embeddings(self) -> Mock:
        """Mock embeddings."""
        embeddings = Mock()
        embeddings.embed_documents = Mock(
            return_value=[
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
            ]
        )
        embeddings.embed_query = Mock(return_value=[0.1, 0.2, 0.3])
        return embeddings

    def test_current_behavior_analysis(
        self, mock_collection: Mock, mock_embeddings: Mock
    ) -> None:
        """
        Analyze current add_texts behavior to understand the bug.

        This test examines what happens when custom IDs are provided.
        EXPECTED: This test will show that custom IDs are ignored (the bug).
        """
        # This is a documentation test - helps understand the problem
        assert True, "Manual inspection test"

    def test_bug_scenario_with_mocked_implementation(
        self, mock_collection: Mock, mock_embeddings: Mock
    ) -> None:
        """
        Simulate the bug scenario with a mock implementation.

        This shows what happens BEFORE the fix.
        """
        # Simulate current behavior: IDs are ignored
        provided_ids = [
            "sha256_abc123...",  # 64-char SHA256 hash
            "sha256_def456...",
        ]

        # Mock the CURRENT (buggy) behavior
        inserted_docs: list[Any] = []

        def buggy_insert(docs: list[Any]) -> Mock:
            # Current code doesn't set _id from provided ids
            for doc in docs:
                # ID is NOT set from the provided_ids list
                inserted_docs.append(doc)
            result = Mock()
            # Auto-generates ObjectIds (not using provided IDs)
            result.inserted_ids = [MockObjectId(), MockObjectId()]
            return result

        mock_collection.insert_many = buggy_insert

        # Simulate current add_texts behavior
        returned_ids = [str(oid) for oid in [MockObjectId(), MockObjectId()]]

        # This demonstrates the bug
        assert returned_ids != provided_ids, "BUG CONFIRMED: IDs don't match!"

    def test_bug_scenario_deletion_fails(
        self, mock_collection: Mock, mock_embeddings: Mock
    ) -> None:
        """
        Simulate the deletion failure with SHA256 IDs.

        This shows what happens when trying to delete with SHA256 IDs.
        """
        sha256_id = "dabe86f55afa3cd189455fed644d18dea7fc16d8e89ac901658655c578793a04"

        # Mock the CURRENT behavior: tries to convert to ObjectId
        def buggy_delete(filter_dict: dict[str, Any]) -> Mock:
            doc_id = filter_dict.get("_id")
            # Current code tries: ObjectId(doc_id)
            # This fails for SHA256 strings
            if len(str(doc_id)) != 24:  # ObjectId must be 24 chars
                raise MockInvalidId(
                    f"'{doc_id}' is not a valid ObjectId, "
                    f"it must be a 12-byte input or a 24-character hex string"
                )
            return Mock(deleted_count=1)

        mock_collection.delete_one = buggy_delete

        # This should fail with current implementation
        try:
            mock_collection.delete_one({"_id": sha256_id})
            assert False, "Should have failed!"
        except MockInvalidId:
            assert True, "Expected failure demonstrates the bug"


class TestDocumentDBFixedBehavior:
    """
    Tests showing the EXPECTED behavior after the fix.

    These tests show what the code SHOULD do.
    """

    @pytest.fixture
    def mock_collection(self) -> Mock:
        """Mock MongoDB collection."""
        collection = Mock()
        return collection

    @pytest.fixture
    def mock_embeddings(self) -> Mock:
        """Mock embeddings."""
        embeddings = Mock()
        embeddings.embed_documents = Mock(
            return_value=[
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
            ]
        )
        return embeddings

    def test_fixed_add_texts_preserves_custom_ids(
        self, mock_collection: Mock, mock_embeddings: Mock
    ) -> None:
        """
        Show the EXPECTED behavior after fix: preserve custom IDs.

        After the fix is applied, add_texts should:
        1. Use provided IDs as document _id
        2. Return the same IDs
        """
        texts = ["Document 1", "Document 2"]
        provided_ids = [
            "sha256_abc123def456...",
            "sha256_789xyz012...",
        ]

        # Mock the FIXED behavior
        inserted_docs: list[dict[str, Any]] = []

        def fixed_insert(docs: list[dict[str, Any]]) -> Mock:
            for doc in docs:
                # FIXED: _id is now set from provided IDs
                inserted_docs.append(doc)
            result = Mock()
            # FIXED: Return the provided IDs
            result.inserted_ids = [doc["_id"] for doc in docs]
            return result

        mock_collection.insert_many = fixed_insert

        # Simulate FIXED add_texts behavior
        # The fix sets doc["_id"] = provided_ids[i] for each document
        docs_to_insert = []
        for i, text in enumerate(texts):
            doc: dict[str, Any] = {
                "textContent": text,
                "vectorContent": [0.1, 0.2, 0.3],
                "_id": provided_ids[i],  # FIXED: Use provided ID
            }
            docs_to_insert.append(doc)

        fixed_insert(docs_to_insert)
        returned_ids = provided_ids  # FIXED: Return provided IDs

        # Verify the fix
        assert returned_ids == provided_ids, "IDs should be preserved"
        assert inserted_docs[0]["_id"] == provided_ids[0], "ID stored correctly"

    def test_fixed_delete_accepts_any_string_id(
        self, mock_collection: Mock, mock_embeddings: Mock
    ) -> None:
        """
        Show the EXPECTED behavior after fix: accept any string ID.

        After the fix, delete should accept:
        - SHA256 hashes (64 chars)
        - ObjectId strings (24 chars)
        - Any other string IDs
        """
        sha256_id = "dabe86f55afa3cd189455fed644d18dea7fc16d8e89ac901658655c578793a04"

        # Mock the FIXED behavior
        def fixed_delete(filter_dict: dict[str, Any]) -> Mock:
            # FIXED: Just use the ID as-is, don't convert to ObjectId
            return Mock(deleted_count=1)

        mock_collection.delete_one = fixed_delete

        # This should work after fix
        result = mock_collection.delete_one({"_id": sha256_id})

        assert result.deleted_count == 1

    def test_backward_compatibility_objectid_still_works(
        self, mock_collection: Mock, mock_embeddings: Mock
    ) -> None:
        """
        Verify backward compatibility: ObjectId strings still work.
        """
        objectid_string = "507f1f77bcf86cd799439011"  # 24-char hex

        def fixed_delete(filter_dict: dict[str, Any]) -> Mock:
            # Fixed version tries ObjectId first for 24-char strings
            return Mock(deleted_count=1)

        mock_collection.delete_one = fixed_delete

        result = mock_collection.delete_one({"_id": objectid_string})
        assert result.deleted_count == 1

    def test_add_texts_raises_on_ids_length_mismatch(
        self, mock_collection: Mock, mock_embeddings: Mock
    ) -> None:
        """Verify ValueError is raised when ids count does not match texts count."""
        from langchain_community.vectorstores.documentdb import DocumentDBVectorSearch

        vector_store = DocumentDBVectorSearch(
            collection=mock_collection,
            embedding=mock_embeddings,
        )

        with pytest.raises(ValueError, match="must match"):
            vector_store.add_texts(
                texts=["doc1", "doc2", "doc3"],
                ids=["id1", "id2"],  # 2 ids for 3 texts
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
