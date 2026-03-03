"""Unit tests for SQLiteVec metadata filtering logic.

These tests exercise ``SQLiteVec._build_metadata_filter`` which converts a
metadata filter dict into a parameterized SQL WHERE clause.  No sqlite-vec
dependency is required.
"""

import pytest

from langchain_community.vectorstores.sqlitevec import SQLiteVec


class TestBuildMetadataFilter:
    """Tests for SQLiteVec._build_metadata_filter."""

    def test_simple_equality(self) -> None:
        clause, params = SQLiteVec._build_metadata_filter({"color": "red"})
        assert clause == "json_extract(e.metadata, '$.color') = ?"
        assert params == ["red"]

    def test_integer_equality(self) -> None:
        clause, params = SQLiteVec._build_metadata_filter({"page": 3})
        assert clause == "json_extract(e.metadata, '$.page') = ?"
        assert params == [3]

    def test_eq_operator(self) -> None:
        clause, params = SQLiteVec._build_metadata_filter({"status": {"$eq": "active"}})
        assert clause == "json_extract(e.metadata, '$.status') = ?"
        assert params == ["active"]

    def test_ne_operator(self) -> None:
        clause, params = SQLiteVec._build_metadata_filter(
            {"status": {"$ne": "deleted"}}
        )
        assert clause == "json_extract(e.metadata, '$.status') != ?"
        assert params == ["deleted"]

    def test_gt_operator(self) -> None:
        clause, params = SQLiteVec._build_metadata_filter({"score": {"$gt": 5}})
        assert clause == "json_extract(e.metadata, '$.score') > ?"
        assert params == [5]

    def test_gte_operator(self) -> None:
        clause, params = SQLiteVec._build_metadata_filter({"score": {"$gte": 5}})
        assert clause == "json_extract(e.metadata, '$.score') >= ?"
        assert params == [5]

    def test_lt_operator(self) -> None:
        clause, params = SQLiteVec._build_metadata_filter({"score": {"$lt": 10}})
        assert clause == "json_extract(e.metadata, '$.score') < ?"
        assert params == [10]

    def test_lte_operator(self) -> None:
        clause, params = SQLiteVec._build_metadata_filter({"score": {"$lte": 10}})
        assert clause == "json_extract(e.metadata, '$.score') <= ?"
        assert params == [10]

    def test_in_operator(self) -> None:
        clause, params = SQLiteVec._build_metadata_filter(
            {"category": {"$in": ["a", "b", "c"]}}
        )
        assert clause == "json_extract(e.metadata, '$.category') IN (?, ?, ?)"
        assert params == ["a", "b", "c"]

    def test_nin_operator(self) -> None:
        clause, params = SQLiteVec._build_metadata_filter(
            {"category": {"$nin": ["x", "y"]}}
        )
        assert clause == "json_extract(e.metadata, '$.category') NOT IN (?, ?)"
        assert params == ["x", "y"]

    def test_multiple_keys_combined_with_and(self) -> None:
        clause, params = SQLiteVec._build_metadata_filter({"color": "red", "page": 3})
        assert "AND" in clause
        assert "json_extract(e.metadata, '$.color') = ?" in clause
        assert "json_extract(e.metadata, '$.page') = ?" in clause
        assert params == ["red", 3]

    def test_mixed_equality_and_operator(self) -> None:
        clause, params = SQLiteVec._build_metadata_filter(
            {"color": "red", "score": {"$gt": 5}}
        )
        assert "json_extract(e.metadata, '$.color') = ?" in clause
        assert "json_extract(e.metadata, '$.score') > ?" in clause
        assert params == ["red", 5]

    def test_dotted_key(self) -> None:
        clause, params = SQLiteVec._build_metadata_filter({"a.b": "val"})
        assert clause == "json_extract(e.metadata, '$.a.b') = ?"
        assert params == ["val"]

    def test_underscore_key(self) -> None:
        clause, params = SQLiteVec._build_metadata_filter({"my_key": "val"})
        assert clause == "json_extract(e.metadata, '$.my_key') = ?"
        assert params == ["val"]

    def test_invalid_key_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid metadata filter key"):
            SQLiteVec._build_metadata_filter({"bad key!": "value"})

    def test_invalid_key_with_quotes_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid metadata filter key"):
            SQLiteVec._build_metadata_filter({"key'--": "value"})

    def test_unsupported_operator_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported filter operator"):
            SQLiteVec._build_metadata_filter({"x": {"$regex": ".*"}})

    def test_in_operator_with_non_list_raises(self) -> None:
        with pytest.raises(ValueError, match="\\$in operator requires a list"):
            SQLiteVec._build_metadata_filter({"x": {"$in": "not_a_list"}})

    def test_nin_operator_with_non_list_raises(self) -> None:
        with pytest.raises(ValueError, match="\\$nin operator requires a list"):
            SQLiteVec._build_metadata_filter({"x": {"$nin": 42}})

    def test_empty_filter(self) -> None:
        clause, params = SQLiteVec._build_metadata_filter({})
        assert clause == ""
        assert params == []
