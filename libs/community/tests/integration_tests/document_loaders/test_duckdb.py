import unittest

from langchain_community.document_loaders.duckdb_loader import DuckDBLoader

try:
    import duckdb  # noqa: F401

    duckdb_installed = True
except ImportError:
    duckdb_installed = False


@unittest.skipIf(not duckdb_installed, "duckdb not installed")
def test_duckdb_loader_no_options() -> None:
    """Test DuckDB loader."""

    loader = DuckDBLoader("SELECT 1 AS a, 2 AS b")
    docs = loader.load()

    assert len(docs) == 1
    assert docs[0].page_content == "a: 1\nb: 2"
    assert docs[0].metadata == {}


@unittest.skipIf(not duckdb_installed, "duckdb not installed")
def test_duckdb_loader_page_content_columns() -> None:
    """Test DuckDB loader."""

    loader = DuckDBLoader(
        "SELECT 1 AS a, 2 AS b UNION SELECT 3 AS a, 4 AS b",
        page_content_columns=["a"],
    )
    docs = loader.load()

    assert len(docs) == 2
    assert docs[0].page_content == "a: 1"
    assert docs[0].metadata == {}

    assert docs[1].page_content == "a: 3"
    assert docs[1].metadata == {}


@unittest.skipIf(not duckdb_installed, "duckdb not installed")
def test_duckdb_loader_metadata_columns() -> None:
    """Test DuckDB loader."""

    loader = DuckDBLoader(
        "SELECT 1 AS a, 2 AS b",
        page_content_columns=["a"],
        metadata_columns=["b"],
    )
    docs = loader.load()

    assert len(docs) == 1
    assert docs[0].page_content == "a: 1"
    assert docs[0].metadata == {"b": 2}


@unittest.skipIf(not duckdb_installed, "duckdb not installed")
def test_duckdb_loader_user_agent() -> None:
    """Test that DuckDBLoader injects langchain user agent."""
    loader = DuckDBLoader("SELECT current_setting('custom_user_agent') as ua")
    docs = loader.load()

    assert len(docs) == 1
    assert "langchain" in docs[0].page_content


@unittest.skipIf(not duckdb_installed, "duckdb not installed")
def test_duckdb_loader_user_agent_preserves_user_config() -> None:
    """Test that DuckDBLoader preserves user-provided config and prepends langchain."""
    loader = DuckDBLoader(
        "SELECT current_setting('custom_user_agent') as ua",
        config={"custom_user_agent": "my-app"},
    )
    docs = loader.load()

    assert len(docs) == 1
    # langchain should come first, followed by user's value
    assert "langchain my-app" in docs[0].page_content
