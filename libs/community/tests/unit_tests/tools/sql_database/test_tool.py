"""Tests for SQL database tools."""

import pytest
import sqlalchemy as sa
from sqlalchemy import Column, Integer, MetaData, String, Table, insert

from langchain_community.tools.sql_database.tool import (
    InfoSQLDatabaseTool,
    ListSQLDatabaseTool,
    QuerySQLDatabaseTool,
    _extract_table_names,
)
from langchain_community.utilities.sql_database import SQLDatabase

metadata_obj = MetaData()

public_table = Table(
    "public_data",
    metadata_obj,
    Column("id", Integer, primary_key=True),
    Column("name", String(50)),
)

sensitive_table = Table(
    "sensitive_data",
    metadata_obj,
    Column("id", Integer, primary_key=True),
    Column("secret", String(100)),
)


@pytest.fixture
def engine() -> sa.engine.Engine:
    """Create an in-memory SQLite database with test tables."""
    engine = sa.create_engine("sqlite:///:memory:")
    metadata_obj.create_all(engine)

    # Insert test data
    with engine.begin() as conn:
        conn.execute(insert(public_table).values(id=1, name="public_value"))
        conn.execute(insert(sensitive_table).values(id=1, secret="secret_value"))

    return engine


@pytest.fixture
def db_with_ignored_table(engine: sa.engine.Engine) -> SQLDatabase:
    """Create SQLDatabase with sensitive_data in ignore_tables."""
    return SQLDatabase(engine, ignore_tables=["sensitive_data"])


@pytest.fixture
def db_with_include_tables(engine: sa.engine.Engine) -> SQLDatabase:
    """Create SQLDatabase with only public_data in include_tables."""
    return SQLDatabase(engine, include_tables=["public_data"])


@pytest.fixture
def db_unrestricted(engine: sa.engine.Engine) -> SQLDatabase:
    """Create SQLDatabase with no table restrictions."""
    return SQLDatabase(engine)


class TestExtractTableNames:
    """Tests for the table extraction function."""

    def test_simple_select(self) -> None:
        query = "SELECT * FROM users"
        assert _extract_table_names(query) == {"users"}

    def test_select_with_join(self) -> None:
        query = "SELECT * FROM users JOIN orders ON users.id = orders.user_id"
        assert _extract_table_names(query) == {"users", "orders"}

    def test_multiple_joins(self) -> None:
        query = """
            SELECT * FROM users
            LEFT JOIN orders ON users.id = orders.user_id
            INNER JOIN products ON orders.product_id = products.id
        """
        assert _extract_table_names(query) == {"users", "orders", "products"}

    def test_insert_into(self) -> None:
        query = "INSERT INTO users (name) VALUES ('test')"
        assert _extract_table_names(query) == {"users"}

    def test_update(self) -> None:
        query = "UPDATE users SET name = 'test' WHERE id = 1"
        assert _extract_table_names(query) == {"users"}

    def test_delete(self) -> None:
        query = "DELETE FROM users WHERE id = 1"
        assert _extract_table_names(query) == {"users"}

    def test_schema_qualified_table(self) -> None:
        query = "SELECT * FROM public.users"
        assert _extract_table_names(query) == {"users"}

    def test_quoted_table_names(self) -> None:
        query = 'SELECT * FROM "users"'
        assert _extract_table_names(query) == {"users"}

    def test_case_preserving(self) -> None:
        query = "select * from USERS join ORDERS on users.id = orders.user_id"
        assert _extract_table_names(query) == {"USERS", "ORDERS"}

    def test_with_comments(self) -> None:
        query = """
            -- This is a comment
            SELECT * FROM users
            /* multi-line
               comment */
            JOIN orders ON users.id = orders.user_id
        """
        assert _extract_table_names(query) == {"users", "orders"}

    def test_subquery(self) -> None:
        query = """
            SELECT * FROM users
            WHERE id IN (SELECT user_id FROM orders)
        """
        assert _extract_table_names(query) == {"users", "orders"}

    def test_invalid_sql_raises_value_error(self) -> None:
        """Test that invalid SQL raises ValueError."""
        query = "THIS IS NOT VALID SQL AT ALL @@##$$"
        with pytest.raises(ValueError, match="Invalid SQL query"):
            _extract_table_names(query)


class TestQuerySQLDatabaseToolValidation:
    """Tests for QuerySQLDatabaseTool table validation."""

    def test_query_allowed_table_with_ignore_tables(
        self, db_with_ignored_table: SQLDatabase
    ) -> None:
        """Test that querying an allowed table succeeds."""
        tool = QuerySQLDatabaseTool(db=db_with_ignored_table)
        result = tool.run("SELECT * FROM public_data")
        assert "public_value" in str(result)
        assert "Error" not in str(result)

    def test_query_ignored_table_blocked(
        self, db_with_ignored_table: SQLDatabase
    ) -> None:
        """Test that querying an ignored table is blocked."""
        tool = QuerySQLDatabaseTool(db=db_with_ignored_table)
        result = tool.run("SELECT * FROM sensitive_data")
        assert "Error" in str(result)
        assert "not allowed" in str(result)
        assert "sensitive_data" in str(result)

    def test_query_allowed_table_with_include_tables(
        self, db_with_include_tables: SQLDatabase
    ) -> None:
        """Test that querying an included table succeeds."""
        tool = QuerySQLDatabaseTool(db=db_with_include_tables)
        result = tool.run("SELECT * FROM public_data")
        assert "public_value" in str(result)
        assert "Error" not in str(result)

    def test_query_excluded_table_blocked_with_include_tables(
        self, db_with_include_tables: SQLDatabase
    ) -> None:
        """Test that querying a non-included table is blocked."""
        tool = QuerySQLDatabaseTool(db=db_with_include_tables)
        result = tool.run("SELECT * FROM sensitive_data")
        assert "Error" in str(result)
        assert "not allowed" in str(result)

    def test_query_with_join_blocked_if_any_table_restricted(
        self, db_with_ignored_table: SQLDatabase
    ) -> None:
        """Test that a JOIN query is blocked if any table is restricted."""
        tool = QuerySQLDatabaseTool(db=db_with_ignored_table)
        query = (
            "SELECT * FROM public_data "
            "JOIN sensitive_data ON public_data.id = sensitive_data.id"
        )
        result = tool.run(query)
        assert "Error" in str(result)
        assert "not allowed" in str(result)

    def test_unrestricted_db_allows_all_queries(
        self, db_unrestricted: SQLDatabase
    ) -> None:
        """Test that a database with no restrictions allows all queries."""
        tool = QuerySQLDatabaseTool(db=db_unrestricted)
        result = tool.run("SELECT * FROM sensitive_data")
        assert "secret_value" in str(result)
        assert "Error" not in str(result)

    def test_exact_table_name_matching(
        self, db_with_ignored_table: SQLDatabase
    ) -> None:
        """Test that table name matching is exact (case-sensitive)."""
        tool = QuerySQLDatabaseTool(db=db_with_ignored_table)
        # Exact match should be blocked
        result = tool.run("SELECT * FROM sensitive_data")
        assert "Error" in str(result)
        assert "not allowed" in str(result)

    def test_invalid_sql_returns_error(
        self, db_with_ignored_table: SQLDatabase
    ) -> None:
        """Test that invalid SQL returns an error message."""
        tool = QuerySQLDatabaseTool(db=db_with_ignored_table)
        result = tool.run("THIS IS NOT VALID SQL AT ALL @@##$$")
        assert "Error" in str(result)
        assert "Invalid SQL query" in str(result)


class TestListSQLDatabaseTool:
    """Tests for ListSQLDatabaseTool respecting table restrictions."""

    def test_list_tables_excludes_ignored(
        self, db_with_ignored_table: SQLDatabase
    ) -> None:
        """Test that ignored tables are not listed."""
        tool = ListSQLDatabaseTool(db=db_with_ignored_table)
        result = tool.run("")
        assert "public_data" in result
        assert "sensitive_data" not in result

    def test_list_tables_only_includes_specified(
        self, db_with_include_tables: SQLDatabase
    ) -> None:
        """Test that only included tables are listed."""
        tool = ListSQLDatabaseTool(db=db_with_include_tables)
        result = tool.run("")
        assert "public_data" in result
        assert "sensitive_data" not in result


class TestInfoSQLDatabaseTool:
    """Tests for InfoSQLDatabaseTool respecting table restrictions."""

    def test_info_allowed_table(self, db_with_ignored_table: SQLDatabase) -> None:
        """Test getting info for an allowed table."""
        tool = InfoSQLDatabaseTool(db=db_with_ignored_table)
        result = tool.run("public_data")
        assert "public_data" in result
        assert "Error" not in result

    def test_info_ignored_table_blocked(
        self, db_with_ignored_table: SQLDatabase
    ) -> None:
        """Test that getting info for an ignored table returns an error."""
        tool = InfoSQLDatabaseTool(db=db_with_ignored_table)
        result = tool.run("sensitive_data")
        assert "Error" in result
