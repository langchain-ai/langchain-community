from langchain_community.utilities.sql_database import SQLDatabase


def test_sql_database_run_content_and_artifact():
    db = SQLDatabase.from_uri("sqlite+pysqlite:///:memory:")
    db.run("CREATE TABLE test (id INTEGER, name TEXT);")
    db.run("INSERT INTO test (id, name) VALUES (1, 'foo');")
    db.run("INSERT INTO test (id, name) VALUES (2, 'bar');")

    # Test content_and_artifact format
    string_result, artifact = db.run(
        "SELECT * FROM test", response_format="content_and_artifact"
    )
    assert "foo" in string_result and "bar" in string_result
    assert (1, "foo") in artifact and (2, "bar") in artifact

    # Test content_and_artifact format with empty result set
    empty_string, empty_artifact = db.run(
        "SELECT * FROM test WHERE id = 999", response_format="content_and_artifact"
    )
    assert empty_string == ""
    assert empty_artifact == []

    # Test default format (string only)
    only_string = db.run("SELECT * FROM test")
    assert isinstance(only_string, str)
