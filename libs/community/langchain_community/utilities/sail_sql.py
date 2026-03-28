"""Utility for interacting with Sail, a drop-in replacement for Spark SQL."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional

from langchain_community.utilities.spark_sql import SparkSQL

if TYPE_CHECKING:
    from pyspark.sql import SparkSession

_DEFAULT_SAIL_URI = "sc://localhost:50051"


class SailSQL(SparkSQL):
    """SailSQL is a utility class for interacting with Sail.

    Sail is a drop-in replacement for Apache Spark that uses the Spark Connect
    protocol. This class extends SparkSQL and defaults to connecting to a Sail
    server rather than a local Spark session.

    See https://docs.lakesail.com/ for more information.
    """

    def __init__(
        self,
        spark_session: Optional[SparkSession] = None,
        sail_uri: str = _DEFAULT_SAIL_URI,
        catalog: Optional[str] = None,
        schema: Optional[str] = None,
        ignore_tables: Optional[List[str]] = None,
        include_tables: Optional[List[str]] = None,
        sample_rows_in_table_info: int = 3,
    ):
        """Initialize a SailSQL object.

        Args:
            spark_session: A SparkSession object. If not provided, a remote
                session will be created using ``sail_uri``.
            sail_uri: The URI of the Sail server. Defaults to
                ``"sc://localhost:50051"``.
            catalog: The catalog to use.
                If not provided, the default catalog will be used.
            schema: The schema to use.
                If not provided, the default schema will be used.
            ignore_tables: A list of tables to ignore.
                If not provided, all tables will be used.
            include_tables: A list of tables to include.
                If not provided, all tables will be used.
            sample_rows_in_table_info: The number of rows to include in the
                table info. Defaults to 3.
        """
        if spark_session is None:
            spark_session = self._create_sail_session(sail_uri)

        super().__init__(
            spark_session=spark_session,
            catalog=catalog,
            schema=schema,
            ignore_tables=ignore_tables,
            include_tables=include_tables,
            sample_rows_in_table_info=sample_rows_in_table_info,
        )

    @staticmethod
    def _create_sail_session(uri: str) -> SparkSession:
        """Create a remote SparkSession connected to a Sail server.

        Args:
            uri: The URI of the Sail server.

        Returns:
            A SparkSession connected to the Sail server.
        """
        try:
            from pyspark.sql import SparkSession
        except ImportError:
            msg = (
                "pyspark is not installed. "
                "Please install it with `pip install pyspark pysail`"
            )
            raise ImportError(msg)

        return SparkSession.builder.remote(uri).getOrCreate()

    @classmethod
    def from_uri(
        cls,
        database_uri: str = _DEFAULT_SAIL_URI,
        engine_args: Optional[dict] = None,
        **kwargs: Any,
    ) -> SailSQL:
        """Create a SailSQL instance from a Sail server URI.

        For example: ``SailSQL.from_uri("sc://localhost:50051")``

        Args:
            database_uri: The URI of the Sail server.
            engine_args: Not used. Kept for API compatibility with SparkSQL.
            kwargs: Additional keyword arguments passed to the constructor.

        Returns:
            A SailSQL instance connected to the specified Sail server.
        """
        session = cls._create_sail_session(database_uri)
        return cls(spark_session=session, **kwargs)

    def _get_create_table_stmt(self, table: str) -> str:
        """Get table schema using DESCRIBE TABLE.

        Sail does not support ``SHOW CREATE TABLE``, so this override uses
        ``DESCRIBE TABLE`` and reconstructs a CREATE TABLE statement from the
        column metadata.
        """
        rows = self._spark.sql(f"DESCRIBE TABLE {table}").collect()
        columns = ", ".join(f"{row.col_name} {row.data_type}" for row in rows)
        return f"CREATE TABLE {table} ({columns});"
