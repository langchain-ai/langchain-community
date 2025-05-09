from typing import Generator

import pytest
from langchain_tests.integration_tests.vectorstores import VectorStoreIntegrationTests

from langchain_community.vectorstores.surrealdb import SurrealDBStore


class TestSurrealDB(VectorStoreIntegrationTests):
    @property
    def has_async(self) -> bool:
        return False

    @pytest.fixture
    def vectorstore(self) -> Generator[SurrealDBStore, None, None]:  # type: ignore[override]
        try:
            from surrealdb import Surreal
        except ImportError as e:
            raise ImportError(
                """Cannot import from surrealdb.
                please install with `pip install surrealdb`."""
            ) from e

        conn = Surreal("ws://localhost:8000/rpc")
        conn.signin({"username": "root", "password": "root"})
        conn.use("langchain", "test")
        store = SurrealDBStore(self.get_embeddings(), conn)
        store.delete()
        try:
            yield store
        finally:
            store.delete()


# FIXME: async test throws "got Future <Future pending> attached to a different
#        loop" error
# class TestSurrealDBAsync(VectorStoreIntegrationTests):
#     @property
#     def has_sync(self) -> bool:
#         return False
#
#     @pytest.fixture
#     def vectorstore(self) -> Generator[SurrealDBStore, None, None]:
#         try:
#             from surrealdb import AsyncSurreal
#         except ImportError as e:
#             raise ImportError(
#                 """Cannot import from surrealdb.
#                 please install with `pip install surrealdb`."""
#             ) from e
#
#         async def _connect() -> AsyncSurreal:
#             conn = AsyncSurreal("ws://localhost:8000/rpc")
#             await conn.signin({"username": "root", "password": "root"})
#             await conn.use("langchain", "test")
#             return conn
#
#         async_conn = asyncio.run(_connect())
#         store = SurrealDBStore(
#           self.get_embeddings(), None, async_connection=async_conn
#         )
#         asyncio.run(store.adelete())
#         try:
#             yield store
#         finally:
#             asyncio.run(store.adelete())
