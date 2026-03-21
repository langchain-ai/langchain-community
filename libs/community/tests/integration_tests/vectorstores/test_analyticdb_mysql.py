"""Test AlibabaCloud AnalyticDB MySQL functionality.
1. create a AlibabaCloud AnalyticDB MySQL instance (https://adb.console.aliyun.com/).
2. connect to instance by mysql client, running:
    'create database vectorstore;'
3. shell running:
    export ADB_HOST=<...>
    export ADB_PORT=<...>
    export ADB_USER=<...>
    export ADB_PASSWORD=<...>
    export DASHSCOPE_API_KEY=<...>
"""

import os

from langchain_core.documents import Document
from libs.community.langchain_community.embeddings.dashscope import DashScopeEmbeddings

from langchain_community.vectorstores.analyticdb_mysql import (
    AnalyticDBMySQL,
    AnalyticDBMySQLSettings,
)


def test_analyticdb_mysql() -> None:
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=".env", override=False)

    texts = ["foo", "bar", "baz"]
    ids = ["id_foo", "id_bar", "id_baz"]
    metas = [{"name": "foo_n"}, {"name": "bar_n"}, {"name": "baz_n"}]

    # configure settings
    settings = AnalyticDBMySQLSettings()
    settings.host = os.getenv("ADB_HOST", "localhost")
    settings.port = int(os.getenv("ADB_PORT", "3306"))
    settings.user = os.getenv("ADB_USER", "admin")
    settings.password = os.getenv("ADB_PASSWORD", "admin")

    # create AnalyticDB MySQL store
    vectorstore = AnalyticDBMySQL.from_texts(
        texts=texts,
        text_ids=ids,
        metadatas=metas,
        embedding=DashScopeEmbeddings(
            model="text-embedding-v4",
            dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
        ),
        config=settings,
    )

    # similarity_search
    output = vectorstore.similarity_search(query="foo", k=1)
    assert output == [Document(page_content="foo", metadata={"name": "foo_n"})]

    output = vectorstore.similarity_search(query="bar", k=1, filter={"name": "bar_n"})
    assert output == [Document(page_content="bar", metadata={"name": "bar_n"})]

    # max_marginal_relevance_search
    output = vectorstore.max_marginal_relevance_search(query="foo", k=1, fetch_k=2)
    assert output == [Document(page_content="foo", metadata={"name": "foo_n"})]

    output = vectorstore.max_marginal_relevance_search(
        query="bar", k=1, fetch_k=2, filter={"name": {"eq": "bar_n"}}
    )
    assert output == [Document(page_content="bar", metadata={"name": "bar_n"})]

    # similarity_search_with_relevance_scores
    result = vectorstore.similarity_search_with_relevance_scores(query="foo", k=1)
    assert result == [(Document(page_content="foo", metadata={"name": "foo_n"}), 0.0)]

    result = vectorstore.similarity_search_with_relevance_scores(
        query="bar", k=1, filter={"name": {"eq": "bar_n"}}
    )
    assert result == [(Document(page_content="bar", metadata={"name": "bar_n"}), 0.0)]

    # get_by_ids
    output = vectorstore.get_by_ids(ids=["id_baz"])
    assert output == [Document(page_content="baz", metadata={"name": "baz_n"})]

    # delete
    vectorstore.delete(ids=ids)
    output = vectorstore.similarity_search("foo", k=1)
    assert output == []

    # truncate
    vectorstore.delete()
