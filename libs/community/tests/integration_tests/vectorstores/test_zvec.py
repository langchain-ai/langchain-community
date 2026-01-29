from time import sleep

from langchain_core.documents import Document

from langchain_community.vectorstores import Zvec
from tests.integration_tests.vectorstores.fake_embeddings import FakeEmbeddings

texts = ["foo", "bar", "baz"]
ids = ["1", "2", "3"]


def test_zvec_from_texts() -> None:
    zvec = Zvec.from_texts(
        texts=texts,
        embedding=FakeEmbeddings(),
        ids=ids,
    )

    # the vector insert operation is async by design, we wait here a bit for the
    # insertion to complete.
    sleep(0.5)
    output = zvec.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]


def test_zvec_with_text_with_metadatas() -> None:
    metadatas = [{"meta": i} for i in range(len(texts))]
    zvec = Zvec.from_texts(
        texts=texts,
        embedding=FakeEmbeddings(),
        metadatas=metadatas,
        ids=ids,
    )

    # the vector insert operation is async by design, we wait here a bit for the
    # insertion to complete.
    sleep(0.5)
    output = zvec.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo", metadata={"meta": 0})]


def test_zvec_search_with_filter() -> None:
    metadatas = [{"meta": i} for i in range(len(texts))]
    zvec = Zvec.from_texts(
        texts=texts,
        embedding=FakeEmbeddings(),
        metadatas=metadatas,
        ids=ids,
    )

    # the vector insert operation is async by design, we wait here a bit for the
    # insertion to complete.
    sleep(0.5)
    output = zvec.similarity_search("foo", filter="meta=2")
    assert output == [Document(page_content="baz", metadata={"meta": 2})]


def test_zvec_search_with_scores() -> None:
    zvec = Zvec.from_texts(
        texts=texts,
        embedding=FakeEmbeddings(),
        ids=ids,
    )

    # the vector insert operation is async by design, we wait here a bit for the
    # insertion to complete.
    sleep(0.5)
    output = zvec.similarity_search_with_relevance_scores("foo")
    docs, scores = zip(*output)

    assert scores[0] < scores[1] < scores[2]
    assert list(docs) == [
        Document(page_content="foo"),
        Document(page_content="bar"),
        Document(page_content="baz"),
    ]


def test_zvec_delete_by_id() -> None:
    zvec = Zvec.from_texts(
        texts=texts,
        embedding=FakeEmbeddings(),
        ids=ids,
    )
    sleep(0.5)
    zvec.delete(ids=["1"])
    assert zvec.similarity_search("foo", k=1) == [Document(page_content="bar")]
