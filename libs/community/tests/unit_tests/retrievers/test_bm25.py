import pytest
from langchain_core.documents import Document

from langchain_community.retrievers.bm25 import BM25Retriever


@pytest.mark.requires("bm25s")
def test_from_texts() -> None:
    input_texts = ["I have a pen.", "Do you have a pen?", "I have a bag.", "I like LangChain."]

    bm25_retriever = BM25Retriever.from_texts(texts=input_texts)
    assert len(bm25_retriever.docs) == 4

    results = bm25_retriever.invoke("pen")
    assert len(results) == len(input_texts)
    assert [d.page_content for d in results[:2]] == [
        "I have a pen.",
        "Do you have a pen?",
    ]


@pytest.mark.requires("bm25s")
def test_from_texts_with_bm25_params() -> None:
    input_texts = ["I have a pen.", "Do you have a pen?", "I have a bag.", "I like LangChain."]
    bm25_retriever = BM25Retriever.from_texts(
        texts=input_texts, bm25_params={"k1": 2.0, "b": 0.5, "delta": 0.0, "method": "lucene", "idf_method": "atire"},
    )

    assert bm25_retriever.vectorizer.k1 == 2.0
    assert bm25_retriever.vectorizer.b == 0.5
    assert bm25_retriever.vectorizer.delta == 0.0
    assert bm25_retriever.vectorizer.method == "lucene"
    assert bm25_retriever.vectorizer.idf_method == "atire"


@pytest.mark.requires("bm25s")
def test_from_documents() -> None:
    input_docs = [
        Document(page_content="I have a pen."),
        Document(page_content="Do you have a pen?"),
        Document(page_content="I have a bag."),
        Document(page_content="I like LangChain."),
    ]
    bm25_retriever = BM25Retriever.from_documents(documents=input_docs)
    assert len(bm25_retriever.docs) == 4

    results = bm25_retriever.invoke("bag")
    assert results[0].page_content == "I have a bag."


@pytest.mark.requires("bm25s")
def test_repr() -> None:
    input_docs = [
        Document(page_content="I have a pen."),
        Document(page_content="Do you have a pen?"),
        Document(page_content="I have a bag."),
        Document(page_content="I like LangChain."),
    ]
    bm25_retriever = BM25Retriever.from_documents(documents=input_docs)
    assert "I have a pen" not in repr(bm25_retriever)


@pytest.mark.requires("bm25s")
def test_doc_id() -> None:
    docs_with_ids = [
        Document(page_content="I have a pen.", id="1"),
        Document(page_content="Do you have a pen?", id="2"),
        Document(page_content="I have a bag.", id="3"),
        Document(page_content="I like LangChain.", id="4"),
    ]
    docs_without_ids = [
        Document(page_content="I have a pen."),
        Document(page_content="Do you have a pen?"),
        Document(page_content="I have a bag."),
        Document(page_content="I like LangChain."),
    ]
    docs_with_some_ids = [
        Document(page_content="I have a pen.", id="1"),
        Document(page_content="Do you have a pen?"),
        Document(page_content="I have a bag.", id="3"),
        Document(page_content="I like LangChain."),
    ]
    bm25_retriever_with_ids = BM25Retriever.from_documents(documents=docs_with_ids)
    bm25_retriever_without_ids = BM25Retriever.from_documents(
        documents=docs_without_ids
    )
    bm25_retriever_with_some_ids = BM25Retriever.from_documents(
        documents=docs_with_some_ids
    )
    for doc in bm25_retriever_with_ids.docs:
        assert doc.id is not None
    for doc in bm25_retriever_without_ids.docs:
        assert doc.id is None
    for doc in bm25_retriever_with_some_ids.docs:
        if doc.page_content == "I have a pen.":
            assert doc.id == "1"
        elif doc.page_content == "Do you have a pen?":
            assert doc.id is None
        elif doc.page_content == "I have a bag.":
            assert doc.id == "3"
        elif doc.page_content == "I like LangChain.":
            assert doc.id is None
        else:
            raise ValueError("Unexpected document")
