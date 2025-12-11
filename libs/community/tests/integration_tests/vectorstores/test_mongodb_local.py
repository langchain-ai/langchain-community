"""Test MongoDB Local Vector Search functionality."""

#THE TESTS FAIL BECAUSE OF THE DETERMINISTIC FAKE EMBEDDINGS, TESTS PASS AFTER USING PROPER EMBEDDING MODEL (eg: MINILM v6)
from __future__ import annotations


import os
from time import sleep
from typing import Any

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.embeddings import FakeEmbeddings
from langchain_community.embeddings import DeterministicFakeEmbedding

from langchain_community.vectorstores.mongodb_local import MongoDBLocalVectorSearch

INDEX_NAME = "langchain-test-index"
NAMESPACE = "langchain_test_db.langchain_test_collection"
CONNECTION_STRING = os.environ.get("MONGODB_LOCAL_URI")
DB_NAME, COLLECTION_NAME = NAMESPACE.split(".")
_dimension=368
_EMBEDDING_FUNCTION = DeterministicFakeEmbedding(size=_dimension)

def get_collection() -> Any:
    from pymongo import MongoClient
    test_client: MongoClient = MongoClient(CONNECTION_STRING)
    return test_client[DB_NAME][COLLECTION_NAME] #collection


@pytest.fixture()
def collection() -> Any:
    #returns the collection
    return get_collection()


class TestMongoDBLocalVectorSearch:
    @classmethod
    def setup_class(cls) -> None:
        #checks to see if the collection is empty
        collection = get_collection()
        assert collection.count_documents({}) == 0
    
    @classmethod
    def teardown_class(cls) -> None:
        collection = get_collection()
        #deletes all docs in the collection
        collection.delete_many({})

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        collection = get_collection()
        # delete all the documents in the collection
        collection.delete_many({})

    def test_from_documents_euclid_flat(
        self , collection: Any
    ) -> None:
        """Test end to end construction and search."""
        documents = [
            Document(page_content="Dogs are tough.", metadata={"a": 1}),
            Document(page_content="Cats have fluff.", metadata={"b": 1}),
            Document(page_content="What is a sandwich?", metadata={"c": 1}),
            Document(page_content="That fence is purple.", metadata={"d": 1, "e": 2}),
        ]
        faiss_config={
            'metric':'euclidean',
            'index_type':'flat'
            }
        
        texts=[]
        metadatas=[]
        for doc in documents:
            doc_json=doc.to_json()
            texts.append(doc.page_content)
            metadatas.append(doc.metadata)

        vectorstore = MongoDBLocalVectorSearch(
            faiss_config=faiss_config,
            embedder_model=_EMBEDDING_FUNCTION,
            collection=collection,
            index_name=INDEX_NAME,
        )
        vectorstore.add_texts(texts=texts,metadatas=metadatas)
        output = vectorstore.similarity_search(query="Sandwich", k=1)
        assert len(output)==1
        assert isinstance(output[0][0], Document)
        assert output[0][0].page_content == "What is a sandwich?"
        assert output[0][0].metadata["c"] == 1

    def test_from_documents_cosine_flat(
        self , collection: Any
    ) -> None:
        """Test end to end construction and search."""
        documents = [
            Document(page_content="Dogs are tough.", metadata={"a": 1}),
            Document(page_content="Cats have fluff.", metadata={"b": 1}),
            Document(page_content="What is a sandwich?", metadata={"c": 1}),
            Document(page_content="That fence is purple.", metadata={"d": 1, "e": 2}),
        ]
        faiss_config={
            'metric':'cosine',
            'index_type':'flat'
            }
        
        texts=[]
        metadatas=[]
        for doc in documents:
            doc_json=doc.to_json()
            texts.append(doc.page_content)
            metadatas.append(doc.metadata)

        vectorstore = MongoDBLocalVectorSearch(
            faiss_config=faiss_config,
            embedder_model=_EMBEDDING_FUNCTION,
            collection=collection,
            index_name=INDEX_NAME,
        )
        vectorstore.add_texts(texts=texts,metadatas=metadatas)
        output = vectorstore.similarity_search(query="Sandwich", k=1)
        assert len(output)==1
        assert isinstance(output[0][0], Document)
        assert output[0][0].page_content == "What is a sandwich?"
        assert output[0][0].metadata["c"] == 1
    
    def test_from_documents_dot_flat(
        self , collection: Any
    ) -> None:
        """Test end to end construction and search."""
        documents = [
            Document(page_content="Dogs are tough.", metadata={"a": 1}),
            Document(page_content="Cats have fluff.", metadata={"b": 1}),
            Document(page_content="What is a sandwich?", metadata={"c": 1}),
            Document(page_content="That fence is purple.", metadata={"d": 1, "e": 2}),
        ]
        faiss_config={
            'metric':'dot',
            'index_type':'flat',
            }
        
        texts=[]
        metadatas=[]
        for doc in documents:
            doc_json=doc.to_json()
            texts.append(doc.page_content)
            metadatas.append(doc.metadata)

        vectorstore = MongoDBLocalVectorSearch(
            faiss_config=faiss_config,
            embedder_model=_EMBEDDING_FUNCTION,
            collection=collection,
            index_name=INDEX_NAME,
        )
        vectorstore.add_texts(texts=texts,metadatas=metadatas)
        output = vectorstore.similarity_search(query="Sandwich", k=1)
        assert len(output)==1
        assert isinstance(output[0][0], Document)
        assert output[0][0].page_content == "What is a sandwich?"
        assert output[0][0].metadata["c"] == 1
    
    def test_from_documents_euclid_hnsw(
        self , collection: Any
    ) -> None:
        """Test end to end construction and search."""
        documents = [
            Document(page_content="Dogs are tough.", metadata={"a": 1}),
            Document(page_content="Cats have fluff.", metadata={"b": 1}),
            Document(page_content="What is a sandwich?", metadata={"c": 1}),
            Document(page_content="That fence is purple.", metadata={"d": 1, "e": 2}),
        ]
        faiss_config={
            'metric':'euclidean',
            'index_type':'hnsw',
            'index_params':{
                'dimensions':_dimension,
                'neighbours':32,
                'efSearch':64,
                'efConstruction':200,
                }
            }
        
        texts=[]
        metadatas=[]
        for doc in documents:
            doc_json=doc.to_json()
            texts.append(doc.page_content)
            metadatas.append(doc.metadata)

        vectorstore = MongoDBLocalVectorSearch(
            faiss_config=faiss_config,
            embedder_model=_EMBEDDING_FUNCTION,
            collection=collection,
            index_name=INDEX_NAME,
        )
        vectorstore.add_texts(texts=texts,metadatas=metadatas)
        output = vectorstore.similarity_search(query="Sandwich", k=1)
        assert len(output)==1
        assert isinstance(output[0][0], Document)
        assert output[0][0].page_content == "What is a sandwich?"
        assert output[0][0].metadata["c"] == 1

    def test_from_documents_cosine_hnsw(
        self , collection: Any
    ) -> None:
        """Test end to end construction and search."""
        documents = [
            Document(page_content="Dogs are tough.", metadata={"a": 1}),
            Document(page_content="Cats have fluff.", metadata={"b": 1}),
            Document(page_content="What is a sandwich?", metadata={"c": 1}),
            Document(page_content="That fence is purple.", metadata={"d": 1, "e": 2}),
        ]
        faiss_config={
            'metric':'cosine',
            'index_type':'hnsw',
            'index_params':{
                'dimensions':_dimension,
                'neighbours':32,
                'efSearch':64,
                'efConstruction':200,
                }
            }
        
        texts=[]
        metadatas=[]
        for doc in documents:
            doc_json=doc.to_json()
            texts.append(doc.page_content)
            metadatas.append(doc.metadata)

        vectorstore = MongoDBLocalVectorSearch(
            faiss_config=faiss_config,
            embedder_model=_EMBEDDING_FUNCTION,
            collection=collection,
            index_name=INDEX_NAME,
        )
        vectorstore.add_texts(texts=texts,metadatas=metadatas)
        output = vectorstore.similarity_search(query="Sandwich", k=1)
        assert len(output)==1
        assert isinstance(output[0][0], Document)
        assert output[0][0].page_content == "What is a sandwich?"
        assert output[0][0].metadata["c"] == 1

    def test_from_documents_dot_hnsw(
        self , collection: Any
    ) -> None:
        """Test end to end construction and search."""
        documents = [
            Document(page_content="Dogs are tough.", metadata={"a": 1}),
            Document(page_content="Cats have fluff.", metadata={"b": 1}),
            Document(page_content="What is a sandwich?", metadata={"c": 1}),
            Document(page_content="That fence is purple.", metadata={"d": 1, "e": 2}),
        ]
        faiss_config={
            'metric':'dot',
            'index_type':'hnsw',
            'index_params':{
                'dimensions':_dimension,
                'neighbours':32,
                'efSearch':64,
                'efConstruction':200,
                }
            }
        
        texts=[]
        metadatas=[]
        for doc in documents:
            doc_json=doc.to_json()
            texts.append(doc.page_content)
            metadatas.append(doc.metadata)

        vectorstore = MongoDBLocalVectorSearch(
            faiss_config=faiss_config,
            embedder_model=_EMBEDDING_FUNCTION,
            collection=collection,
            index_name=INDEX_NAME,
        )
        vectorstore.add_texts(texts=texts,metadatas=metadatas)
        output = vectorstore.similarity_search(query="Sandwich", k=1)
        assert len(output)==1
        assert isinstance(output[0][0], Document)
        assert output[0][0].page_content == "What is a sandwich?"
        assert output[0][0].metadata["c"] == 1
        
    def test_from_texts_euclid_flat(self, collection: Any) -> None:
        texts = [
            "Dogs are tough.",
            "Cats have fluff.",
            "What is a sandwich?",
            "That fence is purple.",
        ]
        faiss_config={
            'metric':'euclidean',
            'index_type':'flat'
        }
        vectorstore = MongoDBLocalVectorSearch.from_texts(
            texts,
            faiss_config=faiss_config,
            embedding=_EMBEDDING_FUNCTION,
            collection=collection,
            index_name=INDEX_NAME,
        )
        #no need to sleep as it immediately updates the index
        output = vectorstore.similarity_search("Sandwich", k=1)
        assert output[0][0].page_content == "What is a sandwich?"

    def test_from_texts_cosine_flat(self, collection: Any) -> None:
        texts = [
            "Dogs are tough.",
            "Cats have fluff.",
            "What is a sandwich?",
            "That fence is purple.",
        ]
        faiss_config={
            'metric':'cosine',
            'index_type':'flat'
        }
        vectorstore = MongoDBLocalVectorSearch.from_texts(
            texts,
            faiss_config=faiss_config,
            embedding=_EMBEDDING_FUNCTION,
            collection=collection,
            index_name=INDEX_NAME,
        )
        #no need to sleep as it immediately updates the index
        output = vectorstore.similarity_search("Sandwich", k=1)
        assert output[0][0].page_content == "What is a sandwich?"

    def test_from_texts_dot_flat(self, collection: Any) -> None:
        texts = [
            "Dogs are tough.",
            "Cats have fluff.",
            "What is a sandwich?",
            "That fence is purple.",
        ]
        faiss_config={
            'metric':'dot',
            'index_type':'flat'
        }
        vectorstore = MongoDBLocalVectorSearch.from_texts(
            texts,
            faiss_config=faiss_config,
            embedding=_EMBEDDING_FUNCTION,
            collection=collection,
            index_name=INDEX_NAME,
        )
        #no need to sleep as it immediately updates the index
        output = vectorstore.similarity_search("Sandwich", k=1)
        assert output[0][0].page_content == "What is a sandwich?"

    def test_from_texts_euclid_hnsw(self, collection: Any) -> None:
        texts = [
            "Dogs are tough.",
            "Cats have fluff.",
            "What is a sandwich?",
            "That fence is purple.",
        ]
        faiss_config={
            'metric':'euclidean',
            'index_type':'hnsw',
            'index_params':{
                'dimensions':_dimension,
                'neighbours':32,
                'efSearch':64,
                'efConstruction':200,
                }

        }
        vectorstore = MongoDBLocalVectorSearch.from_texts(
            texts,
            faiss_config=faiss_config,
            embedding=_EMBEDDING_FUNCTION,
            collection=collection,
            index_name=INDEX_NAME,
        )
        #no need to sleep as it immediately updates the index
        output = vectorstore.similarity_search("Sandwich", k=1)
        assert output[0][0].page_content == "What is a sandwich?"

    def test_from_texts_cosine_hnsw(self, collection: Any) -> None:
        texts = [
            "Dogs are tough.",
            "Cats have fluff.",
            "What is a sandwich?",
            "That fence is purple.",
        ]
        faiss_config={
            'metric':'cosine',
            'index_type':'hnsw',
            'index_params':{
                'dimensions':_dimension,
                'neighbours':32,
                'efSearch':64,
                'efConstruction':200,
                }
        }
        vectorstore = MongoDBLocalVectorSearch.from_texts(
            texts,
            faiss_config=faiss_config,
            embedding=_EMBEDDING_FUNCTION,
            collection=collection,
            index_name=INDEX_NAME,
        )
        #no need to sleep as it immediately updates the index
        output = vectorstore.similarity_search("Sandwich", k=1)
        assert output[0][0].page_content == "What is a sandwich?"   
    
    def test_from_texts_dot_hnsw(self, collection: Any) -> None:
        texts = [
            "Dogs are tough.",
            "Cats have fluff.",
            "What is a sandwich?",
            "That fence is purple.",
        ]
        faiss_config={
            'metric':'dot',
            'index_type':'hnsw',
            'index_params':{
                'dimensions':_dimension,
                'neighbours':32,
                'efSearch':64,
                'efConstruction':200,
                }
        }
        vectorstore = MongoDBLocalVectorSearch.from_texts(
            texts,
            faiss_config=faiss_config,
            embedding=_EMBEDDING_FUNCTION,
            collection=collection,
            index_name=INDEX_NAME,
        )
        #no need to sleep as it immediately updates the index
        output = vectorstore.similarity_search("Sandwich", k=1)
        assert output[0][0].page_content == "What is a sandwich?"  