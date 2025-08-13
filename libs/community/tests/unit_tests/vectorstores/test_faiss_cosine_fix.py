import numpy as np
import faiss
import pytest
from langchain_community.utils.math import cosine_similarity
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.docstore.in_memory import InMemoryDocstore

@pytest.mark.parametrize("query, docs", [
    ("new movie", ["good movie", "bad movie"]),
])
def test_faiss_cosine_scores_match_manual(query, docs):
    # Step 1: Initialize embeddings and FAISS index
    embeddings = FastEmbedEmbeddings()
    dim = len(embeddings.embed_query("hello world"))
    index = faiss.IndexFlatIP(dim)  # Inner product index for cosine
    store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
        distance_strategy=DistanceStrategy.COSINE,  # Use enum, not string
    )

    # Step 2: Add test documents
    store.add_texts(docs)

    # Step 3: LangChain similarity search
    results = store.similarity_search_with_relevance_scores(query)

    # Step 4: Manual cosine similarity check
    query_emb = embeddings.embed_query(query)
    for (doc, lc_score) in results:
        manual_score = cosine_similarity(
            [query_emb], [embeddings.embed_query(doc.page_content)]
        )[0][0]
        assert np.isclose(lc_score, manual_score, atol=1e-6), (
            f"Mismatch for '{doc.page_content}': "
            f"LangChain={lc_score}, Manual={manual_score}"
        )
