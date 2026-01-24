"""Causal Knowledge Graph Retriever for zero-hallucination RAG."""

from typing import Any, Dict, List, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


class CausalRetriever(BaseRetriever):
    """.causal knowledge graph retriever for deterministic, zero-hallucination RAG.

    The .causal format is a binary knowledge graph with embedded inference.
    Unlike vector-based RAG which returns "similar" text, CausalRetriever
    returns logically connected facts with full provenance.

    Setup:
        Install ``dotcausal``:

        .. code-block:: bash

            pip install -U dotcausal

    Key init args:
        file_path: str
            Path to the .causal knowledge graph file
        top_k: int
            Maximum number of results to return (default: 10)
        include_inferred: bool
            Whether to include inferred triplets (default: True)
        min_confidence: float
            Minimum confidence threshold (default: 0.0)

    Instantiate:
        .. code-block:: python

            from langchain_community.retrievers import CausalRetriever

            retriever = CausalRetriever(
                file_path="knowledge.causal",
                top_k=10,
                include_inferred=True,
            )

    Usage:
        .. code-block:: python

            docs = retriever.invoke("COVID fatigue")
            docs[0].page_content

        .. code-block:: none

            '[INFERRED] SARS-CoV-2 → damages → mitochondria → causes → chronic fatigue'

        .. code-block:: python

            docs[0].metadata

        .. code-block:: none

            {'trigger': 'SARS-CoV-2',
             'mechanism': 'indirectly causes',
             'outcome': 'chronic fatigue',
             'confidence': 0.765,
             'is_inferred': True,
             'provenance': ['triplet_123', 'triplet_456']}

    Use within a chain:
        .. code-block:: python

            from langchain_core.output_parsers import StrOutputParser
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.runnables import RunnablePassthrough
            from langchain_openai import ChatOpenAI

            prompt = ChatPromptTemplate.from_template(
                \"\"\"Answer based on these verified facts (zero hallucination):

            {context}

            Question: {question}\"\"\"
            )

            llm = ChatOpenAI(model="gpt-4")

            def format_docs(docs):
                return "\\n".join(doc.page_content for doc in docs)

            chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )

            chain.invoke("What mechanisms connect COVID to chronic fatigue?")

    References:
        - dotcausal PyPI: https://pypi.org/project/dotcausal/
        - GitHub: https://github.com/DT-Foss/dotcausal
        - Whitepaper: https://doi.org/10.5281/zenodo.18326222
    """

    file_path: str
    """Path to the .causal knowledge graph file."""

    top_k: int = 10
    """Maximum number of results to return."""

    include_inferred: bool = True
    """Whether to include inferred (derived) triplets."""

    min_confidence: float = 0.0
    """Minimum confidence threshold for results."""

    _reader: Optional[Any] = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        file_path: str,
        top_k: int = 10,
        include_inferred: bool = True,
        min_confidence: float = 0.0,
        **kwargs: Any,
    ) -> None:
        """Initialize the CausalRetriever.

        Args:
            file_path: Path to the .causal knowledge graph file
            top_k: Maximum number of results to return
            include_inferred: Whether to include inferred triplets
            min_confidence: Minimum confidence threshold
        """
        super().__init__(
            file_path=file_path,
            top_k=top_k,
            include_inferred=include_inferred,
            min_confidence=min_confidence,
            **kwargs,
        )
        self._load_reader()

    def _load_reader(self) -> None:
        """Load the .causal file using dotcausal."""
        try:
            from dotcausal import CausalReader
        except ImportError as e:
            raise ImportError(
                "Could not import dotcausal. "
                "Please install it with: pip install dotcausal"
            ) from e

        from pathlib import Path

        if not Path(self.file_path).exists():
            raise FileNotFoundError(f"Causal file not found: {self.file_path}")

        self._reader = CausalReader(self.file_path)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        """Retrieve relevant documents from the knowledge graph.

        Args:
            query: The search query
            run_manager: Callback manager (optional)

        Returns:
            List of Document objects with causal triplets
        """
        if self._reader is None:
            raise ValueError("Reader not initialized. Call _load_reader() first.")

        # Search the knowledge graph
        results = self._reader.search(
            query=query,
            limit=self.top_k * 2,  # Get more to filter
        )

        # Filter by confidence and inferred status
        filtered = []
        for r in results:
            if r.get("confidence", 1.0) < self.min_confidence:
                continue
            if not self.include_inferred and r.get("is_inferred", False):
                continue
            filtered.append(r)
            if len(filtered) >= self.top_k:
                break

        # Convert to LangChain Documents
        documents = []
        for r in filtered:
            tag = "[INFERRED]" if r.get("is_inferred") else "[EXPLICIT]"
            content = f"{tag} {r['trigger']} → {r['mechanism']} → {r['outcome']}"

            metadata: Dict[str, Any] = {
                "trigger": r["trigger"],
                "mechanism": r["mechanism"],
                "outcome": r["outcome"],
                "confidence": r.get("confidence", 1.0),
                "is_inferred": r.get("is_inferred", False),
                "source": r.get("source", ""),
                "provenance": r.get("provenance", []),
            }

            documents.append(Document(page_content=content, metadata=metadata))

        return documents

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the loaded knowledge graph.

        Returns:
            Dictionary with triplet counts and amplification stats
        """
        if self._reader is None:
            raise ValueError("Reader not initialized.")
        return self._reader.get_stats()
