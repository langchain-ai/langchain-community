from __future__ import annotations

from copy import deepcopy
from enum import Enum
from importlib.metadata import version
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence

from rank_llm.rerank.pairwise.pairwise_rankllm import PromptMode

from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from langchain_core.callbacks.manager import Callbacks
from langchain_core.documents import Document
from langchain_core.utils import get_from_dict_or_env
from packaging.version import Version
from pydantic import ConfigDict, Field, PrivateAttr, model_validator

if TYPE_CHECKING:
    from rank_llm.data import Candidate, Query, Request
else:
    # Avoid pydantic annotation issues when actually instantiating
    # while keeping this import optional
    try:
        from rank_llm.data import Candidate, Query, Request
    except ImportError:
        pass


class RankLLMRerank(BaseDocumentCompressor):
    """Document compressor using RankLLM with dynamic model coordinator creation."""
    
    model_path: str = Field(default="rank_zephyr")
    top_n: int = Field(default=3)
    window_size: int = Field(default=20)
    context_size: int = Field(default=4096)
    prompt_mode: Any = Field(default=PromptMode.RANK_GPT)
    stride: int = Field(default=10)
    openai_api_key: Optional[str] = Field(default=None)
    genai_api_key: Optional[str] = Field(default=None)
    use_azure_openai: bool = Field(default=False)
    model_coordinator: Any = Field(default=None, exclude=True)
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Dict:
        """Create the appropriate RankLLM model coordinator based on model_path."""
        
        try:
            from rank_llm.rerank.reranker import Reranker
            from rank_llm.rerank import (
                PromptMode,
                get_azure_openai_args,
                get_genai_api_key,
                get_openai_api_key,
            )
            from rank_llm.rerank.listwise import SafeOpenai, SafeGenai
        except ImportError as e:
            raise ImportError(
                "Could not import rank_llm python package. "
                "Please install it with `pip install rank_llm`."
            ) from e
    
        if values.get("model_coordinator") is None:
            model_path = values.get("model_path", "rank_zephyr")
            kwargs = {
                "model_path": model_path,
                "default_model_coordinator": None,
                "context_size": values.get("context_size", 4096),
                "prompt_mode": values.get("prompt_mode", PromptMode.RANK_GPT),
                "interactive": False,
                "window_size": values.get("window_size", 20),
                "stride": values.get("stride", 10),
                "use_azure_openai": values.get("use_azure_openai", False),
            }
    
            if "gpt" in model_path or kwargs["use_azure_openai"]:
                kwargs["openai_api_key"] = values.get("openai_api_key") or get_from_dict_or_env(
                    values, "openai_api_key", "OPENAI_API_KEY"
                )
            elif "gemini" in model_path:
                kwargs["genai_api_key"] = values.get("genai_api_key") or get_genai_api_key()
    
            from rank_llm.rerank.reranker import Reranker
            values["client"] = Reranker.create_model_coordinator(**kwargs)
    
        return values

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        request = Request(
            query=Query(text=query, qid=1),
            candidates=[
                Candidate(doc={"text": doc.page_content}, docid=index, score=1)
                for index, doc in enumerate(documents)
            ],
        )

        rerank_results = self.client.rerank(
            request,
            rank_end=len(documents),
            window_size=min(20, len(documents)),
            step=10,
        )

        # Handle results
        reranked_candidates = rerank_results.candidates if hasattr(rerank_results, "candidates") else rerank_results
        
        # Create new Document objects with original metadata
        final_results = []
        for candidate in reranked_candidates[:self.top_n]:
            orig_idx = int(candidate.docid)
            if orig_idx < len(documents):
                doc = documents[orig_idx]
                new_doc = Document(
                    page_content=doc.page_content,
                    metadata=deepcopy(doc.metadata)
                )
                final_results.append(new_doc)
        
        return final_results