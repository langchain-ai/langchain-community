from __future__ import annotations

from copy import deepcopy
from enum import Enum
from importlib.metadata import version
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence

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
    """The path (Hugging Face model hub or local directory) to the reranker LLM. 
        Default is 'castorini/rank_zephyr_7b_v1_full' (a pre-trained RankZephyr model)."""
        
    top_n: int = Field(default=3)
    """The number of top-ranked passages/documents to return after reranking."""
    
    window_size: int = Field(default=20)
    """The number of passages/documents to consider in each sliding window during reranking."""
    
    context_size: int = Field(default=4096)
    """The maximum context length (in tokens) the model can handle for input sequences."""
    
    prompt_mode: str = Field(default="rank_GPT")
    """The style of prompt template to use for the reranker (e.g., 'rank_GPT' for GPT-style formatting)."""
    
    num_gpus: int = Field(default=1)
    """The number of GPUs to use for inference (for distributed/parallel processing)."""
    
    num_few_shot_examples: int = Field(default=0)
    """The number of few-shot examples to include in the prompt for potentially better reranking results."""
    
    few_shot_file: Optional[str] = Field(default=None)
    """Optional path to a json file containing few-shot examples for the reranker. Must include if num_few_shot_examples > 0"""
    
    use_logits: bool = Field(default=False)
    """This parameter enables FIRST (Faster Improved Reranking with Single Token Decoding), a technique that 
        examines only the logits (prediction scores) of the first token instead of generating the full ranking text."""
    
    use_alpha: bool = Field(default=False)
    """This parameter switches from numerical identifiers ([1], [2], [3]) to alphabetical identifiers ([A], [B], [C]) in prompts."""
    
    variable_passages: bool = Field(default=False)
    """This parameter indicates whether the model can handle a variable number of passages in the input prompt.
        If True, the model adapts its prompt examples based on the actual number of documents being ranked."""
    
    stride: int = Field(default=10)
    """This parameter controls the step size for the sliding window algorithm, determining how much the window 
        advances between iterations."""
    
    use_azure_openai: bool = Field(default=False)
    """If True, the system configures the OpenAI client to use Azure's OpenAI service instead of OpenAI's direct API."""
    
    model_coordinator: Any = Field(default=None, exclude=True)
    """If a default model_coordinator (object with custom logic that is used to perform reranking) is passed"""
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate the imports and param values, if no error, create a model coordinator"""
        
        """
        Try to import rank_llm modules:
            Reranker: main class for performing document/passage reranking
            PromptMode: Enum defining different prompt template styles
            Azure/OpenAI/GENAI API utility functions
            Listwise reranking implementations
        """
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
    
        # If no model_coordinator object is passed in the params, create a custom model coordinator based on the other params
        if values.get("model_coordinator") is None:
            model_path = values.get("model_path", "rank_zephyr") # default reranking model is castorini/rank_zephyr_7b_v1_full
            
            # Create a dictionary for all the params needed to create a model_coordinator and their default values
            kwargs = {
                "model_path": model_path,
                "default_model_coordinator": None,
                "context_size": values.get("context_size", 4096),
                "prompt_mode": values.get("prompt_mode", PromptMode.RANK_GPT),
                "num_gpus": values.get("num_gpus", 1),
                "use_logits": values.get("use_logits", False),
                "use_alpha": values.get("use_alpha", False),
                "num_few_shot_examples": values.get("num_few_shot_examples", 0),
                "few_shot_file": values.get("few_shot_file", None),
                "variable_passages": values.get("variable_passages", False),
                "interactive": False,
                "window_size": values.get("window_size", 20),
                "use_azure_openai": values.get("use_azure_openai", False),
            }
            
            # create the model coordinator from the dict above
            values["model_coordinator"] = Reranker.create_model_coordinator(**kwargs)
    
        return values

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """
        Compresses documents to return only the top_n most relevant documents
        
        Parameters:
        - documents (Sequence[Document]): The list of documents to compress and retrieve from
        - query (str): The query string used to return the top_n most relevant documents
        """
        # Use the rank_llm abstract data structure to create a request object based on the query and documents provided
        request = Request(
            query=Query(text=query, qid=1),
            candidates=[
                Candidate(doc={"text": doc.page_content}, docid=index, score=1)
                for index, doc in enumerate(documents)
            ],
        )
        
        # Instantiate a Reranker object based on the created/passed model_coordinator
        reranker = Reranker(self.model_coordinator)

        # Use the Reranker object to perform reranking on the request object
        rerank_results = reranker.rerank(
            request,
            rank_end=len(documents),
            window_size=min(20, len(documents)),
            stride=self.stride
        )

        # Handle results
        reranked_candidates = rerank_results.candidates if hasattr(rerank_results, "candidates") else rerank_results
        
        # Create new Document objects with original metadata
        final_results = []
        for candidate in reranked_candidates[:self.top_n]: # select only the top_n most relevant documents
            orig_idx = int(candidate.docid)
            if orig_idx < len(documents):
                doc = documents[orig_idx]
                new_doc = Document(
                    page_content=doc.page_content,
                    metadata=deepcopy(doc.metadata)
                )
                final_results.append(new_doc)
        
        return final_results