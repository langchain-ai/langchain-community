from typing import Any, Dict, List, Tuple

from pydantic import BaseModel, ConfigDict, Field

from collections import defaultdict

from langchain_community.cross_encoders.base import BaseCrossEncoder

DEFAULT_MODEL_NAME = "mixedbread-ai/mxbai-rerank-base-v2"


class MixedbreadAICrossEncoder(BaseModel, BaseCrossEncoder):
"""Mixedbread cross encoder models.
    Args:
        model_name: The name or identifier of the Mixedbread AI model to use.
        model_kwargs: Additional keyword arguments to pass to the model.
        normalize_scores: Whether to normalize the scores returned by the model.
            Defaults to True.
    Example:
        .. code-block:: python
            from langchain_community.cross_encoders import MixedbreadAICrossEncoder
            model_name = "mixedbread-ai/mxbai-rerank-base-v2"
            model_kwargs = {'top_k': 10}
            mb = MixedbreadAICrossEncoder(
                model_name=model_name,
                model_kwargs=model_kwargs
            )
    """

    client: Any = None  #: :meta private:
    model_name: str = DEFAULT_MODEL_NAME
    """Model name to use."""
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Keyword arguments to pass to the model."""
    normalize_scores: bool = Field(default=True)
    """Whether to normalize scores to [0, 1] range."""

    def __init__(self, **kwargs: Any):
        """Initialize the mixbread reranker."""
        super().__init__(**kwargs)
        try:
            from mxbai_rerank import MxbaiRerankV2
        except ImportError as exc:
            raise ImportError(
                "Could not import mxbai_rerank python package. "
                "Please install it with `pip install mxbai-rerank`."
            ) from exc

        self.client = MxbaiRerankV2(self.model_name, **self.model_kwargs)

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalise scores to [0, 1] range."""
        if not scores:
            return scores
        
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return [0.0] * len(scores)
        
        return [(score - min_score) / (max_score - min_score) for score in scores]

    def score(self, text_pairs: List[Tuple[str, str]]) -> List[float]:
        """Compute similarity scores using a Mixbread transformer model.

        Args:
            text_pairs: The list of text pairs to score the similarity.
                       Each tuple should be (query, document).

        Returns:
            List of similarity/relevance scores between query-document pairs,
            one float score for each input pair.
        """
        if not text_pairs:
            return []

        query_groups = defaultdict(list)
        for i, (query, doc) in enumerate(text_pairs):
            query_groups[query].append((i, doc))
            
        
        # Process each query group
        scores = [0.0] * len(text_pairs)
        for query, doc_entries in query_groups.items():
            indices = [i for i, _ in doc_entries]
            documents = [doc for _, doc in doc_entries]
            
            results = self.client.rank(
                query,
                documents,
                return_documents=False,
                top_k=len(documents)
            )
            
            # Map scores back to original positions
            for res_idx, result in enumerate(results):
                orig_idx = indices[result.index]
                scores[orig_idx] = result.score

        # Normalize scores if requested
        if self.normalize_scores:
            scores = self._normalize_scores(scores)

        return scores
