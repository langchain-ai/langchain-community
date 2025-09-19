from typing import Any, Dict, List, Tuple
from pydantic import BaseModel, ConfigDict, Field
from langchain_community.cross_encoders.base import BaseCrossEncoder

DEFAULT_MODEL_NAME = "BAAI/bge-reranker-base"

class HuggingFaceCrossEncoder(BaseModel, BaseCrossEncoder):
    """HuggingFace cross encoder models.
    
    Example:
        .. code-block:: python
            from langchain_community.cross_encoders import HuggingFaceCrossEncoder
            model_name = "BAAI/bge-reranker-base"
            model_kwargs = {'device': 'cpu'}
            hf = HuggingFaceCrossEncoder(
                model_name=model_name,
                model_kwargs=model_kwargs
            )
    """
    
    client: Any = None  #: :meta private:
    tokenizer: Any = None  #: :meta private:
    model_name: str = DEFAULT_MODEL_NAME
    """Model name to use."""
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Keyword arguments to pass to the model."""
    
    def __init__(self, **kwargs: Any):
        """Initialize the sentence_transformer."""
        super().__init__(**kwargs)
        try:
            import sentence_transformers
        except ImportError as exc:
            raise ImportError(
                "Could not import sentence_transformers python package. "
                "Please install it with `pip install sentence-transformers`."
            ) from exc
        
        self.client = sentence_transformers.CrossEncoder(
            self.model_name, **self.model_kwargs
        )
        
        # Store tokenizer reference for pad token management
        self.tokenizer = self.client.tokenizer
        
        # Ensure padding token is available
        self._ensure_padding_token()
    
    def _ensure_padding_token(self):
        """Ensure that a padding token is available for the tokenizer."""
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.client.config.pad_token_id = self.tokenizer.eos_token_id
            elif hasattr(self.tokenizer, 'unk_token') and self.tokenizer.unk_token:
                self.tokenizer.pad_token = self.tokenizer.unk_token
                self.client.config.pad_token_id = self.tokenizer.unk_token_id
            else:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                self.client.resize_token_embeddings(len(self.tokenizer))
                self.client.config.pad_token_id = self.tokenizer.pad_token_id
    
    model_config = ConfigDict(extra="forbid", protected_namespaces=())
    
    def score(self, text_pairs: List[Tuple[str, str]]) -> List[float]:
        """Compute similarity scores using a HuggingFace transformer model.
        
        Args:
            text_pairs: The list of text text_pairs to score the similarity.
        
        Returns:
            List of scores, one for each pair.
        """
        try:
            scores = self.client.predict(text_pairs)
        except ValueError as e:
            if "Cannot handle batch sizes > 1" in str(e) or "pad_token" in str(e).lower():
                # Fallback to processing pairs individually for models without pad_token
                scores = []
                for pair in text_pairs:
                    score = self.client.predict([pair])
                    scores.extend(score if isinstance(score, list) else [score])
            else:
                raise e
        
        # Some models e.g bert-multilingual-passage-reranking-msmarco
        # gives two score not_relevant and relevant as compare with the query.
        if hasattr(scores, 'shape') and len(scores.shape) > 1:  # we are going to get the relevant scores
            scores = [x[1] for x in scores]
        
        return scores
