import threading
from enum import Enum, auto
from typing import Any, Dict, List

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

MODEL_COST_PER_1M_TOKENS = {
    "gemini-2.5-pro": 1.25,
    "gemini-2.5-pro-completion": 10,
    "gemini-2.5-flash": 0.3,
    "gemini-2.5-flash-completion": 2.5,
    "gemini-2.5-flash-lite": 0.1,
    "gemini-2.5-flash-lite-completion": 0.4,
    "gemini-2.0-flash": 0.1,
    "gemini-2.0-flash-completion": 0.4,
    "gemini-2.0-flash-lite": 0.075,
    "gemini-2.0-flash-lite-completion": 0.3,
    "gemini-1.5-pro": 1.25,
    "gemini-1.5-pro-completion": 5,
    "gemini-1.5-flash": 0.075,
    "gemini-1.5-flash-completion": 0.3,
}
MODEL_COST_PER_1K_TOKENS = {k: v / 1000 for k, v in MODEL_COST_PER_1M_TOKENS.items()}


class TokenType(Enum):
    """Token type enum."""

    PROMPT = auto()
    PROMPT_CACHED = auto()
    COMPLETION = auto()



def standardize_model_name(
    model_name: str,
    token_type: TokenType = TokenType.PROMPT,
) -> str:
    """Standardize the model name to a format that can be used in the Gemini API.
    
    Args:
        model_name: The name of the model to standardize.
        token_type: The type of token, defaults to PROMPT.
    """
    model_name = model_name.lower()
    if token_type == TokenType.COMPLETION:
        return model_name + "-completion"
    else:
        return model_name



def get_gemini_token_cost_for_model(
    model_name: str,
    num_tokens: int,
    is_completion: bool = False,
    *,
    token_type: TokenType = TokenType.PROMPT,
) -> float:
    """Get the cost in USD for a given model and number of tokens."""
    if is_completion:
        token_type = TokenType.COMPLETION
    model_name = standardize_model_name(model_name, token_type=token_type)
    if model_name not in MODEL_COST_PER_1K_TOKENS:
        raise ValueError(
            f"Unknown model: {model_name}. Please provide a valid Gemini model name. Known models are: "
            + ", ".join(MODEL_COST_PER_1K_TOKENS.keys())
        )
    return MODEL_COST_PER_1K_TOKENS[model_name] * (num_tokens / 1000)


class GeminiCallbackHandler(BaseCallbackHandler):
    """Callback Handler that tracks Gemini info."""

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.prompt_tokens_cached = 0
        self.completion_tokens = 0
        self.reasoning_tokens = 0
        self.successful_requests = 0
        self.total_cost = 0.0
    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        return f"""Tokens Used: {self.total_tokens}
\tPrompt Tokens: {self.prompt_tokens}
\tPrompt Cached Tokens: {self.prompt_tokens_cached}
\tCompletion Tokens: {self.completion_tokens}
\tReasoning Tokens: {self.reasoning_tokens}
Successful Requests: {self.successful_requests}
Total Cost (USD): ${self.total_cost}"""

    @property
    def always_verbose(self) -> bool:
        """Whether to call verbose callbacks even if verbose is False."""
        return True

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Print out the prompts."""
        pass

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Print out the token."""
        pass

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Collect token usage."""
        # Check for usage_metadata (langchain-core >= 0.2.2)
        try:
            generation = response.generations[0][0]
        except IndexError:
            generation = None
        if isinstance(generation, ChatGeneration):
            try:
                message = generation.message
                if isinstance(message, AIMessage):
                    usage_metadata = message.usage_metadata
                    response_metadata = message.response_metadata
                else:
                    usage_metadata = None
                    response_metadata = None
            except AttributeError:
                usage_metadata = None
                response_metadata = None
        else:
            usage_metadata = None
            response_metadata = None

        prompt_tokens_cached = 0
        reasoning_tokens = 0

        if usage_metadata:
            token_usage = {"total_tokens": usage_metadata["total_tokens"]}
            completion_tokens = usage_metadata["output_tokens"]
            prompt_tokens = usage_metadata["input_tokens"]
            if response_model_name := (response_metadata or {}).get("model_name"):
                model_name = standardize_model_name(response_model_name)
            elif response.llm_output is None:
                model_name = ""
            else:
                model_name = standardize_model_name(
                    response.llm_output.get("model_name", "")
                )
            if "cache_read" in usage_metadata.get("input_token_details", {}):
                prompt_tokens_cached = usage_metadata["input_token_details"][
                    "cache_read"
                ]
            if "reasoning" in usage_metadata.get("output_token_details", {}):
                reasoning_tokens = usage_metadata["output_token_details"]["reasoning"]
        else:
            if response.llm_output is None:
                return None

            if "token_usage" not in response.llm_output:
                with self._lock:
                    self.successful_requests += 1
                return None

            # compute tokens and cost for this request
            token_usage = response.llm_output["token_usage"]
            completion_tokens = token_usage.get("completion_tokens", 0)
            prompt_tokens = token_usage.get("prompt_tokens", 0)
            model_name = standardize_model_name(
                response.llm_output.get("model_name", "")
            )

        if model_name in MODEL_COST_PER_1K_TOKENS:
            uncached_prompt_tokens = prompt_tokens - prompt_tokens_cached
            uncached_prompt_cost = get_gemini_token_cost_for_model(
                model_name, uncached_prompt_tokens, token_type=TokenType.PROMPT
            )
            cached_prompt_cost = get_gemini_token_cost_for_model(
                model_name, prompt_tokens_cached, token_type=TokenType.PROMPT_CACHED
            )
            prompt_cost = uncached_prompt_cost + cached_prompt_cost
            completion_cost = get_gemini_token_cost_for_model(
                model_name, completion_tokens, token_type=TokenType.COMPLETION
            )
        else:
            completion_cost = 0
            prompt_cost = 0

        # update shared state behind lock
        with self._lock:
            self.total_cost += prompt_cost + completion_cost
            self.total_tokens += token_usage.get("total_tokens", 0)
            self.prompt_tokens += prompt_tokens
            self.prompt_tokens_cached += prompt_tokens_cached
            self.completion_tokens += completion_tokens
            self.reasoning_tokens += reasoning_tokens
            self.successful_requests += 1

    def __copy__(self) -> "GeminiCallbackHandler":
        """Return a copy of the callback handler."""
        return self

    def __deepcopy__(self, memo: Any) -> "GeminiCallbackHandler":
        """Return a deep copy of the callback handler."""
        return self
