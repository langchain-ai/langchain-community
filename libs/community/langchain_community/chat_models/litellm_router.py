"""
Deprecated LiteLLM Router wrapper.

⭐  Use `pip install -U langchain-litellm` and import
    `from langchain_litellm import ChatLiteLLMRouter` instead.
"""
from __future__ import annotations

import warnings

warnings.warn(
    "ChatLiteLLMRouter in `langchain-community` is deprecated and will be removed "
    "in a future release.\n"
    "Please install `langchain-litellm` and import ChatLiteLLMRouter from there:\n\n"
    "  pip install -U langchain-litellm\n"
    "  from langchain_litellm import ChatLiteLLMRouter\n",
    DeprecationWarning,
    stacklevel=2,
)

try:
    from langchain_litellm import ChatLiteLLMRouter
except ImportError as e:
    raise ImportError(
        "ChatLiteLLMRouter has been moved to the `langchain-litellm` package.\n"
        "Install it with:\n"
        "  pip install -U langchain-litellm"
    ) from e

__all__ = ["ChatLiteLLMRouter"]
