"""
Deprecated LiteLLM wrapper.

⭐  Use `pip install langchain-litellm` and import
    `from langchain_litellm import ChatLiteLLM` instead.
"""
from __future__ import annotations

import warnings

warnings.warn(
    "ChatLiteLLM in `langchain-community` is deprecated and will be removed in a future release.\n"
    "Please install `langchain-litellm` and import ChatLiteLLM from there:\n\n"
    "  pip install langchain-litellm\n"
    "  from langchain_litellm import ChatLiteLLM\n",
    DeprecationWarning,
    stacklevel=2,
)

try:
    from langchain_litellm import ChatLiteLLM
except ImportError as e:
    raise ImportError(
        "ChatLiteLLM has been moved to the `langchain-litellm` package.\n"
        "Install it with:\n"
        "  pip install -U langchain-litellm"
    ) from e

__all__ = ["ChatLiteLLM"]
