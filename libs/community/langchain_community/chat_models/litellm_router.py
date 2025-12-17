"""
Deprecated LiteLLM Router wrapper.

⭐ Use `pip install -U langchain-litellm` and import
   `from langchain_litellm import ChatLiteLLMRouter` instead.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

warnings.warn(
    "ChatLiteLLMRouter in `langchain-community` is deprecated and will be removed.\n"
    "Install and import from `langchain-litellm` instead:\n\n"
    "  pip install -U langchain-litellm\n"
    "  from langchain_litellm import ChatLiteLLMRouter\n",
    DeprecationWarning,
    stacklevel=2,
)

if TYPE_CHECKING:
    from langchain_litellm import ChatLiteLLMRouter
else:

    class ChatLiteLLMRouter:
        def __init__(self, *_, **__):
            raise ImportError(
                "ChatLiteLLMRouter has moved to `langchain-litellm`.\n"
                "Install it with:\n"
                "  pip install -U langchain-litellm"
            )


__all__ = ["ChatLiteLLMRouter"]
