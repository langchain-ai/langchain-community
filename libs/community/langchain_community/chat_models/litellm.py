"""
Deprecated LiteLLM wrapper.

⭐ Use `pip install -U langchain-litellm` and import
   `from langchain_litellm import ChatLiteLLM` instead.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

warnings.warn(
    "ChatLiteLLM in `langchain-community` is deprecated and will be removed.\n"
    "Install and import from `langchain-litellm` instead:\n\n"
    "  pip install -U langchain-litellm\n"
    "  from langchain_litellm import ChatLiteLLM\n",
    DeprecationWarning,
    stacklevel=2,
)

if TYPE_CHECKING:
    from langchain_litellm import ChatLiteLLM
else:

    class ChatLiteLLM:
        def __init__(self, *_, **__):
            raise ImportError(
                "ChatLiteLLM has moved to `langchain-litellm`.\n"
                "Install it with:\n"
                "  pip install -U langchain-litellm"
            )


__all__ = ["ChatLiteLLM"]
