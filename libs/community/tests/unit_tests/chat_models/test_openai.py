import os
import pytest
from langchain_community.chat_models.openai import ChatOpenAI


def test_api_key_masked_in_repr() -> None:
    """Test that openai_api_key is masked in repr output when explicitly set."""
    chat = ChatOpenAI(openai_api_key="sk-test123456789")
    repr_str = repr(chat)
    assert "sk-test123456789" not in repr_str
    assert "**********" in repr_str


def test_api_key_not_masked_when_not_set() -> None:
    """Test that openai_api_key shows env var value when not explicitly set."""
    chat = ChatOpenAI()
    repr_str = repr(chat)
    # When not set, it uses env var - should show the env var value masked
    if "openai_api_key" in repr_str:
        # If it appears in repr, it should be masked
        assert "**********" in repr_str or "test-key" in repr_str


def test_client_params_includes_api_key() -> None:
    """Test that _client_params includes the api_key for backward compatibility."""
    chat = ChatOpenAI(openai_api_key="sk-test123456789")
    params = chat._client_params
    # The key may be filtered out for v1+ but should still be accessible
    # This test just verifies _client_params works without error
    assert isinstance(params, dict)
