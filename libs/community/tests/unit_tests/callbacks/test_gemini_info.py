from unittest.mock import MagicMock
from uuid import uuid4

import numpy as np
import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from langchain_community.callbacks.gemini_info import GeminiCallbackHandler


@pytest.fixture
def handler() -> GeminiCallbackHandler:
    return GeminiCallbackHandler()


def test_on_llm_end(handler: GeminiCallbackHandler) -> None:
    response = LLMResult(
        generations=[],
        llm_output={
            "token_usage": {
                "prompt_tokens": 2,
                "completion_tokens": 1,
                "total_tokens": 3,
            },
            "model_name": "gemini-2.5-pro",
        },
    )
    handler.on_llm_end(response)
    assert handler.successful_requests == 1
    assert handler.total_tokens == 3
    assert handler.prompt_tokens == 2
    assert handler.completion_tokens == 1
    assert handler.total_cost > 0


def test_on_llm_end_with_chat_generation(handler: GeminiCallbackHandler) -> None:
    """Test handling of ChatGeneration with usage_metadata in AIMessage.

    Note: The Gemini callback currently doesn't parse usage_metadata from
    ChatGeneration messages, it only looks at llm_output["token_usage"].
    This test verifies the current behavior.
    """
    response = LLMResult(
        generations=[
            [
                ChatGeneration(
                    text="Hello, world!",
                    message=AIMessage(
                        content="Hello, world!",
                        usage_metadata={
                            "input_tokens": 2,
                            "output_tokens": 2,
                            "total_tokens": 4,
                        },
                    ),
                )
            ]
        ],
        llm_output={
            "model_name": "gemini-2.5-pro",
        },
    )
    handler.on_llm_end(response)
    assert handler.successful_requests == 1
    # Since there's no token_usage in llm_output, tokens should be 0
    assert handler.total_tokens == 4
    assert handler.prompt_tokens == 2
    assert handler.completion_tokens == 2
    assert handler.total_cost > 0


def test_on_llm_end_custom_model(handler: GeminiCallbackHandler) -> None:
    response = LLMResult(
        generations=[],
        llm_output={
            "token_usage": {
                "prompt_tokens": 2,
                "completion_tokens": 1,
                "total_tokens": 3,
            },
            "model_name": "foo-bar",
        },
    )
    handler.on_llm_end(response)
    assert handler.total_cost == 0


@pytest.mark.parametrize(
    "model_name, expected_cost",
    [
        ("gemini-2.5-pro", 0.01125),
        ("gemini-1.5-pro", 0.00625),
        ("gemini-2.5-flash-lite", 0.0005),
    ],
)
def test_on_llm_end_gemini_model(
    handler: GeminiCallbackHandler, model_name: str, expected_cost: float
) -> None:
    response = LLMResult(
        generations=[],
        llm_output={
            "token_usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 1000,
                "total_tokens": 2000,
            },
            "model_name": model_name,
        },
    )
    handler.on_llm_end(response)
    assert np.isclose(handler.total_cost, expected_cost)


@pytest.mark.parametrize("model_name", ["unknown-model", "gpt-4", "claude-3"])
def test_on_llm_end_no_cost_invalid_model(
    handler: GeminiCallbackHandler, model_name: str
) -> None:
    """Test that unknown models result in zero cost."""
    response = LLMResult(
        generations=[],
        llm_output={
            "token_usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 1000,
                "total_tokens": 2000,
            },
            "model_name": model_name,
        },
    )
    handler.on_llm_end(response)
    assert handler.total_cost == 0


def test_on_llm_end_no_llm_output(handler: GeminiCallbackHandler) -> None:
    """Test behavior when llm_output is None."""
    response = LLMResult(
        generations=[],
        llm_output=None,
    )
    handler.on_llm_end(response)
    # When llm_output is None, the handler returns early and doesn't increment
    assert handler.successful_requests == 0
    assert handler.total_tokens == 0
    assert handler.prompt_tokens == 0
    assert handler.completion_tokens == 0
    assert handler.total_cost == 0


def test_on_llm_end_no_token_usage(handler: GeminiCallbackHandler) -> None:
    """Test behavior when token_usage is missing from llm_output."""
    response = LLMResult(
        generations=[],
        llm_output={
            "model_name": "gemini-2.5-pro",
        },
    )
    handler.on_llm_end(response)
    assert handler.successful_requests == 1
    assert handler.total_tokens == 0
    assert handler.prompt_tokens == 0
    assert handler.completion_tokens == 0
    assert handler.total_cost == 0


def test_multiple_requests_accumulation(handler: GeminiCallbackHandler) -> None:
    """Test that multiple requests accumulate correctly."""
    # First request
    response1 = LLMResult(
        generations=[],
        llm_output={
            "token_usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            },
            "model_name": "gemini-2.5-pro",
        },
    )
    handler.on_llm_end(response1)

    # Second request
    response2 = LLMResult(
        generations=[],
        llm_output={
            "token_usage": {
                "prompt_tokens": 200,
                "completion_tokens": 100,
                "total_tokens": 300,
            },
            "model_name": "gemini-1.5-pro",
        },
    )
    handler.on_llm_end(response2)

    assert handler.successful_requests == 2
    assert handler.total_tokens == 450
    assert handler.prompt_tokens == 300
    assert handler.completion_tokens == 150
    assert handler.total_cost > 0


def test_on_llm_start_no_op(handler: GeminiCallbackHandler) -> None:
    """Test that on_llm_start does nothing (no-op)."""
    # This should not raise any exceptions
    handler.on_llm_start({}, ["test prompt"])


def test_on_llm_new_token_no_op(handler: GeminiCallbackHandler) -> None:
    """Test that on_llm_new_token does nothing (no-op)."""
    # This should not raise any exceptions
    handler.on_llm_new_token("test")


def test_handler_copy(handler: GeminiCallbackHandler) -> None:
    """Test handler copy methods."""
    import copy

    # Test shallow copy
    handler_copy = copy.copy(handler)
    assert handler_copy is handler  # Should return the same instance

    # Test deep copy
    handler_deepcopy = copy.deepcopy(handler)
    assert handler_deepcopy is handler  # Should return the same instance


def test_on_retry_works(handler: GeminiCallbackHandler) -> None:
    handler.on_retry(MagicMock(), run_id=uuid4())
