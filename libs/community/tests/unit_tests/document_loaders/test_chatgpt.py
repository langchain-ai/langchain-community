"""Unit tests for ChatGPTLoader."""

import json
import tempfile
from pathlib import Path

import pytest

from langchain_community.document_loaders.chatgpt import ChatGPTLoader


@pytest.fixture
def sample_chatgpt_export() -> dict:
    """Sample ChatGPT export data with multiple conversations."""
    return [
        {
            "title": "Conversation 1",
            "mapping": {
                "msg1": {
                    "message": {
                        "author": {"role": "system"},
                        "content": {"parts": ["System message"]},
                        "create_time": 1704067200.0,  # 2024-01-01 00:00:00
                    }
                },
                "msg2": {
                    "message": {
                        "author": {"role": "user"},
                        "content": {"parts": ["Hello!"]},
                        "create_time": 1704067260.0,
                    }
                },
                "msg3": {
                    "message": {
                        "author": {"role": "assistant"},
                        "content": {"parts": ["Hi there!"]},
                        "create_time": 1704067320.0,
                    }
                },
            },
        },
        {
            "title": "Conversation 2",
            "mapping": {
                "msg1": {
                    "message": {
                        "author": {"role": "system"},
                        "content": {"parts": ["System message"]},
                        "create_time": 1704153600.0,  # 2024-01-02 00:00:00
                    }
                },
                "msg2": {
                    "message": {
                        "author": {"role": "user"},
                        "content": {"parts": ["How are you?"]},
                        "create_time": 1704153660.0,
                    }
                },
            },
        },
        {
            "title": "Conversation 3",
            "mapping": {
                "msg1": {
                    "message": {
                        "author": {"role": "system"},
                        "content": {"parts": ["System message"]},
                        "create_time": 1704240000.0,  # 2024-01-03 00:00:00
                    }
                },
                "msg2": {
                    "message": {
                        "author": {"role": "user"},
                        "content": {"parts": ["Goodbye!"]},
                        "create_time": 1704240060.0,
                    }
                },
            },
        },
    ]



@pytest.fixture
def chatgpt_export_file(sample_chatgpt_export: dict) -> str:
    """Create a temporary file with sample ChatGPT export data."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        json.dump(sample_chatgpt_export, f)
        temp_file = f.name
    
    yield temp_file
    
    Path(temp_file).unlink(missing_ok=True)


def test_load_all_conversations_default(chatgpt_export_file: str) -> None:
    """Test that default num_logs=0 loads all conversations."""
    loader = ChatGPTLoader(chatgpt_export_file)
    documents = loader.load()

    assert len(documents) == 3
    assert "Conversation 1" in documents[0].page_content
    assert "Conversation 2" in documents[1].page_content
    assert "Conversation 3" in documents[2].page_content


def test_load_all_conversations_explicit_zero(chatgpt_export_file: str) -> None:
    """Test that num_logs=0 explicitly loads all conversations."""
    loader = ChatGPTLoader(chatgpt_export_file, num_logs=0)
    documents = loader.load()

    assert len(documents) == 3


def test_load_limited_conversations(chatgpt_export_file: str) -> None:
    """Test that num_logs limits the number of conversations loaded."""
    loader = ChatGPTLoader(chatgpt_export_file, num_logs=2)
    documents = loader.load()

    assert len(documents) == 2
    assert "Conversation 1" in documents[0].page_content
    assert "Conversation 2" in documents[1].page_content


def test_load_single_conversation(chatgpt_export_file: str) -> None:
    """Test loading only one conversation."""
    loader = ChatGPTLoader(chatgpt_export_file, num_logs=1)
    documents = loader.load()

    assert len(documents) == 1
    assert "Conversation 1" in documents[0].page_content


def test_load_more_than_available(chatgpt_export_file: str) -> None:
    """Test that requesting more logs than available returns all."""
    loader = ChatGPTLoader(chatgpt_export_file, num_logs=100)
    documents = loader.load()

    assert len(documents) == 3


def test_system_message_excluded(chatgpt_export_file: str) -> None:
    """Test that system messages at the start are excluded."""
    loader = ChatGPTLoader(chatgpt_export_file, num_logs=1)
    documents = loader.load()

    # System message should not appear in the content
    assert "System message" not in documents[0].page_content
    # But user and assistant messages should
    assert "Hello!" in documents[0].page_content
    assert "Hi there!" in documents[0].page_content


def test_metadata_contains_source(chatgpt_export_file: str) -> None:
    """Test that document metadata contains the source file path."""
    loader = ChatGPTLoader(chatgpt_export_file)
    documents = loader.load()

    for doc in documents:
        assert "source" in doc.metadata
        assert doc.metadata["source"] == chatgpt_export_file


def test_single_conversation_not_empty(sample_chatgpt_export: dict) -> None:
    """Test that loading a single conversation file returns non-empty result.

    This is a regression test for issue #465 where the last message was
    missed due to incorrect slicing with default num_logs=-1.
    """
    # Create a file with only one conversation
    single_conversation = [sample_chatgpt_export[0]]

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        json.dump(single_conversation, f)
        temp_file = f.name

    loader = ChatGPTLoader(temp_file)
    documents = loader.load()

    # Should return exactly 1 document, not 0 (the bug)
    assert len(documents) == 1
    assert "Hello!" in documents[0].page_content
