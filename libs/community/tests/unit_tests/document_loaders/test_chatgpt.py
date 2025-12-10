import json
from pathlib import Path

from langchain_community.document_loaders.chatgpt import ChatGPTLoader


def test_chatgpt_loader_load(tmp_path: Path) -> None:
    """Test loading conversations from a fake ChatGPT export file."""

    # 1. Arrange: Prepare a mock ChatGPT JSON file
    file_path = tmp_path / "logs.json"
    mock_data = [
        {
            "title": "Test Conversation",
            "create_time": 1678888888,
            "mapping": {
                "aaa-bbb-ccc": {
                    "message": {
                        "author": {"role": "system"},
                        "create_time": 1678888888.0,
                        "content": {"parts": ["System prompt"]},
                    }
                },
                "ddd-eee-fff": {
                    "message": {
                        "author": {"role": "user"},
                        "create_time": 1678888890.0,
                        "content": {"parts": ["Hello AI"]},
                    }
                },
                "ggg-hhh-iii": {
                    "message": {
                        "author": {"role": "assistant"},
                        "create_time": 1678888900.0,
                        "content": {"parts": ["Hello Human"]},
                    }
                },
            },
        }
    ]
    file_path.write_text(json.dumps(mock_data), encoding="utf-8")

    # 2. Act: Run Loader
    loader = ChatGPTLoader(str(file_path))
    docs = loader.load()

    # 3. Assert: Verify results
    assert len(docs) == 1
    assert docs[0].metadata["source"] == str(file_path)

    # Verify content
    content = docs[0].page_content
    assert "Test Conversation - user on" in content
    assert "Hello AI" in content
    assert "Test Conversation - assistant on" in content
    assert "Hello Human" in content

    # Verify it correctly skipped the first system message
    assert "System prompt" not in content


def test_chatgpt_loader_num_logs(tmp_path: Path) -> None:
    """Test checking limits on number of logs loaded."""
    file_path = tmp_path / "logs.json"
    # Create mock data with 2 conversations
    mock_data = [
        {"title": "Conv 1", "mapping": {}},
        {"title": "Conv 2", "mapping": {}},
    ]
    file_path.write_text(json.dumps(mock_data), encoding="utf-8")

    # Set num_logs=1
    loader = ChatGPTLoader(str(file_path), num_logs=1)
    docs = loader.load()

    assert len(docs) == 1
