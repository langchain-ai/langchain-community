import sys
from pathlib import Path
from typing import Any, Generator
from unittest.mock import MagicMock, patch

import pytest

from langchain_community.document_loaders.email import OutlookMessageLoader


@pytest.fixture
def mock_extract_msg() -> Generator[Any, None, None]:
    """Mock the extract_msg library to avoid installing it."""
    with patch.dict(sys.modules, {"extract_msg": MagicMock()}):
        yield


def test_outlook_loader_initialization_failure(tmp_path: Path) -> None:
    """Test that loader raises error for non-existent file."""
    fake_path = tmp_path / "non_existent.msg"
    with pytest.raises(ValueError, match="not a valid file"):
        OutlookMessageLoader(str(fake_path))


def test_outlook_loader_success(tmp_path: Path, mock_extract_msg: Any) -> None:
    """Test successful loading of an Outlook .msg file."""
    # 1. Create a dummy file so os.path.isfile passes
    fake_msg_file = tmp_path / "test.msg"
    fake_msg_file.write_text("fake content", encoding="utf-8")

    # 2. Mock the extract_msg.Message class
    with patch("extract_msg.Message") as MockMessage:
        # Setup the mock message object that extract_msg returns
        mock_msg_instance = MockMessage.return_value
        mock_msg_instance.body = "This is the email body."
        mock_msg_instance.subject = "Test Subject"
        mock_msg_instance.sender = "sender@example.com"
        mock_msg_instance.date = "2023-01-01"

        # 3. Run the Loader
        loader = OutlookMessageLoader(str(fake_msg_file))
        docs = loader.load()

        # 4. Assertions
        assert len(docs) == 1
        assert docs[0].page_content == "This is the email body."
        assert docs[0].metadata["subject"] == "Test Subject"
        assert docs[0].metadata["sender"] == "sender@example.com"
        assert docs[0].metadata["source"] == str(fake_msg_file)

        # Verify close was called to ensure resource cleanup
        mock_msg_instance.close.assert_called_once()
