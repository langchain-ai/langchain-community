"""Tests for ElevenLabsText2SpeechTool."""

import os
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _set_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test-key")


@patch.dict("sys.modules", {"elevenlabs": MagicMock()})
def test_run_creates_client_and_converts() -> None:
    """Test that _run uses elevenlabs.ElevenLabs() (not elevenlabs.client.ElevenLabs)."""
    import sys

    mock_elevenlabs = sys.modules["elevenlabs"]
    mock_client = MagicMock()
    mock_client.text_to_speech.convert.return_value = b"fake-audio"
    mock_elevenlabs.ElevenLabs.return_value = mock_client

    from langchain_community.tools.eleven_labs.text2speech import (
        ElevenLabsText2SpeechTool,
    )

    tool = ElevenLabsText2SpeechTool()
    result = tool._run("Hello world")

    mock_elevenlabs.ElevenLabs.assert_called_once()
    mock_client.text_to_speech.convert.assert_called_once_with(
        text="Hello world",
        model_id=tool.model,
        voice_id=tool.voice,
        output_format="mp3_44100_128",
    )
    assert os.path.exists(result)
    with open(result, "rb") as f:
        assert f.read() == b"fake-audio"
    os.remove(result)


@patch.dict("sys.modules", {"elevenlabs": MagicMock()})
def test_stream_speech_creates_client() -> None:
    """Test that stream_speech uses elevenlabs.ElevenLabs()."""
    import sys

    mock_elevenlabs = sys.modules["elevenlabs"]
    mock_client = MagicMock()
    mock_elevenlabs.ElevenLabs.return_value = mock_client

    from langchain_community.tools.eleven_labs.text2speech import (
        ElevenLabsText2SpeechTool,
    )

    tool = ElevenLabsText2SpeechTool()
    tool.stream_speech("Hello world")

    mock_elevenlabs.ElevenLabs.assert_called()
    mock_client.text_to_speech.stream.assert_called_once_with(
        text="Hello world", model_id=tool.model, voice_id=tool.voice
    )
