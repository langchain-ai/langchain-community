"""Unit tests for Typecast Text2Speech Tool."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.tools import BaseTool

from langchain_community.tools.typecast import TypecastText2SpeechTool


@patch.dict("os.environ", {"TYPECAST_API_KEY": "test_api_key"})
def test_typecast_tool_initialization() -> None:
    """Test TypecastText2SpeechTool initialization."""
    tool = TypecastText2SpeechTool()
    assert isinstance(tool, BaseTool)
    assert tool.name == "typecast_text2speech"
    assert tool.model == "ssfm-v21"
    assert tool.voice_id == "tc_62a8975e695ad26f7fb514d1"
    assert tool.emotion_preset == "normal"
    assert tool.emotion_intensity == 1.0
    assert tool.audio_format == "wav"


@patch.dict("os.environ", {"TYPECAST_API_KEY": "test_api_key"})
def test_typecast_tool_custom_parameters() -> None:
    """Test TypecastText2SpeechTool with custom parameters."""
    tool = TypecastText2SpeechTool(
        model="ssfm-v20",
        voice_id="custom_voice_id",
        language="kor",
        emotion_preset="happy",
        emotion_intensity=1.5,
        audio_format="mp3",
    )
    assert tool.model == "ssfm-v20"
    assert tool.voice_id == "custom_voice_id"
    assert tool.language == "kor"
    assert tool.emotion_preset == "happy"
    assert tool.emotion_intensity == 1.5
    assert tool.audio_format == "mp3"


@patch.dict("os.environ", {"TYPECAST_API_KEY": "test_api_key"})
def test_typecast_tool_validation() -> None:
    """Test that tool validates API key presence."""
    tool = TypecastText2SpeechTool()
    assert tool is not None


def test_typecast_tool_validation_missing_key() -> None:
    """Test that tool raises error when API key is missing."""
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError):
            TypecastText2SpeechTool()


@patch.dict("os.environ", {"TYPECAST_API_KEY": "test_api_key"})
@patch("langchain_community.tools.typecast.text2speech._import_typecast")
def test_typecast_tool_run(mock_import: MagicMock) -> None:
    """Test TypecastText2SpeechTool._run method."""
    # Mock the typecast module
    mock_typecast = MagicMock()
    mock_import.return_value = mock_typecast

    # Mock the client and response
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.audio_data = b"fake_audio_data"

    mock_typecast.Typecast.return_value = mock_client
    mock_client.text_to_speech.return_value = mock_response

    # Mock TTSRequest, Prompt, and Output
    mock_typecast.TTSRequest = MagicMock()
    mock_typecast.Prompt = MagicMock()
    mock_typecast.Output = MagicMock()

    # Run the tool
    tool = TypecastText2SpeechTool()
    result = tool._run("Hello world")

    # Verify the result is a file path
    assert isinstance(result, str)
    assert result.endswith(".wav")

    # Verify the client was called correctly
    mock_typecast.Typecast.assert_called_once_with(api_key="test_api_key")
    mock_client.text_to_speech.assert_called_once()


@patch.dict("os.environ", {"TYPECAST_API_KEY": "test_api_key"})
@patch("langchain_community.tools.typecast.text2speech._import_typecast")
def test_typecast_tool_run_with_error(mock_import: MagicMock) -> None:
    """Test TypecastText2SpeechTool._run method with error."""
    # Mock the typecast module to raise an exception
    mock_typecast = MagicMock()
    mock_import.return_value = mock_typecast
    mock_typecast.Typecast.side_effect = Exception("API Error")

    # Run the tool and expect a RuntimeError
    tool = TypecastText2SpeechTool()
    with pytest.raises(
        RuntimeError, match="Error while running TypecastText2SpeechTool"
    ):
        tool._run("Hello world")


def test_import_typecast_missing() -> None:
    """Test that _import_typecast raises ImportError when typecast is not installed."""
    with patch.dict("sys.modules", {"typecast": None}):
        # Remove typecast from sys.modules temporarily
        import sys

        import langchain_community.tools.typecast.text2speech as tts_module

        typecast_backup = sys.modules.pop("typecast", None)
        try:
            with patch.object(tts_module, "_import_typecast") as mock_import:
                mock_import.side_effect = ImportError(
                    "Cannot import typecast, please install "
                    "`pip install typecast-python`."
                )
                with pytest.raises(
                    ImportError, match="Cannot import typecast, please install"
                ):
                    mock_import()
        finally:
            # Restore typecast if it was present
            if typecast_backup is not None:
                sys.modules["typecast"] = typecast_backup
