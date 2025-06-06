"""Test CambAI Text2Speech tool."""

import pytest
from unittest.mock import MagicMock, patch

from langchain_community.tools.cambai.text2speech import CambAIText2SpeechTool


def test_camb_ai_text2speech_tool_initialization() -> None:
    """Test initialization of the CambAIText2SpeechTool."""
    with patch.dict("os.environ", {"CAMB_API_KEY": "test-api-key"}):
        tool = CambAIText2SpeechTool()
        assert tool.name == "camb_ai_text2speech"
        assert "CambAI Text2Speech" in tool.description
        assert tool.voice_id == 20303


@patch("langchain_community.tools.cambai.text2speech._import_cambai")
def test_camb_ai_text2speech_tool_run(mock_import_cambai: MagicMock) -> None:
    """Test running the CambAIText2SpeechTool."""
    # Setup mock
    mock_client = MagicMock()
    mock_import_cambai.return_value = MagicMock(return_value=mock_client)
    
    with patch.dict("os.environ", {"CAMB_API_KEY": "test-api-key"}):
        tool = CambAIText2SpeechTool()
        result = tool._run("Hello world")
        
        # Verify the client was called correctly
        mock_client.text_to_speech.assert_called_once()
        assert "cambai_speech.wav" in result


@patch("langchain_community.tools.cambai.text2speech._import_cambai")
@patch("pygame.mixer")
def test_camb_ai_text2speech_tool_play(
    mock_pygame_mixer: MagicMock, mock_import_cambai: MagicMock
) -> None:
    """Test playing audio with the CambAIText2SpeechTool."""
    # Setup mocks
    mock_client = MagicMock()
    mock_import_cambai.return_value = MagicMock(return_value=mock_client)
    
    with patch.dict("os.environ", {"CAMB_API_KEY": "test-api-key"}):
        tool = CambAIText2SpeechTool()
        
        # Test play method
        with patch("pygame.time.Clock") as mock_clock:
            mock_pygame_mixer.music.get_busy.side_effect = [True, False]  # Play once then stop
            tool.play("Hello world")
            
            # Verify pygame was used correctly
            mock_pygame_mixer.init.assert_called_once()
            mock_pygame_mixer.music.load.assert_called_once()
            mock_pygame_mixer.music.play.assert_called_once()