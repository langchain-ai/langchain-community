import tempfile
from enum import Enum
from typing import Any, Dict, Optional, Union

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from langchain_core.utils import get_from_dict_or_env
from pydantic import model_validator

def _import_cambai() -> Any:
    try:
        from cambai import CambAI
    except ImportError as e:
        raise ImportError(
            "Cannot import cambai, please install `pip install cambai`."
        ) from e
    return CambAI

from cambai.models.output_type import OutputType 
class CambAIText2SpeechTool(BaseTool):
    """Tool that queries the CambAI Text2Speech API.

    In order to set this up, follow instructions at:
    https://docs.camb.ai/introduction
    """

    name: str = "camb_ai_text2speech"
    description: str = (
        "A wrapper around CambAI Text2Speech. "
        "Useful for when you need to convert text to speech. "
        "It supports 140 languages, including English, German, Polish, "
        "Spanish, Italian, French, Portuguese, and Hindi. "
    )
    voice_id: int = 20303
    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Any:
        """Validate that api key exists in environment."""
        _ = get_from_dict_or_env(values, "camb_api_key", "CAMB_API_KEY")

        return values
    
    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        try:
            cambai = _import_cambai()
            client = cambai()
            file_path = "cambai_speech.wav"
            print(f"Generating speech and saving to {file_path}...")
            client.text_to_speech (
                text=query,
                voice_id=self.voice_id,
                output_type=OutputType.RAW_BYTES,
                save_to_file=file_path
            )
            print(f"Success! Audio saved to {file_path}")
            return file_path
        except Exception as e:
            raise RuntimeError(f"Error while running CambAIText2SpeechTool: {e}")
    
    def play(self, speech_file: str) -> None:
        """Play the text as speech.
        
        Args:
            speech_file: Path to the audio file to play.
        """
        try:
            import pygame
            
            pygame.mixer.init()
            pygame.mixer.music.load(speech_file)
            pygame.mixer.music.play()
            
            # Wait for the audio to finish playing
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
                
        except ImportError:
            raise ImportError(
                "Cannot import pygame, please install `pip install pygame`."
            )
        except Exception as e:
            raise RuntimeError(f"Error playing audio file: {e}")
        
