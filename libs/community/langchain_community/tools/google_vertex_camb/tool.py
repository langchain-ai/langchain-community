import base64
import json
import os
import random
from typing import Any, Dict, Optional

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from langchain_core.utils import get_from_dict_or_env
from pydantic import model_validator

def _import_vertex_ai() -> Any:
    try:
        from google.cloud import aiplatform
    except ImportError as e:
        raise ImportError(
            "Cannot import Vertex AI, please install `pip install google-cloud-aiplatform`."
        ) from e
    return aiplatform

class GoogleVertexCambTool(BaseTool):
    """Tool that queries the Google Vertex AI MARS7 Text2Speech API.

    In order to set this up, follow instructions at:
    https://docs.camb.ai/introduction
    """

    name: str = "google_vertex_camb"
    description: str = (
        "A wrapper around Google Vertex AI MARS7 CambAI Text2Speech. "
        "Useful for when you need to convert text to speech with voice cloning capabilities. "
        "Supports multilingual synthesis including English, Spanish, and other languages. "
        "Requires reference audio for voice cloning."
    )
    project_id: str = ""
    location: str = ""
    endpoint_id: str = ""
    reference_audio_path: Optional[str] = None
    reference_text: Optional[str] = None
    language: str = "en-us"

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Any:
        """Validate that required environment variables exist."""
        values["project_id"] = get_from_dict_or_env(
            values, "project_id", "PROJECT_ID"
        )
        values["location"] = get_from_dict_or_env(
            values, "location", "LOCATION"
        )
        values["endpoint_id"] = get_from_dict_or_env(
            values, "endpoint_id", "ENDPOINT_ID"
        )
        values["reference_audio_path"] = get_from_dict_or_env(
            values, "reference_audio_path", "REFERENCE_AUDIO_PATH", default=None
        )
        values["reference_text"] = get_from_dict_or_env(
            values, "reference_text", "REFERENCE_TEXT", default=None
        )
        
        # Validate Google Cloud credentials
        if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
            raise ValueError(
                "GOOGLE_APPLICATION_CREDENTIALS environment variable must be set "
                "with path to service account key file."
            )
        
        return values
    
    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Convert text to speech using Vertex AI MARS7 model."""
        try:
            aiplatform = _import_vertex_ai()
            
            # Initialize Vertex AI
            aiplatform.init(project=self.project_id, location=self.location)
            
            # Prepare reference audio if provided
            audio_ref_bytes = None
            if self.reference_audio_path and os.path.exists(self.reference_audio_path):
                with open(self.reference_audio_path, "rb") as f:
                    audio_ref_bytes = base64.b64encode(f.read()).decode("utf-8")
            else:
                raise ValueError(
                    f"Reference audio file not found at: {self.reference_audio_path}. "
                    "Please provide a valid reference audio file for voice cloning."
                )
            
            # Prepare prediction instances
            instances = {
                "text": query,
                "audio_ref": audio_ref_bytes,
                "language": self.language
            }
            
            # Add reference text if provided
            if self.reference_text:
                instances["ref_text"] = self.reference_text
            
            # Get endpoint and make prediction
            endpoint = aiplatform.Endpoint(endpoint_name=self.endpoint_id)
            data = {"instances": [instances]}
            
            response = endpoint.raw_predict(
                body=json.dumps(data).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            
            # Generate output filename (MARS7 outputs FLAC format)
            output_file = f"vertex_camb_speech.flac"
            
            # Save audio file
            with open(output_file, "wb") as f:
                predictions = json.loads(response.content)["predictions"]
                if predictions and len(predictions) > 0:
                    audio_bytes = base64.b64decode(predictions[0])
                    f.write(audio_bytes)
                else:
                    raise RuntimeError("No audio predictions returned from the model")
            
            return output_file
            
        except Exception as e:
            raise RuntimeError(f"Error while running GoogleVertexCambTool: {e}")
    
    def play(self, speech_file: str) -> None:
        """Play the generated speech audio file.
        
        Args:
            speech_file: Path to the audio file to play (FLAC format).
        """
        try:
            import tempfile
            
            # Check if file exists
            if not os.path.exists(speech_file):
                raise FileNotFoundError(f"Audio file not found: {speech_file}")
            
            import soundfile as sf
            
            # Read FLAC file
            data, samplerate = sf.read(speech_file)
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                temp_wav_path = tmp_file.name
                sf.write(temp_wav_path, data, samplerate, format='WAV')
            
            # Play the converted WAV file
            import pygame
            pygame.mixer.init()
            pygame.mixer.music.load(temp_wav_path)
            pygame.mixer.music.play()
            
            # Wait for the audio to finish playing
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            
            # Clean up temporary file
            os.unlink(temp_wav_path)
                
        except ImportError:
            raise ImportError(
                "Cannot import required audio libraries. Please install: pip install pygame soundfile"
            )
        except Exception as e:
            raise RuntimeError(f"Error playing audio file: {e}")
            
    def set_reference_audio(self, audio_path: str, reference_text: Optional[str] = None) -> None:
        """Set the reference audio for voice cloning.
        
        Args:
            audio_path: Path to the reference audio file.
            reference_text: Optional transcription of the reference audio.
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Reference audio file not found: {audio_path}")
        
        self.reference_audio_path = audio_path
        if reference_text:
            self.reference_text = reference_text
            
    def set_language(self, language: str) -> None:
        """Set the target language for synthesis.
        
        Args:
            language: Language code (e.g., 'en-us', 'es-es').
        """
        self.language = language