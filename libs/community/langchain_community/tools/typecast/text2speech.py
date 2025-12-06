import tempfile
import logging
from typing import Any, Dict, Optional

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from langchain_core.utils import get_from_dict_or_env
from pydantic import model_validator

logger = logging.getLogger(__name__)


def _import_typecast() -> Any:
    try:
        import typecast
    except ImportError as e:
        raise ImportError(
            "Cannot import typecast, please install `pip install typecast-python`."
        ) from e
    return typecast


class TypecastText2SpeechTool(BaseTool):
    """Tool that queries the Typecast Text2Speech API.

    In order to set this up, follow instructions at:
    https://typecast.ai/docs/overview
    """

    model: str = "ssfm-v21"
    voice_id: str = "tc_62a8975e695ad26f7fb514d1"
    language: Optional[str] = None
    emotion_preset: str = "normal"
    emotion_intensity: float = 1.0
    audio_format: str = "wav"

    name: str = "typecast_text2speech"
    description: str = (
        "A wrapper around Typecast Text2Speech. "
        "Useful for when you need to convert text to speech. "
        "It supports 27+ languages including English, Korean, Spanish, Japanese, "
        "Chinese, and many more with emotion control capabilities. "
    )

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Any:
        """Validate that api key exists in environment."""
        _ = get_from_dict_or_env(values, "typecast_api_key", "TYPECAST_API_KEY")

        return values

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        typecast = _import_typecast()

        try:
            # Get API key from environment
            import os

            api_key = os.environ.get("TYPECAST_API_KEY")

            # Initialize client
            client = typecast.Typecast(api_key=api_key)

            # Create TTS request
            request = typecast.TTSRequest(
                text=query,
                model=self.model,
                voice_id=self.voice_id,
                language=self.language,
                prompt=typecast.Prompt(
                    emotion_preset=self.emotion_preset,
                    emotion_intensity=self.emotion_intensity,
                ),
                output=typecast.Output(audio_format=self.audio_format),
            )

            # Convert text to speech
            response = client.text_to_speech(request)

            # Save to temporary file
            suffix = f".{self.audio_format}"
            with tempfile.NamedTemporaryFile(
                mode="wb", suffix=suffix, delete=False
            ) as f:
                f.write(response.audio_data)

            return f.name
        except Exception as e:
            raise RuntimeError(f"Error while running TypecastText2SpeechTool: {e}")

    def play(self, speech_file: str) -> None:
        """Play the text as speech."""
        try:
            import sounddevice as sd
            import soundfile as sf
        except ImportError as e:
            logger.error(
                "Cannot import sounddevice or soundfile. "
                "Please install them with `pip install sounddevice soundfile`. "
                f"Error: {e}"
            )
            return

        try:
            data, samplerate = sf.read(speech_file)

            # Get the current output device
            output_device = sd.default.device[1]  # [input, output]

            # Play on the current output device
            sd.play(data, samplerate, device=output_device)
            sd.wait()
        except Exception as e:
            logger.error(f"Error while playing audio: {e}")
