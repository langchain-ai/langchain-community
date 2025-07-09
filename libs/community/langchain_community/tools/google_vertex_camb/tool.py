"""Google Vertex AI MARS7 Text-to-Speech Tool.

This module provides a LangChain tool for converting text to speech using
Google Cloud's Vertex AI platform with the MARS7 model from CambAI.

The tool supports:
- Voice cloning with reference audio
- Multilingual synthesis (10+ languages)
- High-quality speech synthesis
- Integration with LangChain agents and chains

Dependencies:
    - google-cloud-aiplatform: For Vertex AI API access
    - soundfile: For audio processing (optional)
    - pygame: For audio playback (optional)

Environment Variables:
    - PROJECT_ID: Google Cloud project ID
    - LOCATION: Google Cloud region (e.g., us-central1)
    - ENDPOINT_ID: Vertex AI endpoint ID for MARS7 model
    - REFERENCE_AUDIO_PATH: Path to reference audio file
    - REFERENCE_TEXT: Optional reference text for voice cloning
    - GOOGLE_APPLICATION_CREDENTIALS: Path to service account key file

Example:
    Basic usage:
        >>> from langchain_community.tools import GoogleVertexCambTool
        >>> tool = GoogleVertexCambTool()
        >>> audio_file = tool.invoke("Hello, this is a test message.")
        >>> # Returns path to generated FLAC audio file
"""

import base64
import json
import os
import uuid
from typing import Any, Dict, Literal, Optional

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from langchain_core.utils import get_from_dict_or_env
from pydantic import model_validator

Mars7Language = Literal[
    "de-de",
    "en-gb",
    "en-us",
    "es-us",
    "es-es",
    "fr-ca",
    "fr-fr",
    "ja-jp",
    "ko-kr",
    "zh-cn",
]
"""Supported language codes for MARS7 text-to-speech synthesis.

Each language code follows the format: language-country (e.g., en-us for US English).
"""


def _import_vertex_ai() -> Any:
    """Import Google Cloud AI Platform library with proper error handling.

    Returns:
        The google.cloud.aiplatform module.

    Raises:
        ImportError: If google-cloud-aiplatform package is not installed.
    """
    try:
        from google.cloud import aiplatform
    except ImportError as e:
        raise ImportError(
            "Cannot import Vertex AI, "
            "please install `pip install google-cloud-aiplatform`."
        ) from e
    return aiplatform


class GoogleVertexCambTool(BaseTool):
    """Tool that queries the Google Vertex AI MARS7 Text2Speech API.

    This tool provides text-to-speech conversion with voice cloning capabilities
    using Google Cloud's Vertex AI platform and the MARS7 model from CambAI.
    It supports multilingual synthesis and requires a reference audio file for
    voice cloning.

    Setup Instructions:
        1. Install required dependencies:
           pip install google-cloud-aiplatform soundfile pygame

        2. Set up Google Cloud credentials:
           export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json

        3. Configure environment variables:
           export PROJECT_ID=your-gcp-project-id
           export LOCATION=us-central1
           export ENDPOINT_ID=your-vertex-ai-endpoint-id
           export REFERENCE_AUDIO_PATH=/path/to/reference-audio.wav
           export REFERENCE_TEXT="Optional reference text"

        4. Deploy MARS7 model to Vertex AI endpoint (see https://docs.camb.ai/introduction)

    Attributes:
        name: The tool name identifier ("google_vertex_camb")
        description: Human-readable description of the tool's capabilities
        project_id: Google Cloud project ID where the endpoint is deployed
        location: Google Cloud region where the endpoint is located
        endpoint_id: Vertex AI endpoint ID for the MARS7 model
        reference_audio_path: Path to reference audio file for voice cloning
        reference_text: Optional reference text that matches the reference audio
        language: Target language for synthesis (default: "en-us")

    Supported Languages:
        - de-de: German (Germany)
        - en-gb: English (UK)
        - en-us: English (US)
        - es-us: Spanish (US)
        - es-es: Spanish (Spain)
        - fr-ca: French (Canada)
        - fr-fr: French (France)
        - ja-jp: Japanese (Japan)
        - ko-kr: Korean (Korea)
        - zh-cn: Chinese (China)

    Example:
        Basic usage:
            >>> from langchain_community.tools import GoogleVertexCambTool
            >>> tool = GoogleVertexCambTool(
            ...     project_id="my-project",
            ...     location="us-central1",
            ...     endpoint_id="my-endpoint",
            ...     reference_audio_path="/path/to/voice.wav"
            ... )
            >>> audio_file = tool.invoke("Hello, this is a test message.")
            >>> # Returns: "vertex_camb_speech_<uuid>.flac"

        Using environment variables:
            >>> import os
            >>> os.environ["PROJECT_ID"] = "my-project"
            >>> os.environ["LOCATION"] = "us-central1"
            >>> os.environ["ENDPOINT_ID"] = "my-endpoint"
            >>> os.environ["REFERENCE_AUDIO_PATH"] = "/path/to/voice.wav"
            >>> os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/path/to/key.json"
            >>> tool = GoogleVertexCambTool()
            >>> audio_file = tool.invoke("Hello world!")

        With custom language:
            >>> tool = GoogleVertexCambTool(
            ...     project_id="my-project",
            ...     location="us-central1",
            ...     endpoint_id="my-endpoint",
            ...     reference_audio_path="/path/to/voice.wav",
            ...     language="es-us"
            ... )
            >>> audio_file = tool.invoke("Hola mundo!")

    Note:
        - The tool generates unique filenames using UUID to prevent conflicts
        - Output format is FLAC as required by the MARS7 model
        - Reference audio file is required for voice cloning functionality
        - Google Cloud credentials must be properly configured
        - The Vertex AI endpoint must be deployed and accessible

    Raises:
        ValueError: If required environment variables are missing or invalid
        ImportError: If required dependencies are not installed
        RuntimeError: If the Vertex AI API call fails or returns no predictions
        FileNotFoundError: If the reference audio file doesn't exist
    """

    name: str = "google_vertex_camb"
    description: str = (
        "A wrapper around Google Vertex AI MARS7 CambAI Text2Speech. "
        "Useful for when you need to convert text to speech with voice cloning "
        "capabilities. Supports multilingual synthesis including English, Spanish, "
        "and other languages. Requires reference audio for voice cloning."
    )
    project_id: str
    location: str
    endpoint_id: str
    reference_audio_path: str
    reference_text: Optional[str] = None
    language: Mars7Language = "en-us"

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Any:
        """Validate that required environment variables exist and are valid.

        This method is called during tool initialization to ensure all required
        configuration is present. It retrieves values from environment variables
        if not provided directly.

        Args:
            values: Dictionary of configuration values

        Returns:
            Updated dictionary with validated configuration values

        Raises:
            ValueError: If required environment variables are missing or invalid
        """
        values["project_id"] = get_from_dict_or_env(values, "project_id", "PROJECT_ID")
        values["location"] = get_from_dict_or_env(values, "location", "LOCATION")
        values["endpoint_id"] = get_from_dict_or_env(
            values, "endpoint_id", "ENDPOINT_ID"
        )
        values["reference_audio_path"] = get_from_dict_or_env(
            values, "reference_audio_path", "REFERENCE_AUDIO_PATH"
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
        """Convert text to speech using Vertex AI MARS7 model.

        This method performs the core text-to-speech conversion by:
        1. Initializing the Vertex AI client
        2. Loading and encoding the reference audio file
        3. Making a prediction request to the MARS7 endpoint
        4. Saving the generated audio to a unique FLAC file

        Args:
            query: The text to convert to speech
            run_manager: Optional callback manager for monitoring the run (unused)

        Returns:
            Path to the generated audio file (FLAC format)

        Raises:
            ValueError: If the reference audio file is not found
            RuntimeError: If the Vertex AI API call fails or returns no predictions
            ImportError: If required dependencies are not installed
        """
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
                "language": self.language,
            }

            # Add reference text if provided
            if self.reference_text:
                instances["ref_text"] = self.reference_text

            # Get endpoint and make prediction
            endpoint = aiplatform.Endpoint(endpoint_name=self.endpoint_id)
            data = {"instances": [instances]}

            response = endpoint.raw_predict(
                body=json.dumps(data).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )

            # Generate output filename (MARS7 outputs FLAC format)
            output_file = f"vertex_camb_speech_{uuid.uuid4()}.flac"

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
