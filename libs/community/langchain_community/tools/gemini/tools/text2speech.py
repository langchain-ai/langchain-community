"""
Implementation of a Text-to-Speech tool using Google Gemini.

This module provides a LangChain-compatible tool for text-to-speech
conversion using Google Gemini's TTS models. The implementation includes
full support for all 30 available voices, organized by categories to
facilitate selection based on the context of use.

The voices follow a thematic convention based on mythology and astronomy,
offering a rich variety of tones and vocal characteristics.
"""

import logging
import tempfile
import wave
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from langchain_core.utils import get_from_dict_or_env
from pydantic import model_validator

from .players import FFmpegAudioPlayer

# Configure logging to provide detailed operational information
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _import_google_genai() -> Any:
    """
    Imports the google.genai library with appropriate error handling.

    Returns:
        The google.genai module if imported successfully.

    Raises:
        ImportError: If the library is not installed.
    """
    try:
        from google import genai
    except ImportError as e:
        raise ImportError("Cannot import google.genai, please install `pip install google-genai`.") from e
    return genai


class GeminiTTSModel(str, Enum):
    """
    Available models for Google Gemini Text-to-Speech.

    Attributes:
        G25_FLASH_TTS: Gemini 2.5 Flash model optimized for TTS (faster, efficient).
        G25_PRO_TTS: Gemini 2.5 Pro model with superior TTS quality (slower, higher quality).
    """

    G25_FLASH_TTS = "gemini-2.5-flash-preview-tts"
    G25_PRO_TTS = "gemini-2.5-pro-preview-tts"


class GeminiVoice(str, Enum):
    """
    Available voices for Google Gemini TTS.

    Google Gemini offers 30 distinct voices, all named following
    mythological and astronomical conventions. This rich variety allows
    for precise selection based on the context of use, target audience, and
    aesthetic preferences.

    Voice Categories:

    **Greek and Roman Mythology** (voices with classic personalities):
    - ZEPHYR: West wind in Greek mythology (soft, ethereal).
    - CHARON: Ferryman of Hades (deep, mysterious).
    - KORE: Another name for Persephone (feminine, elegant).
    - AOEDE: One of the muses of song (musical, harmonious).
    - CALLIRRHOE: Nymph with a beautiful voice (melodious, fluid).
    - AUTONOE: Mother of Actaeon (maternal, caring).
    - LEDA: Mother of Helen of Troy (noble, distinguished).

    **Norse Mythology** (voice with characteristic strength):
    - FENRIR: Giant wolf (powerful, imposing).

    **Astronomy - Moons and Satellites** (voices with unique characteristics):
    - ENCELADUS: Moon of Saturn (bright, clear).
    - IAPETUS: Moon of Saturn (contrasting, varied).
    - UMBRIEL: Moon of Uranus (dark, deep).
    - DESPINA: Moon of Neptune (delicate, subtle).
    - ERINOME: Moon of Jupiter (small, precise).
    - LAOMEDEIA: Moon of Neptune (distant, ethereal).

    **Astronomy - Stars** (bright and distinct voices):
    - ALGIEBA: Double star in Leo (dual personality).
    - ALGENIB: Star in Pegasus (lofty, inspiring).
    - RASALGETHI: Star in Hercules (heroic, strong).
    - ACHERNAR: Brightest star in Eridanus (intense, striking).
    - ALNILAM: Star in Orion's Belt (centered, balanced).
    - SCHEDAR: Star in Cassiopeia (regal, majestic).
    - GACRUX: Star in the Southern Cross (guiding, clear).
    - PULCHERRIMA: "The most beautiful" (aesthetic, refined).
    - ACHIRD: Star in Cassiopeia (subtle, sophisticated).
    - ZUBENELGENUBI: Southern claw of Libra (balanced, fair).
    - VINDEMIATRIX: "The grape-harvester" (hardworking, earthy).
    - SADACHBIA: "Lucky star of the king" (fortunate, positive).
    - SADALTAGER: "The merchant's star" (communicative, expressive).
    - SULAFAT: Star in Lyra (musical, harmonic).

    **Other Origins** (voices with special characteristics):
    - PUCK: Shakespearean character (playful, mischievous).
    - ORUS: Derivation of Horus (wise, ancient).
    """

    # Greek and Roman Mythology - Voices with classic personalities
    ZEPHYR = "Zephyr"  # West wind - soft, ethereal
    CHARON = "Charon"  # Ferryman of Hades - deep, mysterious
    KORE = "Kore"  # Persephone - feminine, elegant
    AOEDE = "Aoede"  # Muse of song - musical, harmonious
    CALLIRRHOE = "Callirrhoe"  # Nymph with a beautiful voice - melodious, fluid
    AUTONOE = "Autonoe"  # Mother of Actaeon - maternal, caring
    LEDA = "Leda"  # Mother of Helen - noble, distinguished

    # Norse Mythology - Voice with characteristic strength
    FENRIR = "Fenrir"  # Giant wolf - powerful, imposing

    # Astronomy - Moons and Satellites
    ENCELADUS = "Enceladus"  # Moon of Saturn - bright, clear
    IAPETUS = "Iapetus"  # Moon of Saturn - contrasting, varied
    UMBRIEL = "Umbriel"  # Moon of Uranus - dark, deep
    DESPINA = "Despina"  # Moon of Neptune - delicate, subtle
    ERINOME = "Erinome"  # Moon of Jupiter - small, precise
    LAOMEDEIA = "Laomedeia"  # Moon of Neptune - distant, ethereal

    # Astronomy - Stars
    ALGIEBA = "Algieba"  # Double star - dual personality
    ALGENIB = "Algenib"  # Star in Pegasus - lofty, inspiring
    RASALGETHI = "Rasalgethi"  # Star in Hercules - heroic, strong
    ACHERNAR = "Achernar"  # Brightest in Eridanus - intense
    ALNILAM = "Alnilam"  # Orion's Belt - balanced
    SCHEDAR = "Schedar"  # Star in Cassiopeia - regal, majestic
    GACRUX = "Gacrux"  # Southern Cross - guiding, clear
    PULCHERRIMA = "Pulcherrima"  # "The most beautiful" - aesthetic, refined
    ACHIRD = "Achird"  # Star in Cassiopeia - subtle, sophisticated
    ZUBENELGENUBI = "Zubenelgenubi"  # Southern claw of Libra - balanced
    VINDEMIATRIX = "Vindemiatrix"  # "The grape-harvester" - hardworking, earthy
    SADACHBIA = "Sadachbia"  # "Lucky star" - fortunate, positive
    SADALTAGER = "Sadaltager"  # "Merchant star" - communicative
    SULAFAT = "Sulafat"  # Star in Lyra - musical, harmonic

    # Other Origins
    PUCK = "Puck"  # Shakespearean - playful, mischievous
    ORUS = "Orus"  # Derivation of Horus - wise, ancient


class VoiceRecommendationEngine:
    """
    Intelligent recommendation system for voice selection.

    This class assists in the appropriate selection of voices based on the
    context of use, content type, and target audience, turning the choice
    among 30 options into an informed and strategic decision.
    """

    @staticmethod
    def get_recommended_voices(context: str) -> List[GeminiVoice]:
        """
        Recommends voices based on the context of use.

        Args:
            context: Context of use ('formal', 'casual', 'educational',
                    'storytelling', 'commercial', 'technical').

        Returns:
            A list of recommended voices for the specified context.
        """
        recommendations = {
            "formal": [
                GeminiVoice.SCHEDAR,  # Regal, majestic
                GeminiVoice.LEDA,  # Noble, distinguished
                GeminiVoice.ORUS,  # Wise, ancient
                GeminiVoice.ZUBENELGENUBI,  # Balanced, fair
            ],
            "casual": [
                GeminiVoice.PUCK,  # Playful, mischievous
                GeminiVoice.ZEPHYR,  # Soft, ethereal
                GeminiVoice.KORE,  # Feminine, elegant
                GeminiVoice.DESPINA,  # Delicate, subtle
            ],
            "educational": [
                GeminiVoice.ORUS,  # Wise, ancient
                GeminiVoice.GACRUX,  # Guiding, clear
                GeminiVoice.ALNILAM,  # Balanced
                GeminiVoice.ENCELADUS,  # Bright, clear
            ],
            "storytelling": [
                GeminiVoice.CHARON,  # Deep, mysterious
                GeminiVoice.AOEDE,  # Musical, harmonious
                GeminiVoice.CALLIRRHOE,  # Melodious, fluid
                GeminiVoice.FENRIR,  # Powerful, imposing
            ],
            "commercial": [
                GeminiVoice.SADALTAGER,  # Communicative, expressive
                GeminiVoice.PULCHERRIMA,  # Aesthetic, refined
                GeminiVoice.SADACHBIA,  # Fortunate, positive
                GeminiVoice.ALGENIB,  # Lofty, inspiring
            ],
            "technical": [
                GeminiVoice.ERINOME,  # Small, precise
                GeminiVoice.ACHIRD,  # Subtle, sophisticated
                GeminiVoice.IAPETUS,  # Contrasting, varied
                GeminiVoice.VINDEMIATRIX,  # Hardworking, earthy
            ],
        }

        return recommendations.get(
            context.lower(),
            [
                GeminiVoice.KORE,  # Default elegant
                GeminiVoice.ZEPHYR,  # Default soft
                GeminiVoice.ENCELADUS,  # Default clear
            ],
        )

    @staticmethod
    def get_voice_characteristics(voice: GeminiVoice) -> Dict[str, str]:
        """
        Returns detailed characteristics of a specific voice.

        Args:
            voice: The voice for which to get characteristics.

        Returns:
            A dictionary with the voice's characteristics.
        """
        characteristics = {
            # Greek and Roman Mythology
            GeminiVoice.ZEPHYR: {
                "origin": "Greek Mythology",
                "personality": "Soft, ethereal",
                "best_for": "Relaxing content, meditation, nature narration",
                "tone": "Calm and gentle",
            },
            GeminiVoice.CHARON: {
                "origin": "Greek Mythology",
                "personality": "Deep, mysterious",
                "best_for": "Dramatic narratives, suspense, historical content",
                "tone": "Grave and engaging",
            },
            GeminiVoice.KORE: {
                "origin": "Greek Mythology",
                "personality": "Feminine, elegant",
                "best_for": "Presentations, education, general communication",
                "tone": "Balanced and refined",
            },
            GeminiVoice.AOEDE: {
                "origin": "Greek Mythology",
                "personality": "Musical, harmonious",
                "best_for": "Audiobooks, artistic content, singing applications",
                "tone": "Melodic and expressive",
            },
            GeminiVoice.CALLIRRHOE: {
                "origin": "Greek Mythology",
                "personality": "Melodious, fluid",
                "best_for": "Lyrical prose, storytelling, flowing narration",
                "tone": "Smooth and pleasant",
            },
            GeminiVoice.AUTONOE: {
                "origin": "Greek Mythology",
                "personality": "Maternal, caring",
                "best_for": "E-learning for children, comforting messages, audio guides",
                "tone": "Warm and reassuring",
            },
            GeminiVoice.LEDA: {
                "origin": "Greek Mythology",
                "personality": "Noble, distinguished",
                "best_for": "Formal announcements, corporate videos, epic tales",
                "tone": "Dignified and clear",
            },
            # Norse Mythology
            GeminiVoice.FENRIR: {
                "origin": "Norse Mythology",
                "personality": "Powerful, imposing",
                "best_for": "Movie trailers, gaming, dramatic and impactful content",
                "tone": "Strong and resonant",
            },
            # Astronomy - Moons and Satellites
            GeminiVoice.ENCELADUS: {
                "origin": "Astronomy - Moon of Saturn",
                "personality": "Bright, clear",
                "best_for": "News reading, tutorials, clear instructions",
                "tone": "Crisp and articulate",
            },
            GeminiVoice.IAPETUS: {
                "origin": "Astronomy - Moon of Saturn",
                "personality": "Contrasting, varied",
                "best_for": "Character voice-overs, dynamic presentations, advertisements",
                "tone": "Versatile and dynamic",
            },
            GeminiVoice.UMBRIEL: {
                "origin": "Astronomy - Moon of Uranus",
                "personality": "Dark, deep",
                "best_for": "Mystery, thrillers, noir-style narration",
                "tone": "Low-pitched and somber",
            },
            GeminiVoice.DESPINA: {
                "origin": "Astronomy - Moon of Neptune",
                "personality": "Delicate, subtle",
                "best_for": "Poetry, intimate narration, soft-spoken content",
                "tone": "Gentle and soft",
            },
            GeminiVoice.ERINOME: {
                "origin": "Astronomy - Moon of Jupiter",
                "personality": "Small, precise",
                "best_for": "Technical instructions, data reporting, language learning",
                "tone": "Exact and controlled",
            },
            GeminiVoice.LAOMEDEIA: {
                "origin": "Astronomy - Moon of Neptune",
                "personality": "Distant, ethereal",
                "best_for": "Ambient narration, science fiction, dream sequences",
                "tone": "Airy and remote",
            },
            # Astronomy - Stars
            GeminiVoice.ALGIEBA: {
                "origin": "Astronomy - Star in Leo",
                "personality": "Dual personality",
                "best_for": "Dialogues, character acting, complex narratives",
                "tone": "Adaptable and multifaceted",
            },
            GeminiVoice.ALGENIB: {
                "origin": "Astronomy - Star in Pegasus",
                "personality": "Lofty, inspiring",
                "best_for": "Motivational speeches, award ceremonies, grand announcements",
                "tone": "Elevated and powerful",
            },
            GeminiVoice.RASALGETHI: {
                "origin": "Astronomy - Star in Hercules",
                "personality": "Heroic, strong",
                "best_for": "Action stories, adventure narration, fitness applications",
                "tone": "Bold and confident",
            },
            GeminiVoice.ACHERNAR: {
                "origin": "Astronomy - Star in Eridanus",
                "personality": "Intense, striking",
                "best_for": "High-energy commercials, impactful marketing",
                "tone": "Sharp and attention-grabbing",
            },
            GeminiVoice.ALNILAM: {
                "origin": "Astronomy - Star in Orion",
                "personality": "Centered, balanced",
                "best_for": "Corporate narration, documentaries, neutral announcements",
                "tone": "Steady and reliable",
            },
            GeminiVoice.SCHEDAR: {
                "origin": "Astronomy - Star in Cassiopeia",
                "personality": "Regal, majestic",
                "best_for": "Luxury brand advertising, historical content, formal proclamations",
                "tone": "Authoritative and grand",
            },
            GeminiVoice.GACRUX: {
                "origin": "Astronomy - Star in Southern Cross",
                "personality": "Guiding, clear",
                "best_for": "GPS navigation, tutorials, instructional videos",
                "tone": "Direct and helpful",
            },
            GeminiVoice.PULCHERRIMA: {
                "origin": "Astronomy - Star System",
                "personality": "Aesthetic, refined",
                "best_for": "Art gallery guides, high-fashion content, elegant product descriptions",
                "tone": "Beautiful and sophisticated",
            },
            GeminiVoice.ACHIRD: {
                "origin": "Astronomy - Star in Cassiopeia",
                "personality": "Subtle, sophisticated",
                "best_for": "Scientific explanations, financial reports, intellectual content",
                "tone": "Understated and intelligent",
            },
            GeminiVoice.ZUBENELGENUBI: {
                "origin": "Astronomy - Star in Libra",
                "personality": "Balanced, fair",
                "best_for": "Legal disclaimers, news reporting, unbiased commentary",
                "tone": "Even and impartial",
            },
            GeminiVoice.VINDEMIATRIX: {
                "origin": "Astronomy - Star in Virgo",
                "personality": "Hardworking, earthy",
                "best_for": "DIY tutorials, documentaries about labor, down-to-earth content",
                "tone": "Grounded and practical",
            },
            GeminiVoice.SADACHBIA: {
                "origin": "Astronomy - Star in Aquarius",
                "personality": "Fortunate, positive",
                "best_for": "Uplifting messages, family-oriented commercials, positive affirmations",
                "tone": "Optimistic and bright",
            },
            GeminiVoice.SADALTAGER: {
                "origin": "Astronomy - Star",
                "personality": "Communicative, expressive",
                "best_for": "Podcasts, radio commercials, social media content",
                "tone": "Engaging and articulate",
            },
            GeminiVoice.SULAFAT: {
                "origin": "Astronomy - Star in Lyra",
                "personality": "Musical, harmonic",
                "best_for": "Jingles, creative content intros, melodic storytelling",
                "tone": "Harmonious and lyrical",
            },
            # Other Origins
            GeminiVoice.PUCK: {
                "origin": "Shakespearean Character",
                "personality": "Playful, mischievous",
                "best_for": "Children's stories, comedy, fun and engaging content",
                "tone": "Lively and animated",
            },
            GeminiVoice.ORUS: {
                "origin": "Derivation of Horus (Egyptian Mythology)",
                "personality": "Wise, ancient",
                "best_for": "Historical documentaries, philosophical texts, epic narration",
                "tone": "Knowledgeable and profound",
            },
        }

        return characteristics.get(
            voice,
            {
                "origin": "Unknown",
                "personality": "Uncatalogued characteristics",
                "best_for": "General use",
                "tone": "Neutral",
            },
        )


class GeminiText2SpeechTool(BaseTool):
    """
    Tool that uses the Google Gemini Text-to-Speech API.

    This implementation offers full access to the 30 available voices in
    Gemini TTS, organized and categorized to facilitate appropriate selection
    based on the context of use. It includes an intelligent recommendation
    system and rich metadata for each audio generation.

    The tool is fully compatible with LangChain and follows the same
    standards established by other TTS tools in the ecosystem,
    ensuring seamless integration into existing projects.

    To configure, obtain an API key from Google AI Studio:
    https://makersuite.google.com/app/apikey

    Attributes:
        model: The TTS model to be used (default: FLASH_TTS).
        voice: The voice for synthesis (default: KORE - elegant and balanced).
        sample_rate: The audio sample rate (default: 24000 Hz).
        name: The name of the tool for identification in LangChain.
        description: A description of the tool's capabilities.
    """

    model: Union[GeminiTTSModel, str] = GeminiTTSModel.G25_FLASH_TTS
    voice: Union[GeminiVoice, str] = GeminiVoice.KORE  # Default elegant voice
    sample_rate: int = 24000

    name: str = "gemini_text2speech"
    description: str = (
        "A wrapper around Google Gemini Text2Speech API with 30 unique voices. "
        "Useful for converting text to speech with exceptional quality and variety. "
        "Supports multiple languages, natural-sounding voices with distinct personalities, "
        "and intelligent voice recommendation based on content context. "
        "Powered by Google's state-of-the-art AI models with mythological and "
        "astronomical voice naming for easy identification and selection. "
        "Returns the path to the generated high-quality audio file."
    )

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Any:
        """
        Validates if the API key exists in the environment.

        This method ensures that the tool is only initialized with
        valid credentials, preventing failures during execution.

        Args:
            values: A dictionary with configuration values.

        Returns:
            The validated values.

        Raises:
            ValueError: If the API key is not found in the environment.
        """
        logger.info("Validating environment for Google API key.")
        get_from_dict_or_env(values, "google_api_key", "GOOGLE_API_KEY")
        logger.info("Google API key found in environment.")
        return values

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """
        Executes the text-to-speech conversion.

        This method orchestrates the entire audio generation process:
        connects to Gemini, sets quality parameters,
        processes the response, and saves the file with rich metadata.

        Args:
            query: The text to be converted to speech.
            run_manager: The LangChain callback manager (optional).

        Returns:
            The path to the generated audio file.

        Raises:
            RuntimeError: If an error occurs during generation or saving.
            ValueError: If the input text is invalid.
        """
        if not query or not query.strip():
            logger.error("The text for conversion cannot be empty.")
            raise ValueError("The text for conversion cannot be empty")

        genai = _import_google_genai()
        logger.info(f"Starting TTS generation for query: '{query[:50]}...'")
        logger.info(f"Using model: {self.model}, voice: {self.voice}")

        try:
            # Securely get credentials from the environment
            api_key = get_from_dict_or_env({}, "google_api_key", "GOOGLE_API_KEY")

            # Initialize client with robust error handling
            client = genai.Client(api_key=api_key)

            # Import necessary types for advanced configuration
            from google.genai import types

            # Configure TTS request with optimized parameters
            response = client.models.generate_content(
                model=self.model,
                contents=query.strip(),  # Remove unnecessary spaces
                config=types.GenerateContentConfig(
                    responseModalities=["audio"],
                    speechConfig=types.SpeechConfig(
                        voiceConfig=types.VoiceConfig(
                            prebuiltVoiceConfig=types.PrebuiltVoiceConfig(
                                voiceName=self.voice,
                            )
                        )
                    ),
                ),
            )

            # Validate response before processing
            if not response.candidates:
                raise RuntimeError("No audio response was generated")

            if not response.candidates[0].content.parts:
                raise RuntimeError("Response does not contain audio data")

            # Extract audio data with validation
            audio_data = response.candidates[0].content.parts[0].inline_data.data

            if not audio_data:
                raise RuntimeError("Audio data is empty")

            logger.info("Successfully received audio data from Gemini API.")

            # Create a temporary file with a descriptive name
            with tempfile.NamedTemporaryFile(
                mode="wb", suffix=".wav", prefix=f"gemini_tts_{self.voice.lower()}_", delete=False
            ) as temp_file:
                self._save_as_wav(temp_file.name, audio_data)
                logger.info(f"Audio successfully saved to temporary file: {temp_file.name}")
                return temp_file.name

        except Exception as e:
            error_msg = f"Error in GeminiText2SpeechTool: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

    def _save_as_wav(self, filename: str, pcm_data: bytes) -> None:
        """
        Saves PCM data as a WAV file with optimized settings.

        This method creates WAV files with consistent quality,
        using parameters optimized for maximum compatibility
        with different systems and applications.

        Args:
            filename: The path of the file to be saved.
            pcm_data: Audio data in raw PCM format.

        Raises:
            IOError: If it fails to save the file.
            ValueError: If the PCM data is invalid.
        """
        if not pcm_data:
            logger.error("PCM data for WAV saving cannot be empty.")
            raise ValueError("PCM data cannot be empty")

        try:
            logger.debug(f"Saving PCM data to WAV file: {filename}")
            with wave.open(filename, "wb") as wav_file:
                # Optimized settings for quality and compatibility
                wav_file.setnchannels(1)  # Mono for TTS
                wav_file.setsampwidth(2)  # 16-bit for quality
                wav_file.setframerate(self.sample_rate)  # Configurable rate
                wav_file.writeframes(pcm_data)
            logger.debug(f"Successfully saved WAV file: {filename}")

        except Exception as e:
            error_msg = f"Failed to save WAV file: {filename}. Reason: {e}"
            logger.error(error_msg, exc_info=True)
            raise IOError(f"Failed to save WAV file: {e}") from e

    def get_voice_recommendation(self, context: str) -> List[GeminiVoice]:
        """
        Recommends appropriate voices for a specific context.

        This method uses the intelligent recommendation system to
        suggest the most suitable voices based on the content type
        and intended use.

        Args:
            context: The context of use ('formal', 'casual', 'educational', etc.).

        Returns:
            A list of recommended voices for the context.

        Example:
            ```python
            tts = GeminiText2SpeechTool()
            formal_voices = tts.get_voice_recommendation('formal')
            # Returns: [SCHEDAR, LEDA, ORUS, ZUBENELGENUBI]
            ```
        """
        logger.info(f"Getting voice recommendations for context: '{context}'")
        recommendations = VoiceRecommendationEngine.get_recommended_voices(context)
        logger.info(f"Recommended voices: {[voice.value for voice in recommendations]}")
        return recommendations

    def get_voice_info(self, voice: Optional[GeminiVoice] = None) -> Dict[str, str]:
        """
        Gets detailed information about a specific voice.

        Args:
            voice: The voice to query (uses the current voice if not specified).

        Returns:
            A dictionary with the voice's characteristics.
        """
        target_voice = voice or GeminiVoice(self.voice)
        logger.info(f"Getting information for voice: '{target_voice.value}'")
        return VoiceRecommendationEngine.get_voice_characteristics(target_voice)

    def play(self, speech_file: str) -> None:
        """
        Plays the generated audio file using the system's default player.

        This functionality is useful for quick tests and validation of
        the generated audio quality, working cross-platform.

        Args:
            speech_file: The path to the audio file.

        Raises:
            FileNotFoundError: If the file does not exist.
            RuntimeError: If it fails to play the file.
        """
        try:
            # Validate file existence
            file_path = Path(speech_file)
            if not file_path.exists():
                logger.error(f"Audio file not found for playback: {speech_file}")
                raise FileNotFoundError(f"Audio file not found: {speech_file}")

            # Play using the default OS player
            logger.info(f"Attempting to play audio file: {speech_file}")
            player = FFmpegAudioPlayer(speech_file)
            player.play()
            logger.info(f"Playback of {speech_file} completed.")

        except Exception as e:
            error_msg = f"Error playing audio file {speech_file}: {e}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(f"Error playing audio file: {e}") from e

    def generate_with_custom_config(
        self,
        text: str,
        voice: Optional[Union[GeminiVoice, str]] = None,
        model: Optional[Union[GeminiTTSModel, str]] = None,
        sample_rate: Optional[int] = None,
    ) -> str:
        """
        Generates audio with temporary custom settings.

        This method allows overriding instance settings
        for a specific generation, offering maximum flexibility
        without permanently modifying the tool's configuration.

        Args:
            text: The text to be converted to speech.
            voice: The voice to be used (temporarily overrides).
            model: The model to be used (temporarily overrides).
            sample_rate: The sample rate (temporarily overrides).

        Returns:
            The path to the generated audio file.

        Example:
            ```python
            tts = GeminiText2SpeechTool()  # Default configuration

            # Generate with a specific voice for this text
            file_path = tts.generate_with_custom_config(
                text="Welcome to the formal presentation",
                voice=GeminiVoice.SCHEDAR,  # Majestic voice
                model=GeminiTTSModel.PRO_TTS  # Maximum quality
            )
            ```
        """
        # Preserve original settings
        original_voice = self.voice
        original_model = self.model
        original_sample_rate = self.sample_rate

        logger.info("Generating with custom configuration.")
        logger.debug(
            f"Original config - Voice: {original_voice}, Model: {original_model}, Rate: {original_sample_rate}"
        )

        try:
            # Apply temporary settings
            if voice is not None:
                self.voice = voice
                logger.info(f"Temporarily setting voice to: {voice}")
            if model is not None:
                self.model = model
                logger.info(f"Temporarily setting model to: {model}")
            if sample_rate is not None:
                self.sample_rate = sample_rate
                logger.info(f"Temporarily setting sample_rate to: {sample_rate}")

            # Generate audio with the new settings
            return self._run(text)

        finally:
            # Guarantee restoration of original settings
            self.voice = original_voice
            self.model = original_model
            self.sample_rate = original_sample_rate
            logger.info("Restored original tool configuration.")

    def get_usage_metadata(self, speech_file: str) -> Dict[str, Any]:
        """
        Gets detailed metadata from the generated audio file.

        This method provides technical information and statistics about
        the generated file, useful for quality analysis, debugging,
        and usage reports.

        Args:
            speech_file: The path to the audio file.

        Returns:
            A dictionary with complete metadata of the file.

        Example:
            ```python
            # import logging
            # logging.basicConfig(level=logging.INFO)
            metadata = tts.get_usage_metadata("audio.wav")
            logging.info(f"Duration: {metadata.get('duration_seconds', 0):.2f}s")
            logging.info(f"Quality: {metadata.get('frame_rate', 0)} Hz")
            ```
        """
        logger.info(f"Extracting usage metadata from file: {speech_file}")
        try:
            file_path = Path(speech_file)

            # Basic file metadata
            metadata = {
                "file_path": str(file_path.absolute()),
                "file_name": file_path.name,
                "file_size_bytes": file_path.stat().st_size,
                "file_size_kb": round(file_path.stat().st_size / 1024, 2),
                "file_extension": file_path.suffix,
                "creation_time": file_path.stat().st_ctime,
            }

            # Specific WAV audio metadata
            if file_path.suffix.lower() == ".wav":
                try:
                    with wave.open(str(file_path), "rb") as wav_file:
                        frames = wav_file.getnframes()
                        rate = wav_file.getframerate()
                        duration = frames / rate if rate > 0 else 0

                        metadata.update(
                            {
                                "audio_format": "WAV",
                                "channels": wav_file.getnchannels(),
                                "sample_width_bytes": wav_file.getsampwidth(),
                                "sample_width_bits": wav_file.getsampwidth() * 8,
                                "frame_rate": rate,
                                "total_frames": frames,
                                "duration_seconds": round(duration, 2),
                                "duration_formatted": f"{int(duration // 60)}:{int(duration % 60):02d}",
                                "bitrate_kbps": round((rate * wav_file.getsampwidth() * 8) / 1000, 1),
                            }
                        )

                        # Add quality analysis
                        if rate >= 22050:
                            quality = "High" if rate >= 44100 else "Medium"
                        else:
                            quality = "Low"

                        metadata["quality_assessment"] = quality

                except Exception as wav_error:
                    logger.warning(f"Could not read WAV metadata from {speech_file}: {wav_error}")
                    metadata["audio_error"] = f"Could not read WAV metadata: {wav_error}"

            logger.info(f"Successfully extracted metadata for {speech_file}.")
            return metadata

        except Exception as e:
            logger.error(f"Could not get metadata for {speech_file}: {e}", exc_info=True)
            return {"error": f"Could not get metadata: {e}", "file_path": speech_file}

    def list_all_voices(self) -> Dict[str, List[str]]:
        """
        Lists all available voices organized by category by dynamically
        retrieving them from the GeminiVoice Enum.

        Returns:
            A dictionary with voices organized by origin/category.
        """
        categories = {
            "Greek and Roman Mythology": [
                GeminiVoice.ZEPHYR,
                GeminiVoice.CHARON,
                GeminiVoice.KORE,
                GeminiVoice.AOEDE,
                GeminiVoice.CALLIRRHOE,
                GeminiVoice.AUTONOE,
                GeminiVoice.LEDA,
            ],
            "Norse Mythology": [GeminiVoice.FENRIR],
            "Astronomy - Moons and Satellites": [
                GeminiVoice.ENCELADUS,
                GeminiVoice.IAPETUS,
                GeminiVoice.UMBRIEL,
                GeminiVoice.DESPINA,
                GeminiVoice.ERINOME,
                GeminiVoice.LAOMEDEIA,
            ],
            "Astronomy - Stars": [
                GeminiVoice.ALGIEBA,
                GeminiVoice.ALGENIB,
                GeminiVoice.RASALGETHI,
                GeminiVoice.ACHERNAR,
                GeminiVoice.ALNILAM,
                GeminiVoice.SCHEDAR,
                GeminiVoice.GACRUX,
                GeminiVoice.PULCHERRIMA,
                GeminiVoice.ACHIRD,
                GeminiVoice.ZUBENELGENUBI,
                GeminiVoice.VINDEMIATRIX,
                GeminiVoice.SADACHBIA,
                GeminiVoice.SADALTAGER,
                GeminiVoice.SULAFAT,
            ],
            "Other Origins": [GeminiVoice.PUCK, GeminiVoice.ORUS],
        }

        return {category: [voice.value for voice in voices] for category, voices in categories.items()}

    def compare_voices(self, text: str, voices: List[GeminiVoice]) -> Dict[str, str]:
        """
        Generates the same text with multiple voices for comparison.

        Args:
            text: The text to generate with different voices.
            voices: A list of voices to compare.

        Returns:
            A dictionary mapping the voice name to the path of the generated file.
        """
        results = {}
        logger.info(f"Starting voice comparison for voices: {[v.value for v in voices]}")

        for voice in voices:
            try:
                logger.info(f"Generating audio for comparison with voice: {voice.value}")
                file_path = self.generate_with_custom_config(text=text, voice=voice)
                results[voice.value] = file_path
                logger.info(f"Successfully generated file for voice {voice.value}: {file_path}")
            except Exception as e:
                logger.error(f"Failed to generate audio for voice {voice.value}: {e}", exc_info=True)
                results[voice.value] = f"Error: {e}"

        logger.info("Voice comparison finished.")
        return results
