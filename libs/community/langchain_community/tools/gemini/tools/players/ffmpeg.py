"""
Audio player using FFmpeg/FFplay.

This implementation leverages the robustness and compatibility of FFmpeg to
create a player that works with hundreds of different formats.
"""

import logging
import shutil
import subprocess
import time
from pathlib import Path
from typing import List, Optional, Union

from .abstracts import AbstractAudioPlayer, PlaybackState

log = logging.getLogger(__name__)


class FFmpegAudioPlayer(AbstractAudioPlayer):
    """
    Audio player using FFmpeg/FFplay.

    This player delegates playback to FFplay, which is part of the
    FFmpeg tool suite. It supports virtually any audio or video format.

    ADVANTAGES OF FFMPEG:
    ===================
    • Universal format support
    • Optimized decoding and hardware acceleration
    • Ability to stream from remote URLs

    LIMITATIONS:
    ===========
    • Requires FFmpeg to be installed on the system.
    • Less programmatic control during playback.
    """

    def __init__(self, audio_source: Union[str, Path, bytes], **kwargs):
        """
        Initializes the FFmpeg player.

        Args:
            audio_source: The source of the audio (file path, URL, or bytes).
            **kwargs: Additional arguments from the base class.
        """
        super().__init__(**kwargs)

        self.audio_source = audio_source
        self.is_bytes_audio = isinstance(audio_source, bytes)
        self.is_remote_url = isinstance(audio_source, str) and audio_source.startswith(("http://", "https://"))
        self.temp_file: Optional[Path] = None
        self.ffplay_process: Optional[subprocess.Popen] = None

        self._verify_ffplay_availability()

    def _verify_ffplay_availability(self) -> None:
        """
        Checks if FFplay is installed and accessible on the system.

        Raises:
            RuntimeError: If FFplay is not available.
        """
        if not shutil.which("ffplay"):
            system_instructions = {
                "win32": "Windows: Download from https://ffmpeg.org or use 'choco install ffmpeg'",
                "darwin": "macOS: Run 'brew install ffmpeg'",
                "linux": "Linux: Run 'sudo apt install ffmpeg'",
            }
            import sys

            instruction = system_instructions.get(sys.platform, "Install FFmpeg from https://ffmpeg.org")
            raise RuntimeError(f"FFplay not found. It is required for playback.\nInstallation: {instruction}")

    def _prepare_playback(self) -> bool:
        """
        Prepares the audio source for playback with FFplay.

        Returns:
            bool: True if preparation was successful.
        """
        try:
            if self.is_bytes_audio:
                import tempfile

                # The extension helps FFplay determine the format.
                self.temp_file = Path(tempfile.mktemp(suffix=".wav"))
                with open(self.temp_file, "wb") as f:
                    f.write(self.audio_source)
                log.debug("Audio data saved to temporary file: %s", self.temp_file)

            elif self.is_remote_url:
                log.debug("Preparing remote stream: %s", self.audio_source)

            else:
                audio_path = Path(self.audio_source)
                if not audio_path.exists():
                    log.error("File not found: %s", audio_path)
                    return False
                if not audio_path.is_file():
                    log.error("Path is not a file: %s", audio_path)
                    return False
                log.debug("Local file validated: %s", audio_path)

            return True

        except Exception:
            log.exception("Error during playback preparation.")
            return False

    def _cleanup_playback(self) -> None:
        """Cleans up FFmpeg-specific resources."""
        try:
            if self.ffplay_process and self.ffplay_process.poll() is None:
                self.ffplay_process.terminate()
                try:
                    self.ffplay_process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    self.ffplay_process.kill()

            if self.temp_file and self.temp_file.exists():
                self.temp_file.unlink()
                log.debug("Temporary file removed: %s", self.temp_file)

        except Exception as e:
            log.warning("Error during cleanup: %s", e, exc_info=True)
        finally:
            self.ffplay_process = None
            self.temp_file = None

    def _play_audio(self) -> None:
        """Implements specific playback using FFplay."""
        try:
            if self.is_bytes_audio:
                audio_input = str(self.temp_file)
            else:
                audio_input = str(self.audio_source)

            ffplay_args = [
                "ffplay",
                "-autoexit",
                "-nodisp",
                "-loglevel",
                "error",  # Log only critical errors from ffplay
                audio_input,
            ]
            log.debug("Starting FFplay with command: %s", " ".join(ffplay_args))

            self.ffplay_process = subprocess.Popen(
                ffplay_args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
            )
            log.info("Playback started (PID: %d)", self.ffplay_process.pid)

            while not self._stop_event.is_set():
                return_code = self.ffplay_process.poll()
                if return_code is not None:
                    if return_code == 0:
                        log.info("Playback completed successfully.")
                    else:
                        stderr_output = self.ffplay_process.stderr.read().decode("utf-8", errors="ignore")
                        log.warning("FFplay exited with code %d. Error: %s", return_code, stderr_output.strip())
                    break
                time.sleep(0.1)

            if self._stop_event.is_set() and self.ffplay_process.poll() is None:
                log.info("Stopping playback...")
                self.ffplay_process.terminate()
                try:
                    self.ffplay_process.wait(timeout=2.0)
                    log.debug("Process terminated gracefully.")
                except subprocess.TimeoutExpired:
                    log.warning("Forcing process termination after timeout.")
                    self.ffplay_process.kill()

        except Exception:
            log.exception("An unhandled error occurred during playback.")
            self._set_state(PlaybackState.ERROR)
            raise

    def get_supported_formats(self) -> List[str]:
        """Returns a list of formats commonly supported by FFmpeg."""
        return [
            "mp3",
            "wav",
            "flac",
            "ogg",
            "aac",
            "m4a",
            "wma",
            "opus",
            "webm",
            "ape",
            "ac3",
            "dts",
            "amr",
            "aiff",
            "au",
            "raw",
            "pcm",
            "mkv",
            "mp4",
            "avi",
            "mov",
            "wmv",
        ]

    def supports_remote_urls(self) -> bool:
        """Indicates whether this player supports remote URLs."""
        return True

    def get_file_info(self) -> dict:
        """Gets information about the audio source."""
        # This method remains unchanged as it doesn't use print
        try:
            base_info = {
                "player_type": "ffmpeg",
                "ffplay_available": shutil.which("ffplay") is not None,
                "supports_remote_streaming": True,
                "supported_formats": len(self.get_supported_formats()),
                "state": self.get_state().value,
            }
            if self.is_bytes_audio:
                base_info.update(
                    {
                        "source_type": "bytes",
                        "data_size_bytes": len(self.audio_source),
                        "temp_file": str(self.temp_file) if self.temp_file else None,
                    }
                )
            elif self.is_remote_url:
                base_info.update({"source_type": "remote_url", "url": str(self.audio_source)})
            else:
                audio_path = Path(self.audio_source)
                if audio_path.exists():
                    stat = audio_path.stat()
                    base_info.update(
                        {
                            "source_type": "local_file",
                            "file_path": str(audio_path.absolute()),
                            "file_name": audio_path.name,
                            "file_size_bytes": stat.st_size,
                            "file_extension": audio_path.suffix.lower(),
                        }
                    )
            return base_info
        except Exception as e:
            return {"player_type": "ffmpeg", "error": str(e)}

    def __str__(self) -> str:
        """String representation of the player."""
        source_desc = "bytes" if self.is_bytes_audio else str(self.audio_source)
        return f"FFmpegAudioPlayer('{source_desc}', state={self.get_state().value})"

    def __repr__(self) -> str:
        """Technical representation of the player."""
        return (
            f"FFmpegAudioPlayer(audio_source='{self.audio_source}', "
            f"is_bytes={self.is_bytes_audio}, is_remote={self.is_remote_url})"
        )
