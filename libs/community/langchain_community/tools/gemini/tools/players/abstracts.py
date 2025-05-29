"""
Base abstract class for audio players.

This module defines the common interface and basic behaviors that all
audio players must implement, following the Template Method pattern.

The AbstractAudioPlayer class acts as a "contract" that ensures all
specific players (pygame, sounddevice, playsound) have the same
external interface, even if they implement playback differently
internally.

Important concepts implemented:
- Template Method Pattern: Defines the general playback algorithm.
- State Pattern: Manages playback states in a thread-safe manner.
- Context Manager: Supports with/as for automatic cleanup.
- Observer Pattern: Controls stop events via threading.Event.
"""

import logging
import threading
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional

# Set up a logger for this module. The calling application should configure the handler.
logger = logging.getLogger(__name__)

# Conditional import of pyaudio to maintain compatibility.
# If pyaudio is not available, we use default values.
try:
    import pyaudio

    _PYAUDIO_AVAILABLE = True
except ImportError:
    _PYAUDIO_AVAILABLE = False
    logger.debug("pyaudio library not found. Using fallback constants.")

    # Define equivalent constants if pyaudio is not available.
    class _PyAudioConstants:
        paInt16 = 16
        paInt8 = 8
        paInt24 = 24
        paInt32 = 32

    pyaudio = _PyAudioConstants()


class PlaybackState(Enum):
    """
    Possible states of the audio player.

    This enumeration defines all the states a player can be in
    during its lifecycle. Using an Enum ensures type safety and makes
    the code more readable and less prone to errors.

    Available states:
    - STOPPED: Player is stopped, position is at the beginning.
    - PLAYING: Active playback is in progress.
    - PAUSED: Playback is paused, position is maintained.
    - ERROR: An error occurred during operation, the player is unusable.
    """

    STOPPED = "stopped"
    PLAYING = "playing"
    PAUSED = "paused"
    ERROR = "error"


class AbstractAudioPlayer(ABC):
    """
    Base abstract class for audio players.

    This class implements the Template Method pattern, defining the common
    structure for audio playback while allowing subclasses to
    implement the specific details of each technology (pygame,
    sounddevice, playsound, etc.).

    TEMPLATE METHOD PATTERN EXPLAINED:
    ================================

    The Template Method is a design pattern that defines the skeleton of an
    algorithm in a base class but allows subclasses to redefine
    specific steps without changing the overall structure.

    In our case:
    1. play() defines the general steps: prepare -> start thread -> monitor
    2. _play_audio() is abstract - each subclass implements it as desired.
    3. The result is a consistent interface with flexible implementations.

    THREAD SAFETY:
    =============

    This class is designed to be thread-safe, allowing multiple
    threads to interact with the player without causing race conditions.
    We use locks to protect shared state and events for
    inter-thread communication.

    CONTEXT MANAGER:
    ===============

    The class supports the context manager protocol (with/as), ensuring
    that resources are cleaned up automatically even if exceptions occur.
    """

    def __init__(
        self,
        audio_format: int = None,
        channels: int = 1,
        rate: int = 44100,
        chunk: int = 1024,
        audio_interface: Optional[object] = None,
    ):
        """
        Initializes the base audio player.

        This constructor establishes the basic configuration that will be used
        by all subclasses. Some parameters may be ignored
        by specific implementations (like playsound), but we maintain
        a consistent interface.

        Args:
            audio_format: Audio format (16-bit by default if pyaudio is available).
            channels: Number of audio channels (1=mono, 2=stereo).
            rate: Sample rate in Hz (44100 is standard CD quality).
            chunk: Buffer size in frames (1024 is a good balance).
            audio_interface: Custom audio interface (used by PyAudio implementations).

        PARAMETER EXPLANATION:
        ========================

        - audio_format: Determines the audio precision (8, 16, 24, 32 bits).
        - channels: Mono (1) uses fewer resources, Stereo (2) for better quality.
        - rate: 44100Hz is CD standard, 48000Hz is professional standard.
        - chunk: Smaller buffers = lower latency, larger buffers = less CPU usage.
        """
        # Audio settings - sensible default values
        self._format = audio_format or (pyaudio.paInt16 if _PYAUDIO_AVAILABLE else 16)
        self._channels = channels
        self._rate = rate
        self._chunk = chunk

        # Audio interface - mainly for compatibility with PyAudio.
        # In most background implementations, this parameter is ignored.
        self._audio_interface = audio_interface

        # THREAD-SAFE STATE CONTROL
        self._state = PlaybackState.STOPPED
        self._state_lock = threading.Lock()

        # THREADING CONTROL
        self._play_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # AUDIO RESOURCES
        self._stream: Optional[object] = None

        logger.debug(
            f"{self.__class__.__name__} initialized with config: "
            f"format={self._format}, channels={self._channels}, "
            f"rate={self._rate}, chunk={self._chunk}"
        )

    def play(self) -> bool:
        """
        Starts audio playback using the Template Method pattern.
        This method is thread-safe.
        """
        with self._state_lock:
            if self._state == PlaybackState.PLAYING:
                logger.warning("Play command ignored: player is already playing.")
                return False

            logger.info("Preparing for playback...")
            if not self._prepare_playback():
                logger.error("Playback preparation failed. Cannot start playing.")
                self._set_state(PlaybackState.ERROR)
                return False

            self._stop_event.clear()
            self._play_thread = threading.Thread(
                target=self._playback_worker, name=f"{self.__class__.__name__}-PlaybackThread"
            )
            self._play_thread.daemon = True

            logger.info("Starting playback thread...")
            self._set_state(PlaybackState.PLAYING)
            self._play_thread.start()

            return True

    def stop(self) -> None:
        """
        Stops audio playback gracefully and cleans up resources.
        This method is thread-safe.
        """
        with self._state_lock:
            if self._state == PlaybackState.STOPPED:
                return  # Already stopped

            logger.info("Stop command received. Initiating shutdown.")
            self._stop_event.set()
            self._set_state(PlaybackState.STOPPED)

        if self._play_thread and self._play_thread.is_alive():
            logger.debug("Waiting for playback thread to terminate...")
            self._play_thread.join(timeout=2.0)
            if self._play_thread.is_alive():
                logger.warning("Playback thread did not terminate in the expected time.")

        logger.debug("Cleaning up playback resources.")
        self._cleanup_playback()
        logger.info("Player stopped successfully.")

    def is_playing(self) -> bool:
        """
        Checks if audio is currently playing in a thread-safe manner.
        """
        with self._state_lock:
            return self._state == PlaybackState.PLAYING

    def get_state(self) -> PlaybackState:
        """
        Gets the current state of the player in a thread-safe manner.
        """
        with self._state_lock:
            return self._state

    def wait_until_finished(self, timeout: Optional[float] = None) -> bool:
        """
        Blocks until playback has finished or a timeout occurs.
        """
        if self._play_thread and self._play_thread.is_alive():
            logger.debug(f"Waiting for playback to finish (timeout: {timeout}s)...")
            self._play_thread.join(timeout=timeout)
            finished = not self._play_thread.is_alive()
            logger.debug(f"Wait finished. Player has stopped: {finished}")
            return finished
        logger.debug("wait_until_finished called, but no active playback thread.")
        return True

    def _prepare_playback(self) -> bool:
        """
        Prepares audio system for playback. To be overridden by subclasses.
        """
        logger.debug("Executing base _prepare_playback method.")
        return True

    def _cleanup_playback(self) -> None:
        """
        Safely cleans up playback resources. To be overridden by subclasses.
        """
        logger.debug("Executing base _cleanup_playback method.")
        pass

    def _playback_worker(self) -> None:
        """
        Worker thread that performs playback and manages state.
        """
        logger.debug("Playback worker thread started.")
        try:
            self._play_audio()
        except Exception as e:
            logger.error(f"Error during playback: {e}", exc_info=True)
            self._set_state(PlaybackState.ERROR)
        finally:
            if self._stop_event.is_set():
                logger.info("Playback worker stopped due to stop event.")
            else:
                # If it wasn't stopped manually, it finished naturally
                with self._state_lock:
                    if self._state == PlaybackState.PLAYING:
                        logger.info("Playback finished naturally.")
                        self._set_state(PlaybackState.STOPPED)
        logger.debug("Playback worker thread finished.")

    def _set_state(self, new_state: PlaybackState) -> None:
        """
        Sets the player's state in a thread-safe manner and logs the transition.
        """
        # No lock needed here as it's called from methods that already hold the lock.
        if self._state != new_state:
            logger.info(f"Player state changed from {self._state.value} to {new_state.value}")
            self._state = new_state

    @abstractmethod
    def _play_audio(self) -> None:
        """
        Abstract method that implements the format-specific playback.
        This method MUST be implemented by subclasses.
        It must periodically check self._stop_event.
        """
        pass

    def __enter__(self):
        """
        Context manager entry - starts playback automatically.
        """
        logger.debug("Entering context manager, starting playback.")
        self.play()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit - stops playback and cleans up resources.
        """
        logger.debug("Exiting context manager, ensuring player is stopped.")
        self.stop()

    def __del__(self):
        """
        Destructor that ensures resource cleanup as a final safeguard.
        """
        try:
            if self._state == PlaybackState.PLAYING:
                logger.warning(
                    f"{self.__class__.__name__} object garbage collected while still playing. "
                    "Forcing stop. Use a context manager or call stop() explicitly."
                )
            self.stop()
            if self._audio_interface and hasattr(self._audio_interface, "terminate"):
                logger.debug("Terminating audio interface in destructor.")
                self._audio_interface.terminate()
        except Exception as e:
            # Destructors should not raise exceptions.
            logger.error(f"Error during {self.__class__.__name__} destructor: {e}", exc_info=False)
            pass
