"""Protocol definitions for transcription backends.

This module defines the abstract interfaces that all transcription backends
must implement, enabling type-safe polymorphism and easier addition of new backends.
"""

from __future__ import annotations

import dataclasses
import queue
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from transcribe_demo import session_logger


@dataclasses.dataclass(frozen=True, kw_only=True)
class TranscriptionChunk:
    """
    Represents a single transcribed chunk of audio.

    This is the unified chunk representation passed to chunk consumers,
    replacing the various positional/keyword argument patterns.
    """

    index: int
    text: str
    start_time: float
    end_time: float
    inference_seconds: float | None = None
    is_partial: bool = False
    audio: np.ndarray | None = None


class ChunkConsumer(Protocol):
    """
    Protocol for consuming transcription chunks.

    All chunk consumers (e.g., ChunkCollectorWithStitching) should
    implement this interface.
    """

    def __call__(self, chunk: TranscriptionChunk) -> None:
        """
        Process a transcription chunk.

        Args:
            chunk: The transcribed chunk with metadata
        """
        ...


class TranscriptionResult(Protocol):
    """
    Protocol for transcription results returned by backends.

    This defines the common interface that both WhisperTranscriptionResult
    and RealtimeTranscriptionResult should implement.
    """

    @property
    def capture_duration(self) -> float:
        """Total duration of audio captured (seconds)."""
        ...

    @property
    def full_audio_transcription(self) -> str | None:
        """Complete audio transcription (if comparison enabled)."""
        ...

    @property
    def metadata(self) -> dict[str, Any]:
        """Backend-specific metadata (model, device, VAD params, etc.)."""
        ...


@dataclasses.dataclass(frozen=True, kw_only=True)
class BackendConfig:
    """
    Base configuration common to all transcription backends.

    Backend-specific configs should inherit from this class.
    """

    # Audio settings
    sample_rate: int = 16000
    channels: int = 1

    # Session settings
    max_capture_duration: float = 120.0
    compare_transcripts: bool = True
    language: str = "en"

    # Audio source
    audio_file: Path | str | None = None
    playback_speed: float = 1.0

    # Session logging
    session_logger: session_logger.SessionLogger | None = None
    min_log_duration: float = 10.0

    # SSL/Security
    disable_ssl_verify: bool = False


class AudioSource(Protocol):
    """
    Protocol for audio sources (microphone, file, etc.).

    Audio sources provide audio data and control audio capture lifecycle.
    """

    @property
    def sample_rate(self) -> int:
        """Audio sample rate in Hz."""
        ...

    @property
    def channels(self) -> int:
        """Number of audio channels."""
        ...

    @property
    def max_capture_duration(self) -> float:
        """Maximum capture duration in seconds (0.0 means unlimited)."""
        ...

    @property
    def stop_event(self) -> threading.Event:
        """Event that signals when audio capture should stop."""
        ...

    @property
    def capture_limit_reached(self) -> threading.Event:
        """Event that signals when max_capture_duration is reached."""
        ...

    @property
    def audio_queue(self) -> queue.Queue[np.ndarray | None]:
        """Queue that provides audio chunks."""
        ...

    def start(self) -> None:
        """Start audio capture."""
        ...

    def stop(self) -> None:
        """Stop audio capture."""
        ...

    def wait_until_stopped(self) -> None:
        """Block until audio capture is stopped."""
        ...

    def close(self) -> None:
        """Clean up resources."""
        ...

    def get_full_audio(self) -> np.ndarray:
        """Get the complete captured audio."""
        ...

    def get_capture_duration(self) -> float:
        """Get the total duration of captured audio in seconds."""
        ...


class TranscriptionBackend(Protocol):
    """
    Protocol for transcription backends.

    All backends (Whisper, Realtime, future backends) should implement
    this interface to enable polymorphic usage.

    Example:
        backend = WhisperBackend(model_name="turbo", ...)
        audio_source = FileAudioSource(...)
        chunk_queue = queue.Queue()
        result = backend.run(audio_source, chunk_queue)
        print(f"Duration: {result.capture_duration}s")
    """

    def run(
        self,
        *,
        audio_source: AudioSource,
        chunk_queue: queue.Queue[TranscriptionChunk | None],
    ) -> TranscriptionResult:
        """
        Run transcription on the given audio source.

        Args:
            audio_source: Audio source to transcribe from
            chunk_queue: Queue to put transcription chunks into

        Returns:
            TranscriptionResult with capture duration, full transcription, and metadata

        Raises:
            RuntimeError: If transcription fails
        """
        ...


# Convenience type alias for the legacy chunk consumer signature used by existing code
# This allows gradual migration to the new TranscriptionChunk-based interface
_LegacyChunkConsumer = Callable[[int, str, float, float, float | None, bool], None] | None


def _adapt_legacy_consumer(*, legacy_consumer: _LegacyChunkConsumer) -> ChunkConsumer | None:
    """
    Adapt a legacy chunk consumer to the new ChunkConsumer protocol.

    This helper function allows gradual migration of existing code to the new
    TranscriptionChunk-based interface.

    Args:
        legacy_consumer: Old-style chunk consumer with positional args

    Returns:
        New-style ChunkConsumer that accepts TranscriptionChunk, or None

    Example:
        old_consumer = lambda idx, text, start, end, inf, partial: print(text)
        new_consumer = adapt_legacy_consumer(old_consumer)
        new_consumer(TranscriptionChunk(index=0, text="Hello", ...))
    """
    if legacy_consumer is None:
        return None

    def adapted_consumer(chunk: TranscriptionChunk) -> None:
        legacy_consumer(
            chunk.index,
            chunk.text,
            chunk.start_time,
            chunk.end_time,
            chunk.inference_seconds,
            chunk.is_partial,
        )

    return adapted_consumer
