"""Protocol definitions for transcription backends.

This module defines the abstract interfaces that all transcription backends
must implement, enabling type-safe polymorphism and easier addition of new backends.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from transcribe_demo.session_logger import SessionLogger


@dataclass
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


@dataclass
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
    session_logger: SessionLogger | None = None
    min_log_duration: float = 10.0

    # SSL/Security
    disable_ssl_verify: bool = False


class TranscriptionBackend(Protocol):
    """
    Protocol for transcription backends.

    All backends (Whisper, Realtime, future backends) should implement
    this interface to enable polymorphic usage.

    Example:
        def run_transcription(backend: TranscriptionBackend, config: BackendConfig):
            result = backend.transcribe(
                config=config,
                chunk_consumer=my_consumer,
            )
            print(f"Duration: {result.capture_duration}s")
    """

    def transcribe(
        self,
        config: BackendConfig,
        chunk_consumer: ChunkConsumer | None = None,
    ) -> TranscriptionResult:
        """
        Run transcription with the given configuration.

        Args:
            config: Backend-specific configuration object
            chunk_consumer: Optional callback for processing chunks in real-time

        Returns:
            TranscriptionResult with capture duration, full transcription, and metadata

        Raises:
            RuntimeError: If transcription fails
        """
        ...


# Convenience type alias for the legacy chunk consumer signature used by existing code
# This allows gradual migration to the new TranscriptionChunk-based interface
LegacyChunkConsumer = Callable[[int, str, float, float, float | None, bool], None] | None


def adapt_legacy_consumer(legacy_consumer: LegacyChunkConsumer) -> ChunkConsumer | None:
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
