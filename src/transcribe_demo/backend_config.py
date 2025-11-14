"""Configuration classes for transcription backends.

This module provides structured configuration for Whisper and Realtime backends,
replacing the previous pattern of passing 20+ individual arguments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from transcribe_demo.backend_protocol import BackendConfig


@dataclass
class VADConfig:
    """
    Voice Activity Detection configuration.

    These settings control how audio is chunked based on speech detection.
    See DESIGN.md for detailed explanation of each parameter.
    """

    aggressiveness: int = 2
    """VAD aggressiveness level (0-3). Higher = more aggressive filtering."""

    min_silence_duration: float = 0.2
    """Minimum duration of silence (seconds) to trigger chunk split."""

    min_speech_duration: float = 0.25
    """Minimum duration of speech (seconds) required before transcribing."""

    speech_pad_duration: float = 0.2
    """Padding duration (seconds) added before speech to avoid cutting words."""

    max_chunk_duration: float = 60.0
    """Maximum chunk duration (seconds). Prevents buffer overflow."""

    def __post_init__(self) -> None:
        """Validate VAD configuration."""
        if not 0 <= self.aggressiveness <= 3:
            raise ValueError(f"aggressiveness must be 0-3, got {self.aggressiveness}")
        if self.min_silence_duration <= 0:
            raise ValueError(f"min_silence_duration must be positive, got {self.min_silence_duration}")
        if self.min_speech_duration <= 0:
            raise ValueError(f"min_speech_duration must be positive, got {self.min_speech_duration}")
        if self.speech_pad_duration < 0:
            raise ValueError(f"speech_pad_duration must be non-negative, got {self.speech_pad_duration}")
        if self.max_chunk_duration <= 0:
            raise ValueError(f"max_chunk_duration must be positive, got {self.max_chunk_duration}")


@dataclass
class PartialTranscriptionConfig:
    """
    Configuration for partial (progressive) transcription.

    Partial transcription provides real-time updates of accumulating audio
    before the chunk is finalized.
    """

    enabled: bool = False
    """Whether to enable partial transcription."""

    model: str = "base.en"
    """Model to use for partial transcription (should be fast, e.g., base.en, tiny.en)."""

    interval: float = 1.0
    """Interval (seconds) between partial transcription updates."""

    max_buffer_seconds: float = 10.0
    """Segment duration (seconds) for partial transcription."""

    def __post_init__(self) -> None:
        """Validate partial transcription configuration."""
        if self.interval <= 0:
            raise ValueError(f"interval must be positive, got {self.interval}")
        if not 1.0 <= self.max_buffer_seconds <= 60.0:
            raise ValueError(f"max_buffer_seconds must be 1.0-60.0, got {self.max_buffer_seconds}")


@dataclass
class WhisperConfig(BackendConfig):
    """
    Configuration for Whisper backend (local transcription).

    Extends BackendConfig with Whisper-specific settings for model selection,
    device placement, VAD, and partial transcription.
    """

    # Model settings
    model: str = "turbo"
    """Whisper model name (e.g., 'turbo', 'base.en', 'small')."""

    device: str = "auto"
    """Device to run on: 'auto', 'cpu', 'cuda', or 'mps'."""

    require_gpu: bool = False
    """If True, exit if GPU unavailable instead of falling back to CPU."""

    # VAD settings
    vad: VADConfig = field(default_factory=VADConfig)
    """Voice Activity Detection configuration."""

    # Partial transcription
    partial: PartialTranscriptionConfig = field(default_factory=PartialTranscriptionConfig)
    """Partial transcription configuration."""

    # SSL/Certificate settings (for model downloads)
    ca_cert: Path | None = None
    """Custom certificate bundle to trust when downloading models."""

    # Debugging
    temp_file: Path | None = None
    """Optional path to persist audio chunks for inspection."""


@dataclass
class RealtimeVADConfig:
    """
    Voice Activity Detection configuration for Realtime API.

    Note: Realtime API uses server-side VAD with different parameters
    than WebRTC VAD used in Whisper backend.
    """

    threshold: float = 0.2
    """
    Server VAD threshold for turn detection (0.0-1.0).
    Lower = more sensitive. Default 0.2 works well for continuous speech.
    """

    silence_duration_ms: int = 100
    """
    Silence duration (milliseconds) required to detect turn boundary.
    Lower values = more frequent chunks. Default 100ms works for fast-paced content.
    """

    def __post_init__(self) -> None:
        """Validate Realtime VAD configuration."""
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(f"threshold must be 0.0-1.0, got {self.threshold}")
        if not 100 <= self.silence_duration_ms <= 2000:
            raise ValueError(f"silence_duration_ms must be 100-2000, got {self.silence_duration_ms}")


@dataclass
class RealtimeConfig(BackendConfig):
    """
    Configuration for Realtime API backend (cloud transcription).

    Extends BackendConfig with Realtime API-specific settings for
    API credentials, endpoints, and server-side VAD.
    """

    # API settings
    api_key: str | None = None
    """OpenAI API key. If None, reads from OPENAI_API_KEY env var."""

    model: str = "gpt-realtime-mini"
    """Realtime model to use."""

    endpoint: str = "wss://api.openai.com/v1/realtime"
    """Realtime websocket endpoint (advanced)."""

    instructions: str = (
        "You are a high-accuracy transcription service. "
        "Return a concise verbatim transcript of the most recent audio buffer. "
        "Do not add commentary or speaker labels."
    )
    """Instruction prompt sent to the realtime model."""

    # Chunking (fixed for Realtime API)
    chunk_duration: float = 2.0
    """
    Fixed chunk duration for Realtime API (seconds).
    Note: This is NOT configurable like Whisper's VAD-based chunking.
    """

    # Server-side VAD
    vad: RealtimeVADConfig = field(default_factory=RealtimeVADConfig)
    """Server-side Voice Activity Detection configuration."""

    # Debugging
    debug: bool = False
    """Enable debug logging for realtime transcription events."""

    def __post_init__(self) -> None:
        """Validate Realtime configuration."""
        if self.api_key is None:
            import os

            self.api_key = os.getenv("OPENAI_API_KEY")
            if self.api_key is None:
                raise ValueError(
                    "api_key is required for Realtime backend. "
                    "Provide api_key or set OPENAI_API_KEY environment variable."
                )
