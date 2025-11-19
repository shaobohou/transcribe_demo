"""Configuration classes for transcription backends.

This module provides structured configuration for Whisper and Realtime backends,
replacing the previous pattern of passing 20+ individual arguments.

All configuration dataclasses are used with simple-parsing for CLI generation.
"""

from __future__ import annotations

import dataclasses
from typing import Literal


@dataclasses.dataclass(frozen=True, kw_only=True)
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


@dataclasses.dataclass(frozen=True, kw_only=True)
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


@dataclasses.dataclass(frozen=True, kw_only=True)
class WhisperConfig:
    """
    Configuration for Whisper backend (local transcription).

    Contains all settings for model selection, device placement, VAD,
    partial transcription, and session management.
    """

    # Model settings
    model: str = "turbo"
    """Whisper model name (e.g., 'turbo', 'base.en', 'small')."""

    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    """Device to run on: 'auto', 'cpu', 'cuda', or 'mps'."""

    require_gpu: bool = False
    """If True, exit if GPU unavailable instead of falling back to CPU."""

    # VAD settings
    vad: VADConfig = dataclasses.field(default_factory=VADConfig)
    """Voice Activity Detection configuration."""

    # Partial transcription
    partial: PartialTranscriptionConfig = dataclasses.field(default_factory=PartialTranscriptionConfig)
    """Partial transcription configuration."""

    # Debugging
    temp_file: str | None = None
    """Optional path to persist audio chunks for inspection."""


@dataclasses.dataclass(frozen=True, kw_only=True)
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


@dataclasses.dataclass(frozen=True, kw_only=True)
class RealtimeConfig:
    """
    Configuration for Realtime API backend (cloud transcription).

    Contains all settings for API credentials, endpoints, server-side VAD,
    and session management.
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
    vad: RealtimeVADConfig = dataclasses.field(default_factory=RealtimeVADConfig)
    """Server-side Voice Activity Detection configuration."""

    # Debugging
    debug: bool = False
    """Enable debug logging for realtime transcription events."""


@dataclasses.dataclass(frozen=True, kw_only=True)
class AudioConfig:
    """Audio capture and playback configuration."""

    sample_rate: int = 16000
    """Input sample rate expected by the model."""

    channels: int = 1
    """Number of microphone input channels."""

    audio_file: str | None = None
    """Path or URL to audio file for simulating live transcription (MP3, WAV, FLAC, etc.).
    Supports local files and HTTP/HTTPS URLs. If provided, audio will be read from
    file/URL instead of microphone."""

    playback_speed: float = 1.0
    """Playback speed multiplier when using audio_file (1.0 = real-time, 2.0 = 2x speed)."""

    def __post_init__(self) -> None:
        """Validate audio configuration."""
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate}")
        if self.channels <= 0:
            raise ValueError(f"channels must be positive, got {self.channels}")
        if not 0.1 <= self.playback_speed <= 10.0:
            raise ValueError(f"playback_speed must be 0.1-10.0, got {self.playback_speed}")


@dataclasses.dataclass(frozen=True, kw_only=True)
class SessionConfig:
    """Session logging and comparison configuration."""

    session_log_dir: str = "./session_logs"
    """Directory to save session logs. All sessions are logged with full audio,
    chunk audio, and metadata."""

    audio_format: Literal["wav", "flac"] = "flac"
    """Audio format for saved session files. 'flac' provides lossless compression
    (~50-60% smaller), 'wav' is uncompressed."""

    min_log_duration: float = 10.0
    """Minimum session duration (seconds) required to save logs. Sessions shorter
    than this are discarded."""

    compare_transcripts: bool = True
    """Compare chunked transcription with full-audio transcription at session end.
    Note: For Realtime API, this doubles API usage cost."""

    max_capture_duration: float = 120.0
    """Maximum duration (seconds) to run the transcription session. Program will
    gracefully stop after this duration. Set to 0 for unlimited duration."""

    def __post_init__(self) -> None:
        """Validate session configuration."""
        if self.audio_format not in ("wav", "flac"):
            raise ValueError(f"audio_format must be 'wav' or 'flac', got {self.audio_format}")
        if self.min_log_duration < 0:
            raise ValueError(f"min_log_duration must be non-negative, got {self.min_log_duration}")
        if self.max_capture_duration < 0:
            raise ValueError(f"max_capture_duration must be non-negative, got {self.max_capture_duration}")


@dataclasses.dataclass(frozen=True, kw_only=True)
class CLIConfig:
    """
    Complete CLI configuration for transcribe-demo.

    This is the top-level configuration that combines all settings and is used
    with simple-parsing to auto-generate command-line arguments.
    """

    backend: Literal["whisper", "realtime"] = "whisper"
    """Transcription backend to use."""

    language: str = "en"
    """Preferred language code for transcription (e.g., en, es). Use 'auto' to let
    the model detect. WARNING: 'auto' can cause hallucinations on silence."""

    # Nested configurations
    audio: AudioConfig = dataclasses.field(default_factory=AudioConfig)
    """Audio capture and playback settings."""

    session: SessionConfig = dataclasses.field(default_factory=SessionConfig)
    """Session logging and comparison settings."""

    whisper: WhisperConfig = dataclasses.field(default_factory=WhisperConfig)
    """Whisper backend configuration (used when backend='whisper')."""

    realtime: RealtimeConfig = dataclasses.field(default_factory=RealtimeConfig)
    """Realtime API configuration (used when backend='realtime')."""

    # Feature flags
    refine_with_context: bool = False
    """[NOT YET IMPLEMENTED] Use 3-chunk sliding window to refine middle chunk transcription."""

    # SSL/Certificate settings (global)
    ca_cert: str | None = None
    """Custom certificate bundle to trust when downloading models or connecting to APIs."""

    disable_ssl_verify: bool = False
    """Disable SSL certificate verification for all network operations. WARNING: This is
    insecure and not recommended for production use."""

    def get_backend_config(self) -> WhisperConfig | RealtimeConfig:
        """
        Get the appropriate backend configuration based on the selected backend.

        Returns:
            WhisperConfig if backend='whisper', RealtimeConfig if backend='realtime'
        """
        if self.backend == "whisper":
            return self.whisper
        else:  # realtime
            return self.realtime
