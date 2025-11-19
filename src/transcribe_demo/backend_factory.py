"""Factory functions for creating transcription backends from configuration objects.

This module provides clean factory functions that accept configuration dataclasses
instead of directly accessing global FLAGS, improving modularity and testability.
"""

from __future__ import annotations

from pathlib import Path

from transcribe_demo import backend_config, realtime_backend, session_logger, whisper_backend


def create_whisper_backend(
    *,
    config: backend_config.WhisperConfig,
    language: str,
    compare_transcripts: bool,
    min_log_duration: float,
    ca_cert: str | None,
    disable_ssl_verify: bool,
    session_logger: session_logger.SessionLogger | None,
) -> whisper_backend.WhisperBackend:
    """
    Create and configure a Whisper backend from a configuration object.

    Args:
        config: WhisperConfig with backend-specific settings
        language: Language code for transcription
        compare_transcripts: Whether to compare chunked vs full audio transcription
        min_log_duration: Minimum session duration for logging
        ca_cert: Custom certificate bundle path (None for system default)
        disable_ssl_verify: Whether to disable SSL verification
        session_logger: Session logger for persistence

    Returns:
        Configured WhisperBackend instance
    """
    return whisper_backend.WhisperBackend(
        model_name=config.model,
        device_preference=config.device,
        require_gpu=config.require_gpu,
        vad_aggressiveness=config.vad.aggressiveness,
        vad_min_silence_duration=config.vad.min_silence_duration,
        vad_min_speech_duration=config.vad.min_speech_duration,
        vad_speech_pad_duration=config.vad.speech_pad_duration,
        max_chunk_duration=config.vad.max_chunk_duration,
        enable_partial_transcription=config.partial.enabled,
        partial_model=config.partial.model,
        partial_interval=config.partial.interval,
        max_partial_buffer_seconds=config.partial.max_buffer_seconds,
        language=language,
        compare_transcripts=compare_transcripts,
        session_logger=session_logger,
        min_log_duration=min_log_duration,
        ca_cert=Path(ca_cert) if ca_cert else None,
        disable_ssl_verify=disable_ssl_verify,
        temp_file=Path(config.debug_output_dir) if config.debug_output_dir else None,
    )


def create_realtime_backend(
    *,
    config: backend_config.RealtimeConfig,
    language: str,
    compare_transcripts: bool,
    min_log_duration: float,
    disable_ssl_verify: bool,
    session_logger: session_logger.SessionLogger | None,
) -> realtime_backend.RealtimeBackend:
    """
    Create and configure a Realtime backend from a configuration object.

    Args:
        config: RealtimeConfig with backend-specific settings
        language: Language code for transcription
        compare_transcripts: Whether to compare chunked vs full audio transcription
        min_log_duration: Minimum session duration for logging
        disable_ssl_verify: Whether to disable SSL verification
        session_logger: Session logger for persistence

    Returns:
        Configured RealtimeBackend instance

    Raises:
        ValueError: If API key is not provided
    """
    if not config.api_key:
        raise ValueError(
            "OpenAI API key required for realtime transcription. "
            "Provide --config.realtime.api_key or set OPENAI_API_KEY environment variable."
        )

    return realtime_backend.RealtimeBackend(
        api_key=config.api_key,
        endpoint=config.endpoint,
        model=config.model,
        instructions=config.instructions,
        vad_threshold=config.turn_detection.threshold,
        vad_silence_duration_ms=config.turn_detection.silence_duration_ms,
        debug=config.debug,
        language=language,
        compare_transcripts=compare_transcripts,
        session_logger=session_logger,
        min_log_duration=min_log_duration,
        disable_ssl_verify=disable_ssl_verify,
    )
