"""Factory functions for creating transcription backends from configuration objects.

This module provides clean factory functions that accept configuration dataclasses
instead of directly accessing global FLAGS, improving modularity and testability.
"""

from __future__ import annotations

from transcribe_demo import backend_config, realtime_backend, session_logger, whisper_backend


def create_whisper_backend(
    *,
    config: backend_config.WhisperConfig,
    session_logger: session_logger.SessionLogger | None,
) -> whisper_backend.WhisperBackend:
    """
    Create and configure a Whisper backend from a configuration object.

    Args:
        config: WhisperConfig with all backend settings
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
        language=config.language,
        compare_transcripts=config.compare_transcripts,
        session_logger=session_logger,
        min_log_duration=config.min_log_duration,
        ca_cert=config.ca_cert,
        disable_ssl_verify=config.disable_ssl_verify,
        temp_file=config.temp_file,
    )


def create_realtime_backend(
    *,
    config: backend_config.RealtimeConfig,
    session_logger: session_logger.SessionLogger | None,
) -> realtime_backend.RealtimeBackend:
    """
    Create and configure a Realtime backend from a configuration object.

    Args:
        config: RealtimeConfig with all backend settings
        session_logger: Session logger for persistence

    Returns:
        Configured RealtimeBackend instance

    Raises:
        ValueError: If API key is not provided (checked in config.__post_init__)
    """
    if not config.api_key:
        raise ValueError(
            "OpenAI API key required for realtime transcription. "
            "Provide --realtime.api_key or set OPENAI_API_KEY environment variable."
        )

    return realtime_backend.RealtimeBackend(
        api_key=config.api_key,
        endpoint=config.endpoint,
        model=config.model,
        instructions=config.instructions,
        vad_threshold=config.vad.threshold,
        vad_silence_duration_ms=config.vad.silence_duration_ms,
        debug=config.debug,
        language=config.language,
        compare_transcripts=config.compare_transcripts,
        session_logger=session_logger,
        min_log_duration=config.min_log_duration,
        disable_ssl_verify=config.disable_ssl_verify,
    )
