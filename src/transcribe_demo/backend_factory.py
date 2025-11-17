from __future__ import annotations

from pathlib import Path

from absl import flags

import transcribe_demo.realtime_backend
import transcribe_demo.session_logger
import transcribe_demo.whisper_backend

FLAGS = flags.FLAGS


def create_whisper_backend(*, language: str, session_logger: transcribe_demo.session_logger.SessionLogger | None) -> transcribe_demo.whisper_backend.WhisperBackend:
    """
    Create and configure a Whisper backend from FLAGS.

    Args:
        language: Language preference for transcription
        session_logger: Session logger for persistence

    Returns:
        Configured WhisperBackend instance
    """
    return transcribe_demo.whisper_backend.WhisperBackend(
        model_name=FLAGS.model or "turbo",
        device_preference=FLAGS.device or "auto",
        require_gpu=FLAGS.require_gpu,
        vad_aggressiveness=FLAGS.vad_aggressiveness,
        vad_min_silence_duration=FLAGS.vad_min_silence_duration,
        vad_min_speech_duration=FLAGS.vad_min_speech_duration,
        vad_speech_pad_duration=FLAGS.vad_speech_pad_duration,
        max_chunk_duration=FLAGS.max_chunk_duration,
        enable_partial_transcription=FLAGS.enable_partial_transcription,
        partial_model=FLAGS.partial_model,
        partial_interval=FLAGS.partial_interval,
        max_partial_buffer_seconds=FLAGS.max_partial_buffer_seconds,
        language=language,
        compare_transcripts=FLAGS.compare_transcripts,
        session_logger=session_logger,
        min_log_duration=FLAGS.min_log_duration,
        ca_cert=Path(FLAGS.ca_cert) if FLAGS.ca_cert else None,
        disable_ssl_verify=FLAGS.disable_ssl_verify,
        temp_file=Path(FLAGS.temp_file) if FLAGS.temp_file else None,
    )


def create_realtime_backend(
    *,
    api_key: str | None,
    language: str,
    session_logger: transcribe_demo.session_logger.SessionLogger | None,
) -> transcribe_demo.realtime_backend.RealtimeBackend:
    """
    Create and configure a Realtime backend from FLAGS.

    Args:
        api_key: OpenAI API key for authentication
        language: Language preference for transcription
        session_logger: Session logger for persistence

    Returns:
        Configured RealtimeBackend instance

    Raises:
        RuntimeError: If API key is not provided
    """
    if not api_key:
        raise RuntimeError(
            "OpenAI API key required for realtime transcription. Provide --api-key or set OPENAI_API_KEY."
        )

    return transcribe_demo.realtime_backend.RealtimeBackend(
        api_key=api_key,
        endpoint=FLAGS.realtime_endpoint,
        model=FLAGS.realtime_model,
        instructions=FLAGS.realtime_instructions,
        vad_threshold=FLAGS.realtime_vad_threshold,
        vad_silence_duration_ms=FLAGS.realtime_vad_silence_duration_ms,
        debug=FLAGS.realtime_debug,
        language=language,
        compare_transcripts=FLAGS.compare_transcripts,
        session_logger=session_logger,
        min_log_duration=FLAGS.min_log_duration,
        disable_ssl_verify=FLAGS.disable_ssl_verify,
    )
