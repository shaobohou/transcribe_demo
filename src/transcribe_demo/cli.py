from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import cast

from absl import app, flags

from transcribe_demo import (
    audio_capture,
    backend_config,
    backend_factory,
    backend_protocol,
    chunk_collector,
    file_audio_source,
    session_logger,
    transcribe,
    transcript_diff,
)

# =============================================================================
# Flag Definitions
# =============================================================================

# Backend selection
FLAGS = flags.FLAGS
flags.DEFINE_enum("backend", "whisper", ["whisper", "realtime"], "Transcription backend to use.")
flags.DEFINE_string(
    "language",
    "en",
    "Preferred language code for transcription (e.g., en, es). "
    "Use 'auto' to let the model detect. WARNING: 'auto' can cause hallucinations on silence.",
)
flags.DEFINE_bool("refine_with_context", False, "[NOT YET IMPLEMENTED] Use 3-chunk sliding window.")
flags.DEFINE_string("ca_cert", None, "Custom certificate bundle to trust.")
flags.DEFINE_bool(
    "disable_ssl_verify",
    False,
    "Disable SSL certificate verification. WARNING: Insecure, not for production.",
)

# Audio configuration
flags.DEFINE_integer("audio.sample_rate", 16000, "Input sample rate expected by the model.")
flags.DEFINE_integer("audio.channels", 1, "Number of microphone input channels.")
flags.DEFINE_string(
    "audio.audio_file",
    None,
    "Path or URL to audio file for simulating live transcription. If provided, reads from file instead of mic.",
)
flags.DEFINE_float("audio.playback_speed", 1.0, "Playback speed multiplier when using audio_file.")

# Session configuration
flags.DEFINE_string("session.session_log_dir", "./session_logs", "Directory to save session logs.")
flags.DEFINE_enum("session.audio_format", "flac", ["wav", "flac"], "Audio format for saved session files.")
flags.DEFINE_float("session.min_log_duration", 10.0, "Minimum session duration (seconds) required to save logs.")
flags.DEFINE_bool("session.compare_transcripts", True, "Compare chunked vs full-audio transcription at session end.")
flags.DEFINE_float(
    "session.max_capture_duration", 120.0, "Maximum duration (seconds) to run transcription. 0 = unlimited."
)

# Whisper configuration
flags.DEFINE_string("whisper.model", "turbo", "Whisper model name (e.g., 'turbo', 'base.en', 'small').")
flags.DEFINE_string("whisper.device", "auto", "Device to run on: 'auto', 'cpu', 'cuda', or 'mps'.")
flags.DEFINE_bool("whisper.require_gpu", False, "Exit if GPU unavailable instead of falling back to CPU.")
flags.DEFINE_string("whisper.language", "en", "Language code for transcription.")
flags.DEFINE_bool("whisper.compare_transcripts", True, "Whether to compare chunked vs full audio transcription.")
flags.DEFINE_float("whisper.min_log_duration", 10.0, "Minimum session duration for logging.")
flags.DEFINE_string("whisper.ca_cert", None, "Custom certificate bundle for downloading models.")
flags.DEFINE_bool("whisper.disable_ssl_verify", False, "Disable SSL verification (insecure).")
flags.DEFINE_string("whisper.temp_file", None, "Optional path to persist audio chunks for inspection.")

# VAD configuration (Whisper)
flags.DEFINE_integer("vad.aggressiveness", 2, "VAD aggressiveness level (0-3). Higher = more aggressive filtering.")
flags.DEFINE_float("vad.min_silence_duration", 0.2, "Minimum duration of silence (seconds) to trigger chunk split.")
flags.DEFINE_float(
    "vad.min_speech_duration", 0.25, "Minimum duration of speech (seconds) required before transcribing."
)
flags.DEFINE_float(
    "vad.speech_pad_duration", 0.2, "Padding duration (seconds) added before speech to avoid cutting words."
)
flags.DEFINE_float("vad.max_chunk_duration", 60.0, "Maximum chunk duration (seconds). Prevents buffer overflow.")

# Partial transcription configuration
flags.DEFINE_bool("partial.enabled", False, "Whether to enable partial transcription.")
flags.DEFINE_string("partial.model", "base.en", "Model to use for partial transcription (should be fast).")
flags.DEFINE_float("partial.interval", 1.0, "Interval (seconds) between partial transcription updates.")
flags.DEFINE_float("partial.max_buffer_seconds", 10.0, "Segment duration (seconds) for partial transcription.")

# Realtime API configuration
flags.DEFINE_string("realtime.api_key", None, "OpenAI API key. If None, reads from OPENAI_API_KEY env var.")
flags.DEFINE_string("realtime.model", "gpt-realtime-mini", "Realtime model to use.")
flags.DEFINE_string("realtime.endpoint", "wss://api.openai.com/v1/realtime", "Realtime websocket endpoint.")
flags.DEFINE_string(
    "realtime.instructions",
    (
        "You are a high-accuracy transcription service. "
        "Return a concise verbatim transcript of the most recent audio buffer. "
        "Do not add commentary or speaker labels."
    ),
    "Instruction prompt sent to the realtime model.",
)
flags.DEFINE_float("realtime.chunk_duration", 2.0, "Fixed chunk duration for Realtime API (seconds).")
flags.DEFINE_string("realtime.language", "en", "Language code for transcription.")
flags.DEFINE_bool("realtime.compare_transcripts", True, "Whether to compare chunked vs full audio transcription.")
flags.DEFINE_float("realtime.min_log_duration", 10.0, "Minimum session duration for logging.")
flags.DEFINE_bool("realtime.disable_ssl_verify", False, "Disable SSL verification (insecure).")
flags.DEFINE_bool("realtime.debug", False, "Enable debug logging for realtime transcription events.")

# Realtime VAD configuration
flags.DEFINE_float(
    "realtime.vad.threshold", 0.2, "Server VAD threshold for turn detection (0.0-1.0). Lower = more sensitive."
)
flags.DEFINE_integer(
    "realtime.vad.silence_duration_ms", 100, "Silence duration (ms) required to detect turn boundary."
)


# =============================================================================
# Helper Functions
# =============================================================================


def _flags_to_config() -> backend_config.CLIConfig:
    """Convert parsed flags to CLIConfig dataclass."""
    # Build nested configs
    vad_config = backend_config.VADConfig(
        aggressiveness=cast(int, FLAGS["vad.aggressiveness"].value),
        min_silence_duration=cast(float, FLAGS["vad.min_silence_duration"].value),
        min_speech_duration=cast(float, FLAGS["vad.min_speech_duration"].value),
        speech_pad_duration=cast(float, FLAGS["vad.speech_pad_duration"].value),
        max_chunk_duration=cast(float, FLAGS["vad.max_chunk_duration"].value),
    )

    partial_config = backend_config.PartialTranscriptionConfig(
        enabled=cast(bool, FLAGS["partial.enabled"].value),
        model=cast(str, FLAGS["partial.model"].value),
        interval=cast(float, FLAGS["partial.interval"].value),
        max_buffer_seconds=cast(float, FLAGS["partial.max_buffer_seconds"].value),
    )

    whisper_ca_cert_val = FLAGS["whisper.ca_cert"].value
    whisper_temp_file_val = FLAGS["whisper.temp_file"].value

    whisper_config = backend_config.WhisperConfig(
        model=cast(str, FLAGS["whisper.model"].value),
        device=cast(str, FLAGS["whisper.device"].value),
        require_gpu=cast(bool, FLAGS["whisper.require_gpu"].value),
        vad=vad_config,
        partial=partial_config,
        language=cast(str, FLAGS["whisper.language"].value),
        compare_transcripts=cast(bool, FLAGS["whisper.compare_transcripts"].value),
        min_log_duration=cast(float, FLAGS["whisper.min_log_duration"].value),
        ca_cert=Path(whisper_ca_cert_val) if whisper_ca_cert_val else None,
        disable_ssl_verify=cast(bool, FLAGS["whisper.disable_ssl_verify"].value),
        temp_file=Path(whisper_temp_file_val) if whisper_temp_file_val else None,
    )

    realtime_vad_config = backend_config.RealtimeVADConfig(
        threshold=cast(float, FLAGS["realtime.vad.threshold"].value),
        silence_duration_ms=cast(int, FLAGS["realtime.vad.silence_duration_ms"].value),
    )

    # Handle API key from environment if not provided
    api_key = FLAGS["realtime.api_key"].value
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")

    realtime_config = backend_config.RealtimeConfig(
        api_key=api_key,
        model=cast(str, FLAGS["realtime.model"].value),
        endpoint=cast(str, FLAGS["realtime.endpoint"].value),
        instructions=cast(str, FLAGS["realtime.instructions"].value),
        chunk_duration=cast(float, FLAGS["realtime.chunk_duration"].value),
        vad=realtime_vad_config,
        language=cast(str, FLAGS["realtime.language"].value),
        compare_transcripts=cast(bool, FLAGS["realtime.compare_transcripts"].value),
        min_log_duration=cast(float, FLAGS["realtime.min_log_duration"].value),
        disable_ssl_verify=cast(bool, FLAGS["realtime.disable_ssl_verify"].value),
        debug=cast(bool, FLAGS["realtime.debug"].value),
    )

    audio_config = backend_config.AudioConfig(
        sample_rate=cast(int, FLAGS["audio.sample_rate"].value),
        channels=cast(int, FLAGS["audio.channels"].value),
        audio_file=FLAGS["audio.audio_file"].value,
        playback_speed=cast(float, FLAGS["audio.playback_speed"].value),
    )

    session_config = backend_config.SessionConfig(
        session_log_dir=cast(str, FLAGS["session.session_log_dir"].value),
        audio_format=cast(str, FLAGS["session.audio_format"].value),  # type: ignore[arg-type]
        min_log_duration=cast(float, FLAGS["session.min_log_duration"].value),
        compare_transcripts=cast(bool, FLAGS["session.compare_transcripts"].value),
        max_capture_duration=cast(float, FLAGS["session.max_capture_duration"].value),
    )

    return backend_config.CLIConfig(
        backend=cast(str, FLAGS.backend),  # type: ignore[arg-type]
        language=cast(str, FLAGS.language),
        audio=audio_config,
        session=session_config,
        whisper=whisper_config,
        realtime=realtime_config,
        refine_with_context=cast(bool, FLAGS.refine_with_context),
        ca_cert=FLAGS.ca_cert,
        disable_ssl_verify=cast(bool, FLAGS.disable_ssl_verify),
    )


def _finalize_transcription_session(
    *,
    collector: chunk_collector.ChunkCollector,
    result: backend_protocol.TranscriptionResult | None,
    session_logger: session_logger.SessionLogger,
    compare_transcripts: bool,
    min_log_duration: float,
) -> None:
    """
    Finalize transcription session with common cleanup logic.

    This function consolidates the finalization logic shared between Whisper
    and Realtime backends, eliminating code duplication.

    Args:
        collector: Chunk collector with stitched results
        result: Transcription result from backend (TranscriptionResult protocol)
        session_logger: Session logger for persistence
        compare_transcripts: Whether to compare and show diffs
        min_log_duration: Minimum duration for logging
    """
    # Get final stitched result
    final = collector.get_final_stitched_text()

    # Update session logger with cleaned chunk text
    for chunk_index, cleaned_text in collector.get_cleaned_chunks():
        session_logger.update_chunk_cleaned_text(index=chunk_index, cleaned_text=cleaned_text)

    # Compute diff if comparison is enabled
    similarity = None
    diff_snippets = None
    comparison_text = None

    if result is not None:
        # Both backends populate full_audio_transcription in their results
        # The protocol guarantees this property exists
        comparison_text = result.full_audio_transcription

        if comparison_text:
            similarity, diff_snippets = transcript_diff.compute_transcription_diff(
                stitched_text=final, complete_text=comparison_text
            )

    # Finalize session logging
    if result is not None:
        session_logger.finalize(
            capture_duration=result.capture_duration,
            full_audio_transcription=comparison_text,
            stitched_transcription=final,
            extra_metadata=result.metadata,
            min_duration=min_log_duration,
            transcription_similarity=similarity,
            transcription_diffs=diff_snippets,
        )

    # Print results
    if compare_transcripts:
        complete_audio_text = comparison_text or ""
        transcript_diff.print_transcription_summary(
            stream=sys.stdout, final_text=final, complete_audio_text=complete_audio_text
        )
    else:
        transcript_diff.print_final_stitched(stream=sys.stdout, text=final)

    # Print captured duration
    if result is not None:
        print(f"Total captured audio duration: {result.capture_duration:.2f} seconds", file=sys.stderr)


def main(*, config: backend_config.CLIConfig) -> None:
    """
    Main entry point for transcription CLI.

    Args:
        config: Complete CLI configuration from argument parsing
    """
    # Check for unimplemented features
    if config.refine_with_context:
        print(
            "ERROR: --refine-with-context is not yet implemented.\n"
            "This feature will use a 3-chunk sliding window to refine transcriptions with more context.\n"
            "See TODO comments in main.py for implementation details.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Warn about memory usage with unlimited duration and comparison enabled
    if config.session.compare_transcripts and config.session.max_capture_duration == 0:
        print(
            "WARNING: Running with unlimited duration and comparison enabled will "
            "continuously accumulate audio in memory.\n"
            "Consider setting --session.max_capture_duration or use --session.compare_transcripts=false "
            "to reduce memory usage.\n",
            file=sys.stderr,
        )

    # Confirm long capture durations with comparison enabled
    if config.session.compare_transcripts and config.session.max_capture_duration > 300:  # > 5 minutes
        duration_minutes = config.session.max_capture_duration / 60.0
        print(
            f"You have set a capture duration of {duration_minutes:.1f} minutes with comparison enabled.\n"
            f"This will keep audio in memory for the entire session.",
            file=sys.stderr,
        )
        if config.backend == "realtime":
            print(
                "Note: For Realtime API, this will also double your API usage cost.\n",
                file=sys.stderr,
            )

        # Only prompt if stdin is available
        try:
            response = input("Continue? [y/N]: ").strip().lower()
            if response not in ("y", "yes"):
                print("Cancelled.", file=sys.stderr)
                sys.exit(0)
        except (EOFError, OSError):
            # stdin not available (e.g., running in background), proceed without confirmation
            print(
                "(Proceeding without confirmation - stdin not available)",
                file=sys.stderr,
            )

    # Create session logger (always enabled)
    log_dir = Path(config.session.session_log_dir)
    session_logger_obj = session_logger.SessionLogger(
        output_dir=log_dir,
        sample_rate=config.audio.sample_rate,
        channels=config.audio.channels,
        backend=config.backend,
        save_chunk_audio=True,  # Always save everything
        audio_format=config.session.audio_format,
    )

    # Create chunk collector
    collector = chunk_collector.ChunkCollector(stream=sys.stdout)

    # Create audio source
    if config.audio.audio_file:
        audio_source: backend_protocol.AudioSource = file_audio_source.FileAudioSource(
            audio_file=config.audio.audio_file,
            sample_rate=config.audio.sample_rate,
            channels=config.audio.channels,
            max_capture_duration=config.session.max_capture_duration,
            collect_full_audio=config.session.compare_transcripts or (session_logger_obj is not None),
            playback_speed=config.audio.playback_speed,
        )
    else:
        audio_source = audio_capture.AudioCaptureManager(
            sample_rate=config.audio.sample_rate,
            channels=config.audio.channels,
            max_capture_duration=config.session.max_capture_duration,
            collect_full_audio=config.session.compare_transcripts or (session_logger_obj is not None),
        )

    # Get backend configuration
    backend_cfg = config.get_backend_config()

    # Create backend
    if config.backend == "whisper":
        backend: backend_protocol.TranscriptionBackend = backend_factory.create_whisper_backend(
            config=backend_cfg,  # type: ignore[arg-type]
            session_logger=session_logger_obj,
        )
    elif config.backend == "realtime":
        backend = backend_factory.create_realtime_backend(
            config=backend_cfg,  # type: ignore[arg-type]
            session_logger=session_logger_obj,
        )
    else:
        raise ValueError(f"Unknown backend: {config.backend}")

    # Run transcription using the generator
    result: backend_protocol.TranscriptionResult | None = None
    try:
        # Create transcription generator
        transcription_gen = transcribe.transcribe(backend=backend, audio_source=audio_source)

        # Consume chunks and collect result
        # Must use manual iteration to capture the generator's return value
        try:
            while True:
                chunk = next(transcription_gen)
                collector(chunk=chunk)
        except StopIteration as e:
            result = e.value

    except KeyboardInterrupt:
        pass
    finally:
        # Use common finalization logic
        _finalize_transcription_session(
            collector=collector,
            result=result,
            session_logger=session_logger_obj,
            compare_transcripts=config.session.compare_transcripts,
            min_log_duration=config.session.min_log_duration,
        )


def cli_main(argv):
    """Entry point for the CLI (called by absl.app.run)."""
    # Parse flags
    FLAGS(argv)  # This parses the command-line flags
    config = _flags_to_config()
    main(config=config)


def run():
    """Wrapper for console_scripts entry point."""
    app.run(cli_main)


if __name__ == "__main__":
    app.run(cli_main)
