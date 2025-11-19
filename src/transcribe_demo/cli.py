from __future__ import annotations

import dataclasses
import os
import sys
from pathlib import Path

from simple_parsing import ArgumentGenerationMode, ArgumentParser, DashVariant
from simple_parsing.wrappers.field_wrapper import NestedMode

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
    if config.session.compare_transcripts and config.max_capture_duration == 0:
        print(
            "WARNING: Running with unlimited duration and comparison enabled will "
            "continuously accumulate audio in memory.\n"
            "Consider setting --max_capture_duration or use --session.compare_transcripts=false "
            "to reduce memory usage.\n",
            file=sys.stderr,
        )

    # Confirm long capture durations with comparison enabled
    if config.session.compare_transcripts and config.max_capture_duration > 300:  # > 5 minutes
        duration_minutes = config.max_capture_duration / 60.0
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
            max_capture_duration=config.max_capture_duration,
            collect_full_audio=config.session.compare_transcripts or (session_logger_obj is not None),
            playback_speed=config.audio.playback_speed,
        )
    else:
        audio_source = audio_capture.AudioCaptureManager(
            sample_rate=config.audio.sample_rate,
            channels=config.audio.channels,
            max_capture_duration=config.max_capture_duration,
            collect_full_audio=config.session.compare_transcripts or (session_logger_obj is not None),
        )

    # Get backend configuration
    backend_cfg = config.get_backend_config()

    # Create backend
    if config.backend == "whisper":
        backend: backend_protocol.TranscriptionBackend = backend_factory.create_whisper_backend(
            config=backend_cfg,  # type: ignore[arg-type]
            language=config.language,
            compare_transcripts=config.session.compare_transcripts,
            min_log_duration=config.session.min_log_duration,
            ca_cert=config.ca_cert,
            disable_ssl_verify=config.disable_ssl_verify,
            session_logger=session_logger_obj,
        )
    elif config.backend == "realtime":
        backend = backend_factory.create_realtime_backend(
            config=backend_cfg,  # type: ignore[arg-type]
            language=config.language,
            compare_transcripts=config.session.compare_transcripts,
            min_log_duration=config.session.min_log_duration,
            disable_ssl_verify=config.disable_ssl_verify,
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


def run():
    """Main entry point for the CLI."""
    # Parse arguments using simple-parsing
    parser = ArgumentParser(
        prog="transcribe-demo",
        description="Real-time speech transcription with Whisper and OpenAI Realtime API",
        add_option_string_dash_variants=DashVariant.AUTO,  # Preserve underscores, no dash variants
        argument_generation_mode=ArgumentGenerationMode.NESTED,  # Always use full dotted paths
        nested_mode=NestedMode.WITHOUT_ROOT,  # Remove "config." prefix from flags
    )
    parser.add_arguments(backend_config.CLIConfig, dest="config")
    args = parser.parse_args()
    config: backend_config.CLIConfig = args.config

    # Handle environment variable fallback for API key
    # If api_key is None, try to read from OPENAI_API_KEY env var
    if config.realtime.api_key is None:
        env_api_key = os.getenv("OPENAI_API_KEY")
        if env_api_key:
            # Create updated realtime config with env API key
            config = dataclasses.replace(
                config,
                realtime=dataclasses.replace(config.realtime, api_key=env_api_key),
            )

    main(config=config)


if __name__ == "__main__":
    run()
