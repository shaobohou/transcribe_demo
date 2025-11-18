from __future__ import annotations

import dataclasses
import os
import sys
from pathlib import Path

from simple_parsing import ArgumentParser

import transcribe_demo.audio_capture
import transcribe_demo.backend_config
import transcribe_demo.backend_factory
import transcribe_demo.backend_protocol
import transcribe_demo.chunk_collector
import transcribe_demo.file_audio_source
import transcribe_demo.session_logger
import transcribe_demo.transcribe
import transcribe_demo.transcript_diff


def _finalize_transcription_session(
    *,
    collector: transcribe_demo.chunk_collector.ChunkCollector,
    result: transcribe_demo.backend_protocol.TranscriptionResult | None,
    session_logger: transcribe_demo.session_logger.SessionLogger,
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
            similarity, diff_snippets = transcribe_demo.transcript_diff.compute_transcription_diff(stitched_text=final, complete_text=comparison_text)

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
        transcribe_demo.transcript_diff.print_transcription_summary(stream=sys.stdout, final_text=final, complete_audio_text=complete_audio_text)
    else:
        transcribe_demo.transcript_diff.print_final_stitched(stream=sys.stdout, text=final)

    # Print captured duration
    if result is not None:
        print(f"Total captured audio duration: {result.capture_duration:.2f} seconds", file=sys.stderr)


def main(*, config: transcribe_demo.backend_config.CLIConfig) -> None:
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
    session_logger = transcribe_demo.session_logger.SessionLogger(
        output_dir=log_dir,
        sample_rate=config.audio.sample_rate,
        channels=config.audio.channels,
        backend=config.backend,
        save_chunk_audio=True,  # Always save everything
        audio_format=config.session.audio_format,
    )

    # Create chunk collector
    collector = transcribe_demo.chunk_collector.ChunkCollector(stream=sys.stdout)

    # Create audio source
    if config.audio.audio_file:
        audio_source: transcribe_demo.backend_protocol.AudioSource = transcribe_demo.file_audio_source.FileAudioSource(
            audio_file=config.audio.audio_file,
            sample_rate=config.audio.sample_rate,
            channels=config.audio.channels,
            max_capture_duration=config.session.max_capture_duration,
            collect_full_audio=config.session.compare_transcripts or (session_logger is not None),
            playback_speed=config.audio.playback_speed,
        )
    else:
        audio_source = transcribe_demo.audio_capture.AudioCaptureManager(
            sample_rate=config.audio.sample_rate,
            channels=config.audio.channels,
            max_capture_duration=config.session.max_capture_duration,
            collect_full_audio=config.session.compare_transcripts or (session_logger is not None),
        )

    # Get backend configuration
    backend_config = config.get_backend_config()

    # Create backend
    if config.backend == "whisper":
        backend: transcribe_demo.backend_protocol.TranscriptionBackend = (
            transcribe_demo.backend_factory.create_whisper_backend(
                config=backend_config,  # type: ignore[arg-type]
                session_logger=session_logger,
            )
        )
    elif config.backend == "realtime":
        backend = transcribe_demo.backend_factory.create_realtime_backend(
            config=backend_config,  # type: ignore[arg-type]
            session_logger=session_logger,
        )
    else:
        raise ValueError(f"Unknown backend: {config.backend}")

    # Run transcription using the generator
    result: transcribe_demo.backend_protocol.TranscriptionResult | None = None
    try:
        # Create transcription generator
        transcription_gen = transcribe_demo.transcribe.transcribe(backend=backend, audio_source=audio_source)

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
            session_logger=session_logger,
            compare_transcripts=config.session.compare_transcripts,
            min_log_duration=config.session.min_log_duration,
        )


def cli_main() -> None:
    """Entry point for the CLI (called by pyproject.toml console_scripts)."""
    parser = ArgumentParser(
        prog="transcribe-demo",
        description="Real-time audio transcription with Whisper and OpenAI Realtime API",
    )
    parser.add_arguments(transcribe_demo.backend_config.CLIConfig, dest="config")
    args = parser.parse_args()
    config: transcribe_demo.backend_config.CLIConfig = args.config

    # Populate Realtime API key from environment if not provided via CLI
    if config.realtime.api_key is None:
        api_key_from_env = os.getenv("OPENAI_API_KEY")
        if api_key_from_env is not None:
            # Update the realtime config with the API key from environment
            config = dataclasses.replace(
                config,
                realtime=dataclasses.replace(config.realtime, api_key=api_key_from_env),
            )

    main(config=config)


if __name__ == "__main__":
    cli_main()
