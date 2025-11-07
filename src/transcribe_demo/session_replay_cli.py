"""CLI for session replay utility."""

from __future__ import annotations

import sys

from absl import app, flags

from transcribe_demo.session_replay import (
    list_sessions,
    load_session,
    print_session_details,
    print_session_list,
    retranscribe_session,
)

FLAGS = flags.FLAGS

# Subcommands
flags.DEFINE_enum(
    "command",
    None,
    ["list", "show", "retranscribe"],
    "Command to execute: list (list sessions), show (show session details), retranscribe (retranscribe a session)",
)
flags.mark_flag_as_required("command")

# Common flags
flags.DEFINE_string(
    "session_log_dir",
    "./session_logs",
    "Directory containing session logs",
)

# List command flags
flags.DEFINE_enum(
    "backend",
    None,
    ["whisper", "realtime"],
    "Filter sessions by backend (for list command)",
)
flags.DEFINE_string(
    "start_date",
    None,
    "Filter sessions on or after this date (YYYY-MM-DD format, for list command)",
)
flags.DEFINE_string(
    "end_date",
    None,
    "Filter sessions on or before this date (YYYY-MM-DD format, for list command)",
)
flags.DEFINE_float(
    "min_duration",
    None,
    "Filter sessions with duration >= this value in seconds (for list command)",
)
flags.DEFINE_boolean(
    "verbose",
    False,
    "Show detailed information (for list command)",
)

# Show/retranscribe command flags
flags.DEFINE_string(
    "session_path",
    None,
    "Path to session directory (for show/retranscribe commands)",
)

# Retranscribe command flags
flags.DEFINE_enum(
    "retranscribe_backend",
    "whisper",
    ["whisper", "realtime"],
    "Backend to use for retranscription",
)
flags.DEFINE_string(
    "output_dir",
    "./session_logs",
    "Output directory for retranscription results",
)
flags.DEFINE_string(
    "model",
    "turbo",
    "Whisper model to use for retranscription",
)
flags.DEFINE_enum(
    "device",
    "auto",
    ["auto", "cpu", "cuda", "mps"],
    "Device to use for Whisper retranscription",
)
flags.DEFINE_string(
    "language",
    "en",
    "Language for retranscription",
)
flags.DEFINE_integer(
    "vad_aggressiveness",
    2,
    "VAD aggressiveness level (0-3) for Whisper retranscription",
)
flags.DEFINE_float(
    "vad_min_silence_duration",
    0.2,
    "Minimum silence duration (seconds) for VAD",
)
flags.DEFINE_string(
    "api_key",
    None,
    "OpenAI API key for realtime backend (defaults to OPENAI_API_KEY env var)",
)
flags.DEFINE_string(
    "realtime_model",
    "gpt-realtime-mini",
    "Realtime model to use",
)
flags.DEFINE_enum(
    "audio_format",
    "flac",
    ["wav", "flac"],
    "Audio format for saved files",
)


def main(argv: list[str]) -> None:
    """Main CLI entry point."""
    if FLAGS.command == "list":
        # List sessions
        sessions = list_sessions(
            log_dir=FLAGS.session_log_dir,
            backend=FLAGS.backend,
            start_date=FLAGS.start_date,
            end_date=FLAGS.end_date,
            min_duration=FLAGS.min_duration,
        )
        print_session_list(sessions, verbose=FLAGS.verbose)

    elif FLAGS.command == "show":
        # Show session details
        if not FLAGS.session_path:
            print("ERROR: --session_path is required for 'show' command", file=sys.stderr)
            sys.exit(1)

        try:
            loaded = load_session(FLAGS.session_path)
            print_session_details(loaded)
        except FileNotFoundError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)

    elif FLAGS.command == "retranscribe":
        # Retranscribe a session
        if not FLAGS.session_path:
            print("ERROR: --session_path is required for 'retranscribe' command", file=sys.stderr)
            sys.exit(1)

        try:
            # Load session
            loaded = load_session(FLAGS.session_path)

            # Build backend kwargs
            backend_kwargs = {
                "audio_format": FLAGS.audio_format,
                "language": FLAGS.language,
            }

            if FLAGS.retranscribe_backend == "whisper":
                backend_kwargs.update(
                    {
                        "model": FLAGS.model,
                        "device": FLAGS.device,
                        "vad_aggressiveness": FLAGS.vad_aggressiveness,
                        "vad_min_silence_duration": FLAGS.vad_min_silence_duration,
                    }
                )
            elif FLAGS.retranscribe_backend == "realtime":
                backend_kwargs.update(
                    {
                        "api_key": FLAGS.api_key,
                        "realtime_model": FLAGS.realtime_model,
                    }
                )

            # Retranscribe
            result_path = retranscribe_session(
                loaded_session=loaded,
                output_dir=FLAGS.output_dir,
                backend=FLAGS.retranscribe_backend,
                backend_kwargs=backend_kwargs,
            )
            print(f"\nRetranscription saved to: {result_path}", file=sys.stdout)

        except (FileNotFoundError, ValueError) as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)

    else:
        print(f"ERROR: Unknown command '{FLAGS.command}'", file=sys.stderr)
        sys.exit(1)


def cli_main() -> None:
    """Entry point for the CLI (called by pyproject.toml console_scripts)."""
    app.run(main)


if __name__ == "__main__":
    app.run(main)
