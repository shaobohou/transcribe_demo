"""CLI for session replay utility."""

from __future__ import annotations

import dataclasses
import sys
from typing import Literal

from simple_parsing import ArgumentParser

from transcribe_demo.session_replay import (
    list_sessions,
    load_session,
    print_session_details,
    print_session_list,
    remove_incomplete_sessions,
    retranscribe_session,
)


@dataclasses.dataclass(frozen=True, kw_only=True)
class CommonConfig:
    """Common configuration for all subcommands."""

    session_log_dir: str = "./session_logs"
    """Directory containing session logs."""


@dataclasses.dataclass(frozen=True, kw_only=True)
class ListConfig(CommonConfig):
    """Configuration for list command."""

    backend: Literal["whisper", "realtime"] | None = None
    """Filter sessions by backend."""

    start_date: str | None = None
    """Filter sessions on or after this date (YYYY-MM-DD format)."""

    end_date: str | None = None
    """Filter sessions on or before this date (YYYY-MM-DD format)."""

    min_duration: float | None = None
    """Filter sessions with duration >= this value in seconds."""

    verbose: bool = False
    """Show detailed information."""

    include_incomplete: bool = False
    """Include incomplete sessions (sessions without .complete marker)."""


@dataclasses.dataclass(frozen=True, kw_only=True)
class ShowConfig(CommonConfig):
    """Configuration for show command."""

    session_path: str | None = None
    """Path to session directory."""

    allow_incomplete: bool = False
    """Allow loading incomplete sessions."""


@dataclasses.dataclass(frozen=True, kw_only=True)
class RetranscribeConfig(CommonConfig):
    """Configuration for retranscribe command."""

    session_path: str | None = None
    """Path to session directory."""

    allow_incomplete: bool = False
    """Allow loading incomplete sessions."""

    retranscribe_backend: Literal["whisper", "realtime"] = "whisper"
    """Backend to use for retranscription."""

    output_dir: str = "./session_logs"
    """Output directory for retranscription results."""

    # Whisper-specific
    model: str = "turbo"
    """Whisper model to use for retranscription."""

    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    """Device to use for Whisper retranscription."""

    language: str = "en"
    """Language for retranscription."""

    vad_aggressiveness: int = 2
    """VAD aggressiveness level (0-3) for Whisper retranscription."""

    vad_min_silence_duration: float = 0.2
    """Minimum silence duration (seconds) for VAD."""

    # Realtime-specific
    api_key: str | None = None
    """OpenAI API key for realtime backend (defaults to OPENAI_API_KEY env var)."""

    realtime_model: str = "gpt-realtime-mini"
    """Realtime model to use."""

    audio_format: Literal["wav", "flac"] = "flac"
    """Audio format for saved files."""


@dataclasses.dataclass(frozen=True, kw_only=True)
class RemoveIncompleteConfig(CommonConfig):
    """Configuration for remove-incomplete command."""

    backend: Literal["whisper", "realtime"] | None = None
    """Filter sessions by backend."""

    start_date: str | None = None
    """Filter sessions on or after this date (YYYY-MM-DD format)."""

    end_date: str | None = None
    """Filter sessions on or before this date (YYYY-MM-DD format)."""

    min_duration: float | None = None
    """Filter sessions with duration >= this value in seconds."""

    dry_run: bool = False
    """Dry run mode - show what would be removed without actually removing."""


def list_command(config: ListConfig) -> None:
    """Execute list command."""
    sessions = list_sessions(
        log_dir=config.session_log_dir,
        backend=config.backend,
        start_date=config.start_date,
        end_date=config.end_date,
        min_duration=config.min_duration,
        include_incomplete=config.include_incomplete,
    )
    print_session_list(sessions, verbose=config.verbose)


def show_command(config: ShowConfig) -> None:
    """Execute show command."""
    if not config.session_path:
        print("ERROR: --session_path is required for 'show' command", file=sys.stderr)
        sys.exit(1)

    try:
        loaded = load_session(config.session_path, allow_incomplete=config.allow_incomplete)
        print_session_details(loaded)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


def retranscribe_command(config: RetranscribeConfig) -> None:
    """Execute retranscribe command."""
    if not config.session_path:
        print("ERROR: --session_path is required for 'retranscribe' command", file=sys.stderr)
        sys.exit(1)

    try:
        # Load session
        loaded = load_session(config.session_path, allow_incomplete=config.allow_incomplete)

        # Build backend kwargs
        backend_kwargs: dict[str, str | int | float | None] = {
            "audio_format": config.audio_format,
            "language": config.language,
        }

        if config.retranscribe_backend == "whisper":
            backend_kwargs["model"] = config.model
            backend_kwargs["device"] = config.device
            backend_kwargs["vad_aggressiveness"] = config.vad_aggressiveness
            backend_kwargs["vad_min_silence_duration"] = config.vad_min_silence_duration
        elif config.retranscribe_backend == "realtime":
            backend_kwargs["api_key"] = config.api_key
            backend_kwargs["realtime_model"] = config.realtime_model

        # Retranscribe
        result_path = retranscribe_session(
            loaded_session=loaded,
            output_dir=config.output_dir,
            backend=config.retranscribe_backend,
            backend_kwargs=backend_kwargs,
        )
        print(f"\nRetranscription saved to: {result_path}", file=sys.stdout)

    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


def remove_incomplete_command(config: RemoveIncompleteConfig) -> None:
    """Execute remove-incomplete command."""
    removed_paths = remove_incomplete_sessions(
        log_dir=config.session_log_dir,
        backend=config.backend,
        start_date=config.start_date,
        end_date=config.end_date,
        min_duration=config.min_duration,
        dry_run=config.dry_run,
    )

    if removed_paths and not config.dry_run:
        print(f"\nSuccessfully removed {len(removed_paths)} session(s).", file=sys.stdout)
    elif removed_paths and config.dry_run:
        print(f"\n[DRY RUN] Would remove {len(removed_paths)} session(s).", file=sys.stdout)


def cli_main() -> None:
    """Entry point for the CLI (called by pyproject.toml console_scripts)."""
    parser = ArgumentParser(
        prog="transcribe-session",
        description="Session replay and management utility for transcribe-demo",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # List subcommand
    list_parser = subparsers.add_parser("list", help="List sessions")
    list_parser.add_arguments(ListConfig, dest="config")

    # Show subcommand
    show_parser = subparsers.add_parser("show", help="Show session details")
    show_parser.add_arguments(ShowConfig, dest="config")

    # Retranscribe subcommand
    retranscribe_parser = subparsers.add_parser("retranscribe", help="Retranscribe a session")
    retranscribe_parser.add_arguments(RetranscribeConfig, dest="config")

    # Remove-incomplete subcommand
    remove_parser = subparsers.add_parser("remove-incomplete", help="Remove incomplete sessions")
    remove_parser.add_arguments(RemoveIncompleteConfig, dest="config")

    args = parser.parse_args()

    # Execute appropriate command
    if args.command == "list":
        list_command(args.config)
    elif args.command == "show":
        show_command(args.config)
    elif args.command == "retranscribe":
        retranscribe_command(args.config)
    elif args.command == "remove-incomplete":
        remove_incomplete_command(args.config)
    else:
        print(f"ERROR: Unknown command '{args.command}'", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    cli_main()
