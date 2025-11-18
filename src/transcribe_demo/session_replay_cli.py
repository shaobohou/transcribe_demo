"""CLI for session replay utility."""

from __future__ import annotations

import argparse
import dataclasses
import sys
from typing import Literal

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
    parser = argparse.ArgumentParser(
        prog="transcribe-session",
        description="Session replay and management utility for transcribe-demo",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # List subcommand
    list_parser = subparsers.add_parser("list", help="List sessions")
    list_parser.add_argument("--session_log_dir", default="./session_logs", help="Directory containing session logs")
    list_parser.add_argument("--backend", choices=["whisper", "realtime"], help="Filter sessions by backend")
    list_parser.add_argument("--start_date", help="Filter sessions on or after this date (YYYY-MM-DD)")
    list_parser.add_argument("--end_date", help="Filter sessions on or before this date (YYYY-MM-DD)")
    list_parser.add_argument("--min_duration", type=float, help="Filter sessions with duration >= this value (seconds)")
    list_parser.add_argument("--verbose", action="store_true", help="Show detailed information")
    list_parser.add_argument("--include_incomplete", action="store_true", help="Include incomplete sessions")

    # Show subcommand
    show_parser = subparsers.add_parser("show", help="Show session details")
    show_parser.add_argument("--session_log_dir", default="./session_logs", help="Directory containing session logs")
    show_parser.add_argument("--session_path", required=True, help="Path to session directory")
    show_parser.add_argument("--allow_incomplete", action="store_true", help="Allow loading incomplete sessions")

    # Retranscribe subcommand
    retranscribe_parser = subparsers.add_parser("retranscribe", help="Retranscribe a session")
    retranscribe_parser.add_argument(
        "--session_log_dir", default="./session_logs", help="Directory containing session logs"
    )
    retranscribe_parser.add_argument("--session_path", required=True, help="Path to session directory")
    retranscribe_parser.add_argument(
        "--allow_incomplete", action="store_true", help="Allow loading incomplete sessions"
    )
    retranscribe_parser.add_argument(
        "--retranscribe_backend", choices=["whisper", "realtime"], default="whisper", help="Backend to use"
    )
    retranscribe_parser.add_argument("--output_dir", default="./session_logs", help="Output directory for results")
    retranscribe_parser.add_argument("--model", default="turbo", help="Whisper model to use")
    retranscribe_parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"], help="Device")
    retranscribe_parser.add_argument("--language", default="en", help="Language for retranscription")
    retranscribe_parser.add_argument("--vad_aggressiveness", type=int, default=2, help="VAD aggressiveness (0-3)")
    retranscribe_parser.add_argument("--vad_min_silence_duration", type=float, default=0.2, help="Min silence duration")
    retranscribe_parser.add_argument("--api_key", help="OpenAI API key (defaults to OPENAI_API_KEY env var)")
    retranscribe_parser.add_argument("--realtime_model", default="gpt-realtime-mini", help="Realtime model to use")
    retranscribe_parser.add_argument("--audio_format", choices=["wav", "flac"], default="flac", help="Audio format")

    # Remove-incomplete subcommand
    remove_parser = subparsers.add_parser("remove-incomplete", help="Remove incomplete sessions")
    remove_parser.add_argument("--session_log_dir", default="./session_logs", help="Directory containing session logs")
    remove_parser.add_argument("--backend", choices=["whisper", "realtime"], help="Filter sessions by backend")
    remove_parser.add_argument("--start_date", help="Filter sessions on or after this date (YYYY-MM-DD)")
    remove_parser.add_argument("--end_date", help="Filter sessions on or before this date (YYYY-MM-DD)")
    remove_parser.add_argument(
        "--min_duration", type=float, help="Filter sessions with duration >= this value (seconds)"
    )
    remove_parser.add_argument("--dry_run", action="store_true", help="Dry run - show what would be removed")

    args = parser.parse_args()

    # Build config objects from args
    if args.command == "list":
        config = ListConfig(
            session_log_dir=args.session_log_dir,
            backend=args.backend,  # type: ignore[arg-type]
            start_date=args.start_date,
            end_date=args.end_date,
            min_duration=args.min_duration,
            verbose=args.verbose,
            include_incomplete=args.include_incomplete,
        )
        list_command(config)
    elif args.command == "show":
        config = ShowConfig(
            session_log_dir=args.session_log_dir,
            session_path=args.session_path,
            allow_incomplete=args.allow_incomplete,
        )
        show_command(config)
    elif args.command == "retranscribe":
        config = RetranscribeConfig(
            session_log_dir=args.session_log_dir,
            session_path=args.session_path,
            allow_incomplete=args.allow_incomplete,
            retranscribe_backend=args.retranscribe_backend,  # type: ignore[arg-type]
            output_dir=args.output_dir,
            model=args.model,
            device=args.device,  # type: ignore[arg-type]
            language=args.language,
            vad_aggressiveness=args.vad_aggressiveness,
            vad_min_silence_duration=args.vad_min_silence_duration,
            api_key=args.api_key,
            realtime_model=args.realtime_model,
            audio_format=args.audio_format,  # type: ignore[arg-type]
        )
        retranscribe_command(config)
    elif args.command == "remove-incomplete":
        config = RemoveIncompleteConfig(
            session_log_dir=args.session_log_dir,
            backend=args.backend,  # type: ignore[arg-type]
            start_date=args.start_date,
            end_date=args.end_date,
            min_duration=args.min_duration,
            dry_run=args.dry_run,
        )
        remove_incomplete_command(config)
    else:
        print(f"ERROR: Unknown command '{args.command}'", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    cli_main()
