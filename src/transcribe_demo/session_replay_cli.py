"""CLI for session replay utility."""

from __future__ import annotations

import dataclasses
import sys
from typing import Literal

from simple_parsing import ArgumentGenerationMode, ArgumentParser, DashVariant

from transcribe_demo import backend_config
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

    language: str = "en"
    """Language for retranscription."""

    audio_format: Literal["wav", "flac"] = "flac"
    """Audio format for saved files."""

    # Reuse backend configs from main CLI
    whisper: backend_config.WhisperConfig = dataclasses.field(default_factory=backend_config.WhisperConfig)
    """Whisper backend configuration (used when retranscribe_backend='whisper')."""

    realtime: backend_config.RealtimeConfig = dataclasses.field(default_factory=backend_config.RealtimeConfig)
    """Realtime API configuration (used when retranscribe_backend='realtime')."""


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

        # Build backend kwargs from nested configs
        backend_kwargs: dict[str, str | int | float | None] = {
            "audio_format": config.audio_format,
            "language": config.language,
        }

        if config.retranscribe_backend == "whisper":
            backend_kwargs["model"] = config.whisper.model
            backend_kwargs["device"] = config.whisper.device
            backend_kwargs["vad_aggressiveness"] = config.whisper.vad.aggressiveness
            backend_kwargs["vad_min_silence_duration"] = config.whisper.vad.min_silence_duration
        elif config.retranscribe_backend == "realtime":
            backend_kwargs["api_key"] = config.realtime.api_key
            backend_kwargs["realtime_model"] = config.realtime.model

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
    # Create main parser
    parser = ArgumentParser(
        prog="transcribe-session",
        description="Session replay and management utility for transcribe-demo",
        add_option_string_dash_variants=DashVariant.AUTO,  # Preserve underscores, no dash variants
        argument_generation_mode=ArgumentGenerationMode.NESTED,  # Always use full dotted paths
    )

    # Add subparsers for each command
    subparsers = parser.add_subparsers(dest="command", help="Subcommand to execute")

    # List command
    list_parser = subparsers.add_parser("list", help="List all logged sessions")
    list_parser.add_arguments(ListConfig, dest="config")
    list_parser.set_defaults(func=lambda args: list_command(args.config))

    # Show command
    show_parser = subparsers.add_parser("show", help="Show details of a specific session")
    show_parser.add_arguments(ShowConfig, dest="config")
    show_parser.set_defaults(func=lambda args: show_command(args.config))

    # Retranscribe command
    retranscribe_parser = subparsers.add_parser("retranscribe", help="Retranscribe a session with different settings")
    retranscribe_parser.add_arguments(RetranscribeConfig, dest="config")
    retranscribe_parser.set_defaults(func=lambda args: retranscribe_command(args.config))

    # Remove-incomplete command
    remove_parser = subparsers.add_parser(
        "remove-incomplete", help="Remove incomplete sessions (missing .complete marker)"
    )
    remove_parser.add_arguments(RemoveIncompleteConfig, dest="config")
    remove_parser.set_defaults(func=lambda args: remove_incomplete_command(args.config))

    # Parse arguments and dispatch to command handler
    args = parser.parse_args()

    # If no command specified, print help
    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)

    # Execute the command
    args.func(args)


if __name__ == "__main__":
    cli_main()
