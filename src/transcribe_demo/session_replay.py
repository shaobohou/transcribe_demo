"""Utility for listing, loading, and retranscribing logged sessions."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import wave

from transcribe_demo.backend_protocol import TranscriptionChunk
from transcribe_demo.session_logger import SessionLogger, SessionMetadata


@dataclass
class SessionInfo:
    """Summary information about a logged session."""

    session_path: Path
    session_id: str
    timestamp: str
    backend: str
    capture_duration: float
    total_chunks: int
    sample_rate: int
    channels: int
    is_complete: bool = True  # Whether session was successfully finalized


@dataclass
class LoadedSession:
    """A loaded session with audio and metadata."""

    session_path: Path
    metadata: SessionMetadata
    chunks: list[dict[str, Any]]
    audio: np.ndarray
    sample_rate: int


def is_session_complete(session_dir: Path) -> bool:
    """
    Check if a session has been successfully finalized.

    Args:
        session_dir: Path to the session directory

    Returns:
        True if the session has a completion marker, False otherwise
    """
    marker_path = session_dir / ".complete"
    return marker_path.exists()


def list_sessions(
    log_dir: Path | str,
    backend: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    min_duration: float | None = None,
    include_incomplete: bool = False,
) -> list[SessionInfo]:
    """
    List all logged sessions in a directory.

    Args:
        log_dir: Base directory containing session logs
        backend: Filter by backend ("whisper" or "realtime"), None for all
        start_date: Filter sessions on or after this date (YYYY-MM-DD format)
        end_date: Filter sessions on or before this date (YYYY-MM-DD format)
        min_duration: Filter sessions with duration >= this value (seconds)
        include_incomplete: If True, include sessions without completion marker (default: False)

    Returns:
        List of SessionInfo objects, sorted by timestamp (newest first)
    """
    log_path = Path(log_dir)
    if not log_path.exists():
        return []

    sessions: list[SessionInfo] = []

    # Find all session.json files
    for session_json in log_path.rglob("session.json"):
        try:
            with open(session_json, encoding="utf-8") as f:
                data = json.load(f)

            metadata = data.get("metadata", {})

            # Apply filters
            if backend and metadata.get("backend") != backend:
                continue

            session_timestamp = metadata.get("timestamp", "")
            if start_date or end_date:
                try:
                    session_date = datetime.fromisoformat(session_timestamp).date()
                    if start_date:
                        filter_start = datetime.strptime(start_date, "%Y-%m-%d").date()
                        if session_date < filter_start:
                            continue
                    if end_date:
                        filter_end = datetime.strptime(end_date, "%Y-%m-%d").date()
                        if session_date > filter_end:
                            continue
                except (ValueError, TypeError):
                    # Skip sessions with invalid timestamps
                    continue

            duration = metadata.get("capture_duration", 0.0)
            if min_duration is not None and duration < min_duration:
                continue

            # Check if session is complete
            session_dir = session_json.parent
            is_complete = is_session_complete(session_dir)

            # Skip incomplete sessions unless explicitly requested
            if not is_complete and not include_incomplete:
                continue

            # Create SessionInfo
            info = SessionInfo(
                session_path=session_dir,
                session_id=metadata.get("session_id", session_dir.name),
                timestamp=session_timestamp,
                backend=metadata.get("backend", "unknown"),
                capture_duration=duration,
                total_chunks=metadata.get("total_chunks", 0),
                sample_rate=metadata.get("sample_rate", 16000),
                channels=metadata.get("channels", 1),
                is_complete=is_complete,
            )
            sessions.append(info)

        except (json.JSONDecodeError, OSError) as e:
            print(f"WARNING: Failed to read {session_json}: {e}", file=sys.stderr)
            continue

    # Sort by timestamp (newest first)
    sessions.sort(key=lambda s: s.timestamp, reverse=True)
    return sessions


def load_session(session_path: Path | str, allow_incomplete: bool = False) -> LoadedSession:
    """
    Load a session from disk.

    Args:
        session_path: Path to the session directory (containing session.json)
        allow_incomplete: If False, raise error for incomplete sessions (default: False)

    Returns:
        LoadedSession object with metadata, chunks, and audio

    Raises:
        FileNotFoundError: If session.json or audio file doesn't exist
        ValueError: If session data is invalid or incomplete
    """
    session_dir = Path(session_path)
    session_json = session_dir / "session.json"

    if not session_json.exists():
        raise FileNotFoundError(f"Session file not found: {session_json}")

    # Check if session is complete
    if not allow_incomplete and not is_session_complete(session_dir):
        raise ValueError(
            f"Session is incomplete (missing .complete marker): {session_dir}\n"
            f"This may indicate the session was interrupted or not properly finalized.\n"
            f"Use allow_incomplete=True to load anyway."
        )

    # Load session data
    with open(session_json, encoding="utf-8") as f:
        data = json.load(f)

    metadata_dict = data.get("metadata", {})
    chunks = data.get("chunks", [])

    # Convert metadata dict to SessionMetadata
    metadata = SessionMetadata(**metadata_dict)

    # Find and load audio file (try both flac and wav)
    audio_path = None
    for ext in ["flac", "wav"]:
        candidate = session_dir / f"full_audio.{ext}"
        if candidate.exists():
            audio_path = candidate
            break

    if not audio_path:
        raise FileNotFoundError(f"Audio file not found in {session_dir} (tried full_audio.flac and full_audio.wav)")

    # Load audio
    audio, sample_rate = _load_audio(audio_path)

    return LoadedSession(
        session_path=session_dir,
        metadata=metadata,
        chunks=chunks,
        audio=audio,
        sample_rate=sample_rate,
    )


def _load_audio(audio_path: Path) -> tuple[np.ndarray, int]:
    """
    Load audio file (WAV or FLAC).

    Args:
        audio_path: Path to audio file

    Returns:
        Tuple of (audio_array, sample_rate)
    """
    if audio_path.suffix.lower() == ".flac":
        # Use soundfile for FLAC
        audio, sample_rate = sf.read(str(audio_path), dtype="float32")
        return audio, sample_rate
    else:
        # Use wave module for WAV
        with wave.open(str(audio_path), "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            n_frames = wav_file.getnframes()
            audio_bytes = wav_file.readframes(n_frames)
            audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            return audio, sample_rate


def retranscribe_session(
    loaded_session: LoadedSession,
    output_dir: Path | str,
    backend: str,
    backend_kwargs: dict[str, Any] | None = None,
) -> Path:
    """
    Retranscribe a loaded session and save results to a new directory.

    Args:
        loaded_session: LoadedSession object from load_session()
        output_dir: Base directory for saving retranscription results
        backend: Backend to use ("whisper" or "realtime")
        backend_kwargs: Optional backend-specific parameters (model, device, etc.)

    Returns:
        Path to the new session directory

    Raises:
        ValueError: If backend is invalid or required parameters are missing
    """
    import queue as queue_module
    import threading

    from transcribe_demo import audio_capture as audio_capture_lib
    from transcribe_demo.realtime_backend import run_realtime_transcriber
    from transcribe_demo.whisper_backend import run_whisper_transcriber

    if backend_kwargs is None:
        backend_kwargs = {}

    # Validate backend
    if backend not in ("whisper", "realtime"):
        raise ValueError(f"Invalid backend: {backend}. Must be 'whisper' or 'realtime'.")

    # Create a custom session_id for the retranscription
    original_session_id = loaded_session.metadata.session_id
    now = datetime.now()
    time_str = now.strftime("%H%M%S")
    new_session_id = f"retranscribe_{time_str}_{backend}_from_{original_session_id}"

    # Create session logger for the retranscription
    output_path = Path(output_dir)
    session_logger = SessionLogger(
        output_dir=output_path,
        sample_rate=loaded_session.sample_rate,
        channels=loaded_session.metadata.channels,
        backend=backend,
        save_chunk_audio=True,
        session_id=new_session_id,
        audio_format=backend_kwargs.get("audio_format", "flac"),
    )

    # Save the full audio to the new session
    session_logger.save_full_audio(
        audio=loaded_session.audio,
        capture_duration=loaded_session.metadata.capture_duration,
    )

    print(
        f"Retranscribing session '{original_session_id}' with {backend} backend...",
        file=sys.stderr,
    )
    print(f"Original duration: {loaded_session.metadata.capture_duration:.2f}s", file=sys.stderr)
    print(f"Results will be saved to: {session_logger.session_dir}", file=sys.stderr)

    # Create a fake AudioCaptureManager that feeds the loaded audio
    class FakeAudioCaptureManager:
        """Fake AudioCaptureManager that feeds pre-recorded audio."""

        def __init__(
            self,
            sample_rate: int,
            channels: int,
            max_capture_duration: float = 0.0,
            collect_full_audio: bool = True,
        ):
            self.sample_rate = sample_rate
            self.channels = channels
            self.max_capture_duration = max_capture_duration
            self.collect_full_audio = collect_full_audio
            self.audio_queue = queue_module.Queue()
            self.stop_event = threading.Event()
            self.capture_limit_reached = threading.Event()
            self._full_audio_chunks: list[np.ndarray] = []
            self._feeder_thread: threading.Thread | None = None
            self._frame_size = 480  # Standard frame size for Whisper

        def _feed_audio(self) -> None:
            """Feed loaded audio into queue in a background thread."""
            audio = loaded_session.audio
            for start in range(0, len(audio), self._frame_size):
                if self.stop_event.is_set():
                    break

                frame = audio[start : start + self._frame_size]
                if not frame.size:
                    continue

                # Reshape to match expected format (samples, channels)
                frame_shaped = frame.reshape(-1, 1) if self.channels == 1 else frame
                self.audio_queue.put(frame_shaped)

                # Collect for get_full_audio
                if self.collect_full_audio:
                    mono = frame_shaped.mean(axis=1).astype(np.float32) if frame_shaped.ndim > 1 else frame_shaped
                    self._full_audio_chunks.append(mono)

            # Signal end of stream
            self.audio_queue.put(None)
            self.stop_event.set()

        def start(self) -> None:
            """Start feeding audio in background thread."""
            self._feeder_thread = threading.Thread(target=self._feed_audio, daemon=True)
            self._feeder_thread.start()

        def wait_until_stopped(self) -> None:
            """Wait until stop event is set."""
            self.stop_event.wait()

        def stop(self) -> None:
            """Stop audio capture."""
            self.stop_event.set()

        def close(self) -> None:
            """Close the audio capture manager."""
            if self._feeder_thread and self._feeder_thread.is_alive():
                self._feeder_thread.join(timeout=2.0)

        def get_full_audio(self) -> np.ndarray:
            """Get the full audio buffer."""
            if not self._full_audio_chunks:
                return np.zeros(0, dtype=np.float32)
            return np.concatenate(self._full_audio_chunks)

        def get_capture_duration(self) -> float:
            """Get the total capture duration."""
            return loaded_session.metadata.capture_duration

    # Monkey-patch AudioCaptureManager to use our fake implementation
    original_audio_capture = audio_capture_lib.AudioCaptureManager
    audio_capture_lib.AudioCaptureManager = FakeAudioCaptureManager

    try:
        # Create a simple chunk consumer
        from transcribe_demo.chunk_collector import ChunkCollector
        import queue as queue_module
        import threading

        collector = ChunkCollector(stream=sys.stdout)

        if backend == "whisper":
            # Create chunk queue for receiving transcription chunks
            chunk_queue: queue_module.Queue[TranscriptionChunk | None] = queue_module.Queue()
            result_container: list = []

            # Create audio source
            audio_source = FakeAudioCaptureManager(
                sample_rate=loaded_session.sample_rate,
                channels=loaded_session.metadata.channels,
                max_capture_duration=0.0,  # No duration limit for retranscription
                collect_full_audio=backend_kwargs.get("compare_transcripts", False),
            )

            # Run Whisper backend in background thread
            def backend_worker():
                result = run_whisper_transcriber(
                    model_name=backend_kwargs.get("model", "turbo"),
                    sample_rate=loaded_session.sample_rate,
                    channels=loaded_session.metadata.channels,
                    temp_file=backend_kwargs.get("debug_output_dir"),
                    ca_cert=backend_kwargs.get("ca_cert"),
                    disable_ssl_verify=backend_kwargs.get("disable_ssl_verify", False),
                    device_preference=backend_kwargs.get("device", "auto"),
                    require_gpu=backend_kwargs.get("require_gpu", False),
                    audio_source=audio_source,
                    chunk_queue=chunk_queue,
                    vad_aggressiveness=backend_kwargs.get("vad_aggressiveness", 2),
                    vad_min_silence_duration=backend_kwargs.get("vad_min_silence_duration", 0.2),
                    vad_min_speech_duration=backend_kwargs.get("vad_min_speech_duration", 0.25),
                    vad_speech_pad_duration=backend_kwargs.get("vad_speech_pad_duration", 0.2),
                    max_chunk_duration=backend_kwargs.get("max_chunk_duration", 60.0),
                    compare_transcripts=backend_kwargs.get("compare_transcripts", False),
                    language=backend_kwargs.get("language", "en"),
                    session_logger=session_logger,
                    min_log_duration=0.0,  # Always save retranscription
                    enable_partial_transcription=backend_kwargs.get("partial_enabled", False),
                    partial_model=backend_kwargs.get("partial_model", "base.en"),
                    partial_interval=backend_kwargs.get("partial_interval", 1.0),
                    max_partial_buffer_seconds=backend_kwargs.get("partial_max_buffer_seconds", 10.0),
                )
                result_container.append(result)
                chunk_queue.put(None)  # Sentinel

            worker_thread = threading.Thread(target=backend_worker, daemon=True)
            worker_thread.start()

            # Consume chunks from queue
            while True:
                chunk = chunk_queue.get()
                if chunk is None:
                    break
                collector(chunk=chunk)

            # Wait for worker to complete
            worker_thread.join()

            whisper_result = result_container[0] if result_container else None

            # Get final stitched transcription
            final = collector.get_final_stitched_text()

            # Update session logger with cleaned chunk text
            for chunk_index, cleaned_text in collector.get_cleaned_chunks():
                session_logger.update_chunk_cleaned_text(index=chunk_index, cleaned_text=cleaned_text)

            # Finalize session logging
            if whisper_result:
                session_logger.finalize(
                    capture_duration=whisper_result.capture_duration,
                    full_audio_transcription=whisper_result.full_audio_transcription,
                    stitched_transcription=final,
                    extra_metadata=whisper_result.metadata,
                    min_duration=0.0,
                )

        elif backend == "realtime":
            # Run Realtime backend with loaded audio
            import os

            api_key = backend_kwargs.get("api_key") or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OpenAI API key required for realtime backend. "
                    "Provide api_key in backend_kwargs or set OPENAI_API_KEY."
                )

            # Create chunk queue for receiving transcription chunks
            chunk_queue: queue_module.Queue[TranscriptionChunk | None] = queue_module.Queue()
            result_container: list = []

            # Create audio source
            audio_source = FakeAudioCaptureManager(
                sample_rate=loaded_session.sample_rate,
                channels=loaded_session.metadata.channels,
                max_capture_duration=0.0,  # No duration limit for retranscription
                collect_full_audio=backend_kwargs.get("compare_transcripts", False),
            )

            # Run Realtime backend in background thread
            def backend_worker():
                result = run_realtime_transcriber(
                    api_key=api_key,
                    endpoint=backend_kwargs.get("endpoint", "wss://api.openai.com/v1/realtime"),
                    model=backend_kwargs.get("model", "gpt-realtime-mini"),
                    sample_rate=loaded_session.sample_rate,
                    channels=loaded_session.metadata.channels,
                    chunk_duration=2.0,  # Standard 2s chunks for realtime
                    instructions=backend_kwargs.get(
                        "instructions",
                        "You are a high-accuracy transcription service. "
                        "Return a concise verbatim transcript of the most recent audio buffer. "
                        "Do not add commentary or speaker labels.",
                    ),
                    audio_source=audio_source,
                    disable_ssl_verify=backend_kwargs.get("disable_ssl_verify", False),
                    chunk_queue=chunk_queue,
                    compare_transcripts=backend_kwargs.get("compare_transcripts", False),
                    language=backend_kwargs.get("language", "en"),
                    vad_threshold=backend_kwargs.get("turn_detection_threshold", 0.3),
                    vad_silence_duration_ms=backend_kwargs.get("turn_detection_silence_duration_ms", 200),
                    debug=backend_kwargs.get("debug", False),
                )
                result_container.append(result)
                chunk_queue.put(None)  # Sentinel

            worker_thread = threading.Thread(target=backend_worker, daemon=True)
            worker_thread.start()

            # Consume chunks from queue
            while True:
                chunk = chunk_queue.get()
                if chunk is None:
                    break
                collector(chunk=chunk)

            # Wait for worker to complete
            worker_thread.join()

            realtime_result = result_container[0] if result_container else None

            # Get final stitched transcription
            final = collector.get_final_stitched_text()

            # Update session logger with cleaned chunk text
            for chunk_index, cleaned_text in collector.get_cleaned_chunks():
                session_logger.update_chunk_cleaned_text(index=chunk_index, cleaned_text=cleaned_text)

            # Finalize session logging
            if realtime_result:
                session_logger.finalize(
                    capture_duration=realtime_result.capture_duration,
                    full_audio_transcription=None,  # Realtime doesn't compare by default
                    stitched_transcription=final,
                    extra_metadata=realtime_result.metadata,
                    min_duration=0.0,
                )

    finally:
        # Restore original AudioCaptureManager
        audio_capture_lib.AudioCaptureManager = original_audio_capture

    print(f"\nRetranscription complete! Results saved to: {session_logger.session_dir}", file=sys.stderr)
    return session_logger.session_dir


def remove_incomplete_sessions(
    log_dir: Path | str,
    backend: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    min_duration: float | None = None,
    dry_run: bool = False,
) -> list[Path]:
    """
    Remove incomplete sessions (sessions without completion marker).

    Args:
        log_dir: Base directory containing session logs
        backend: Filter by backend ("whisper" or "realtime"), None for all
        start_date: Filter sessions on or after this date (YYYY-MM-DD format)
        end_date: Filter sessions on or before this date (YYYY-MM-DD format)
        min_duration: Filter sessions with duration >= this value (seconds)
        dry_run: If True, only list sessions that would be removed without actually removing them

    Returns:
        List of paths to removed (or would-be-removed if dry_run) session directories
    """
    import shutil

    # Get all incomplete sessions using the existing list_sessions function
    sessions = list_sessions(
        log_dir=log_dir,
        backend=backend,
        start_date=start_date,
        end_date=end_date,
        min_duration=min_duration,
        include_incomplete=True,
    )

    # Filter to only incomplete sessions
    incomplete_sessions = [s for s in sessions if not s.is_complete]

    if not incomplete_sessions:
        print("No incomplete sessions found.", file=sys.stderr)
        return []

    removed_paths: list[Path] = []

    for session in incomplete_sessions:
        session_path = session.session_path
        if dry_run:
            print(f"[DRY RUN] Would remove: {session_path}", file=sys.stderr)
            removed_paths.append(session_path)
        else:
            try:
                shutil.rmtree(session_path)
                print(f"Removed: {session_path}", file=sys.stderr)
                removed_paths.append(session_path)
            except OSError as e:
                print(f"ERROR: Failed to remove {session_path}: {e}", file=sys.stderr)

    if dry_run:
        print(f"\n[DRY RUN] Would remove {len(removed_paths)} incomplete session(s)", file=sys.stderr)
    else:
        print(f"\nRemoved {len(removed_paths)} incomplete session(s)", file=sys.stderr)

    return removed_paths


def print_session_list(sessions: list[SessionInfo], verbose: bool = False) -> None:
    """
    Print a formatted list of sessions.

    Args:
        sessions: List of SessionInfo objects to print
        verbose: If True, print detailed information for each session
    """
    if not sessions:
        print("No sessions found.")
        return

    print(f"Found {len(sessions)} session(s):\n")

    if verbose:
        for i, session in enumerate(sessions, 1):
            print(f"{i}. {session.session_id}")
            print(f"   Path: {session.session_path}")
            print(f"   Timestamp: {session.timestamp}")
            print(f"   Backend: {session.backend}")
            print(f"   Duration: {session.capture_duration:.2f}s")
            print(f"   Chunks: {session.total_chunks}")
            print(f"   Sample Rate: {session.sample_rate} Hz")
            print()
    else:
        # Compact format
        for session in sessions:
            duration_str = f"{session.capture_duration:.1f}s"
            chunks_str = f"{session.total_chunks} chunks"
            status_marker = "" if session.is_complete else " [INCOMPLETE]"
            print(
                f"  {session.session_id:40s} | {session.backend:8s} | {duration_str:8s} | {chunks_str}{status_marker}"
            )


def print_session_details(loaded: LoadedSession) -> None:
    """
    Print detailed information about a loaded session.

    Args:
        loaded: LoadedSession object
    """
    meta = loaded.metadata

    print(f"Session: {meta.session_id}")
    print("=" * 60)
    print(f"Timestamp: {meta.timestamp}")
    print(f"Backend: {meta.backend}")
    print(f"Duration: {meta.capture_duration:.2f}s")
    print(f"Sample Rate: {meta.sample_rate} Hz")
    print(f"Channels: {meta.channels}")
    print(f"Total Chunks: {meta.total_chunks}")

    if meta.model:
        print(f"Model: {meta.model}")
    if meta.device:
        print(f"Device: {meta.device}")
    if meta.language:
        print(f"Language: {meta.language}")

    if meta.vad_aggressiveness is not None:
        print("\nVAD Parameters:")
        print(f"  Aggressiveness: {meta.vad_aggressiveness}")
        if meta.vad_min_silence_duration:
            print(f"  Min Silence: {meta.vad_min_silence_duration}s")
        if meta.vad_min_speech_duration:
            print(f"  Min Speech: {meta.vad_min_speech_duration}s")

    if meta.stitched_transcription:
        print("\n" + "=" * 60)
        print("STITCHED TRANSCRIPTION:")
        print("=" * 60)
        print(meta.stitched_transcription)

    if meta.full_audio_transcription:
        print("\n" + "=" * 60)
        print("FULL AUDIO TRANSCRIPTION:")
        print("=" * 60)
        print(meta.full_audio_transcription)

    if meta.transcription_similarity is not None:
        print("\n" + "=" * 60)
        print("TRANSCRIPTION COMPARISON:")
        print("=" * 60)
        print(f"Similarity: {meta.transcription_similarity:.2%}")
        if meta.transcription_diffs:
            print(f"Differences: {len(meta.transcription_diffs)}")

    print("\n" + "=" * 60)
    print(f"CHUNKS ({len(loaded.chunks)}):")
    print("=" * 60)
    for chunk in loaded.chunks[:5]:  # Show first 5 chunks
        print(
            f"  Chunk {chunk['index']:03d} [{chunk['start_time']:.2f}s - {chunk['end_time']:.2f}s]: "
            f"{chunk['text'][:60]}..."
        )
    if len(loaded.chunks) > 5:
        print(f"  ... and {len(loaded.chunks) - 5} more chunks")
