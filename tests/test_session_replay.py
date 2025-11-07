"""Tests for session_replay module."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from transcribe_demo.session_logger import SessionLogger
from transcribe_demo.session_replay import (
    SessionInfo,
    list_sessions,
    load_session,
)


@pytest.fixture
def create_test_session(temp_session_dir: Path):
    """Fixture to create a test session."""

    def _create_session(
        session_id: str = "test_session_123456",
        backend: str = "whisper",
        duration: float = 10.0,
        chunks: int = 3,
        audio_format: str = "flac",
    ) -> Path:
        # Create session logger
        logger = SessionLogger(
            output_dir=temp_session_dir,
            sample_rate=16000,
            channels=1,
            backend=backend,
            save_chunk_audio=True,
            session_id=session_id,
            audio_format=audio_format,
        )

        # Log some chunks
        for i in range(chunks):
            start_time = i * (duration / chunks)
            end_time = (i + 1) * (duration / chunks)
            logger.log_chunk(
                index=i,
                text=f"Test chunk {i}.",
                start_time=start_time,
                end_time=end_time,
                inference_seconds=0.1,
                cleaned_text=f"Test chunk {i}",
            )

        # Create fake audio
        num_samples = int(16000 * duration)
        audio = np.random.randn(num_samples).astype(np.float32) * 0.1

        # Save full audio
        logger.save_full_audio(audio, duration)

        # Finalize session
        logger.finalize(
            capture_duration=duration,
            stitched_transcription=" ".join([f"Test chunk {i}" for i in range(chunks)]),
            extra_metadata={
                "model": "turbo",
                "device": "cpu",
                "language": "en",
            },
        )

        return logger.session_dir

    return _create_session


def test_list_sessions_empty(temp_session_dir: Path) -> None:
    """Test listing sessions in an empty directory."""
    sessions = list_sessions(temp_session_dir)
    assert sessions == []


def test_list_sessions_single(temp_session_dir: Path, create_test_session) -> None:
    """Test listing a single session."""
    create_test_session(session_id="session_100000_whisper")
    sessions = list_sessions(temp_session_dir)

    assert len(sessions) == 1
    assert isinstance(sessions[0], SessionInfo)
    assert sessions[0].session_id == "session_100000_whisper"
    assert sessions[0].backend == "whisper"
    assert sessions[0].capture_duration == 10.0
    assert sessions[0].total_chunks == 3


def test_list_sessions_multiple(temp_session_dir: Path, create_test_session) -> None:
    """Test listing multiple sessions."""
    create_test_session(session_id="session_100000_whisper", backend="whisper", duration=5.0)
    create_test_session(session_id="session_110000_whisper", backend="whisper", duration=15.0)
    create_test_session(session_id="session_120000_realtime", backend="realtime", duration=20.0)

    sessions = list_sessions(temp_session_dir)
    assert len(sessions) == 3


def test_list_sessions_filter_by_backend(temp_session_dir: Path, create_test_session) -> None:
    """Test filtering sessions by backend."""
    create_test_session(session_id="session_100000_whisper", backend="whisper")
    create_test_session(session_id="session_110000_realtime", backend="realtime")

    whisper_sessions = list_sessions(temp_session_dir, backend="whisper")
    assert len(whisper_sessions) == 1
    assert whisper_sessions[0].backend == "whisper"

    realtime_sessions = list_sessions(temp_session_dir, backend="realtime")
    assert len(realtime_sessions) == 1
    assert realtime_sessions[0].backend == "realtime"


def test_list_sessions_filter_by_duration(temp_session_dir: Path, create_test_session) -> None:
    """Test filtering sessions by minimum duration."""
    create_test_session(session_id="session_100000_whisper", duration=5.0)
    create_test_session(session_id="session_110000_whisper", duration=15.0)
    create_test_session(session_id="session_120000_whisper", duration=25.0)

    sessions = list_sessions(temp_session_dir, min_duration=10.0)
    assert len(sessions) == 2
    assert all(s.capture_duration >= 10.0 for s in sessions)


def test_list_sessions_sorted_by_timestamp(temp_session_dir: Path, create_test_session) -> None:
    """Test that sessions are sorted by timestamp (newest first)."""
    # Create sessions with different timestamps
    create_test_session(session_id="session_100000_whisper")
    create_test_session(session_id="session_110000_whisper")
    create_test_session(session_id="session_120000_whisper")

    sessions = list_sessions(temp_session_dir)

    # Should be sorted newest first
    # Since session_id contains timestamp, we can check the order
    assert sessions[0].session_id == "session_120000_whisper"
    assert sessions[1].session_id == "session_110000_whisper"
    assert sessions[2].session_id == "session_100000_whisper"


def test_load_session_success(temp_session_dir: Path, create_test_session) -> None:
    """Test loading a session successfully."""
    session_dir = create_test_session(session_id="test_session_123456")

    loaded = load_session(session_dir)

    # Check metadata
    assert loaded.metadata.session_id == "test_session_123456"
    assert loaded.metadata.backend == "whisper"
    assert loaded.metadata.capture_duration == 10.0
    assert loaded.metadata.total_chunks == 3
    assert loaded.metadata.model == "turbo"
    assert loaded.metadata.device == "cpu"
    assert loaded.metadata.language == "en"

    # Check chunks
    assert len(loaded.chunks) == 3
    assert loaded.chunks[0]["text"] == "Test chunk 0."
    assert loaded.chunks[0]["cleaned_text"] == "Test chunk 0"

    # Check audio
    assert loaded.audio.size > 0
    assert loaded.sample_rate == 16000


def test_load_session_missing_json(temp_session_dir: Path) -> None:
    """Test loading a session with missing session.json."""
    fake_dir = temp_session_dir / "nonexistent_session"
    fake_dir.mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="Session file not found"):
        load_session(fake_dir)


def test_load_session_missing_audio(temp_session_dir: Path) -> None:
    """Test loading a session with missing audio file."""
    session_dir = temp_session_dir / "2025-01-01" / "test_session"
    session_dir.mkdir(parents=True)

    # Create session.json without audio file
    session_data = {
        "metadata": {
            "session_id": "test_session",
            "timestamp": "2025-01-01T00:00:00",
            "backend": "whisper",
            "sample_rate": 16000,
            "channels": 1,
            "capture_duration": 10.0,
            "total_chunks": 0,
        },
        "chunks": [],
    }
    with open(session_dir / "session.json", "w") as f:
        json.dump(session_data, f)

    with pytest.raises(FileNotFoundError, match="Audio file not found"):
        load_session(session_dir)


def test_load_session_with_wav_audio(temp_session_dir: Path, create_test_session) -> None:
    """Test loading a session with WAV audio format."""
    session_dir = create_test_session(session_id="test_wav_session", audio_format="wav")

    loaded = load_session(session_dir)

    assert loaded.audio.size > 0
    assert loaded.sample_rate == 16000


def test_load_session_with_flac_audio(temp_session_dir: Path, create_test_session) -> None:
    """Test loading a session with FLAC audio format."""
    session_dir = create_test_session(session_id="test_flac_session", audio_format="flac")

    loaded = load_session(session_dir)

    assert loaded.audio.size > 0
    assert loaded.sample_rate == 16000


def test_list_sessions_nonexistent_directory() -> None:
    """Test listing sessions in a nonexistent directory."""
    sessions = list_sessions("/nonexistent/path/to/sessions")
    assert sessions == []


def test_list_sessions_with_corrupted_json(temp_session_dir: Path, create_test_session) -> None:
    """Test that corrupted session.json files are skipped with a warning."""
    # Create one valid session
    create_test_session(session_id="valid_session")

    # Create a corrupted session
    corrupted_dir = temp_session_dir / "2025-01-01" / "corrupted_session"
    corrupted_dir.mkdir(parents=True)
    with open(corrupted_dir / "session.json", "w") as f:
        f.write("{ invalid json }")

    # Should return only the valid session
    sessions = list_sessions(temp_session_dir)
    assert len(sessions) == 1
    assert sessions[0].session_id == "valid_session"
