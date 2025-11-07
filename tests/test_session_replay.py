"""Tests for session_replay module."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from transcribe_demo.session_logger import SessionLogger
from transcribe_demo.session_replay import (
    SessionInfo,
    is_session_complete,
    list_sessions,
    load_session,
    remove_incomplete_sessions,
    retranscribe_session,
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

    # Add completion marker so we get past the completeness check
    (session_dir / ".complete").touch()

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


def test_is_session_complete(temp_session_dir: Path, create_test_session) -> None:
    """Test checking if a session is complete."""
    session_dir = create_test_session(session_id="complete_session")

    # Session created by fixture should be complete (has .complete marker)
    assert is_session_complete(session_dir)

    # Create incomplete session (without marker)
    incomplete_dir = temp_session_dir / "2025-01-01" / "incomplete_session"
    incomplete_dir.mkdir(parents=True)
    session_data = {
        "metadata": {
            "session_id": "incomplete_session",
            "timestamp": "2025-01-01T00:00:00",
            "backend": "whisper",
            "sample_rate": 16000,
            "channels": 1,
            "capture_duration": 10.0,
            "total_chunks": 0,
        },
        "chunks": [],
    }
    with open(incomplete_dir / "session.json", "w") as f:
        json.dump(session_data, f)

    # Should be incomplete (no .complete marker)
    assert not is_session_complete(incomplete_dir)


def test_list_sessions_excludes_incomplete_by_default(temp_session_dir: Path, create_test_session) -> None:
    """Test that incomplete sessions are excluded by default."""
    # Create complete session
    create_test_session(session_id="complete_session")

    # Create incomplete session (without .complete marker)
    incomplete_dir = temp_session_dir / "2025-01-01" / "incomplete_session"
    incomplete_dir.mkdir(parents=True)
    session_data = {
        "metadata": {
            "session_id": "incomplete_session",
            "timestamp": "2025-01-01T00:00:00",
            "backend": "whisper",
            "sample_rate": 16000,
            "channels": 1,
            "capture_duration": 10.0,
            "total_chunks": 0,
        },
        "chunks": [],
    }
    with open(incomplete_dir / "session.json", "w") as f:
        json.dump(session_data, f)

    # By default, should only include complete sessions
    sessions = list_sessions(temp_session_dir)
    assert len(sessions) == 1
    assert sessions[0].session_id == "complete_session"
    assert sessions[0].is_complete


def test_list_sessions_includes_incomplete_when_requested(temp_session_dir: Path, create_test_session) -> None:
    """Test that incomplete sessions can be included with flag."""
    # Create complete session
    create_test_session(session_id="complete_session")

    # Create incomplete session (without .complete marker)
    incomplete_dir = temp_session_dir / "2025-01-01" / "incomplete_session"
    incomplete_dir.mkdir(parents=True)
    session_data = {
        "metadata": {
            "session_id": "incomplete_session",
            "timestamp": "2025-01-01T00:00:00",
            "backend": "whisper",
            "sample_rate": 16000,
            "channels": 1,
            "capture_duration": 10.0,
            "total_chunks": 0,
        },
        "chunks": [],
    }
    with open(incomplete_dir / "session.json", "w") as f:
        json.dump(session_data, f)

    # With include_incomplete=True, should include both
    sessions = list_sessions(temp_session_dir, include_incomplete=True)
    assert len(sessions) == 2

    # Check that is_complete flag is set correctly
    complete_sessions = [s for s in sessions if s.is_complete]
    incomplete_sessions = [s for s in sessions if not s.is_complete]
    assert len(complete_sessions) == 1
    assert len(incomplete_sessions) == 1
    assert complete_sessions[0].session_id == "complete_session"
    assert incomplete_sessions[0].session_id == "incomplete_session"


def test_load_session_rejects_incomplete(temp_session_dir: Path) -> None:
    """Test that loading incomplete session raises error by default."""
    # Create incomplete session (without .complete marker)
    incomplete_dir = temp_session_dir / "2025-01-01" / "incomplete_session"
    incomplete_dir.mkdir(parents=True)

    # Create valid session.json and audio
    session_data = {
        "metadata": {
            "session_id": "incomplete_session",
            "timestamp": "2025-01-01T00:00:00",
            "backend": "whisper",
            "sample_rate": 16000,
            "channels": 1,
            "capture_duration": 5.0,
            "total_chunks": 0,
        },
        "chunks": [],
    }
    with open(incomplete_dir / "session.json", "w") as f:
        json.dump(session_data, f)

    # Create fake audio
    audio = np.random.randn(16000 * 5).astype(np.float32) * 0.1
    import soundfile as sf

    sf.write(str(incomplete_dir / "full_audio.flac"), audio, 16000, format="FLAC", subtype="PCM_16")

    # Should raise ValueError for incomplete session
    with pytest.raises(ValueError, match="Session is incomplete"):
        load_session(incomplete_dir)


def test_load_session_allows_incomplete_with_flag(temp_session_dir: Path) -> None:
    """Test that incomplete sessions can be loaded with allow_incomplete=True."""
    # Create incomplete session (without .complete marker)
    incomplete_dir = temp_session_dir / "2025-01-01" / "incomplete_session"
    incomplete_dir.mkdir(parents=True)

    # Create valid session.json and audio
    session_data = {
        "metadata": {
            "session_id": "incomplete_session",
            "timestamp": "2025-01-01T00:00:00",
            "backend": "whisper",
            "sample_rate": 16000,
            "channels": 1,
            "capture_duration": 5.0,
            "total_chunks": 0,
        },
        "chunks": [],
    }
    with open(incomplete_dir / "session.json", "w") as f:
        json.dump(session_data, f)

    # Create fake audio
    audio = np.random.randn(16000 * 5).astype(np.float32) * 0.1
    import soundfile as sf

    sf.write(str(incomplete_dir / "full_audio.flac"), audio, 16000, format="FLAC", subtype="PCM_16")

    # Should succeed with allow_incomplete=True
    loaded = load_session(incomplete_dir, allow_incomplete=True)
    assert loaded.metadata.session_id == "incomplete_session"
    assert loaded.audio.size > 0


def test_retranscribe_session_whisper(temp_session_dir: Path, create_test_session, monkeypatch) -> None:
    """Test retranscribing a session with Whisper backend."""
    # Create a test session
    session_dir = create_test_session(session_id="original_session", duration=5.0, chunks=2)

    # Load the session
    loaded = load_session(session_dir)

    # Mock the backend runner to avoid actually running Whisper
    from dataclasses import dataclass

    @dataclass
    class FakeWhisperResult:
        capture_duration: float
        full_audio_transcription: str | None
        metadata: dict

    called_with = {}

    def fake_whisper_runner(**kwargs):
        called_with.update(kwargs)
        # Call the chunk consumer with some fake chunks
        if "chunk_consumer" in kwargs:
            consumer = kwargs["chunk_consumer"]
            consumer(0, "Retranscribed chunk 0.", 0.0, 2.5, 0.1)
            consumer(1, "Retranscribed chunk 1.", 2.5, 5.0, 0.1)
        # Also log chunks if session_logger is provided
        if "session_logger" in kwargs:
            session_logger = kwargs["session_logger"]
            session_logger.log_chunk(0, "Retranscribed chunk 0.", 0.0, 2.5, 0.1)
            session_logger.log_chunk(1, "Retranscribed chunk 1.", 2.5, 5.0, 0.1)
        return FakeWhisperResult(
            capture_duration=5.0,
            full_audio_transcription=None,
            metadata={"model": "turbo", "device": "cpu"},
        )

    monkeypatch.setattr("transcribe_demo.whisper_backend.run_whisper_transcriber", fake_whisper_runner)

    # Retranscribe
    output_dir = temp_session_dir / "retranscriptions"
    result_path = retranscribe_session(
        loaded_session=loaded,
        output_dir=output_dir,
        backend="whisper",
        backend_kwargs={"model": "small", "vad_aggressiveness": 3},
    )

    # Verify the backend was called
    assert "model_name" in called_with
    assert called_with["model_name"] == "small"
    assert called_with["vad_aggressiveness"] == 3
    assert called_with["sample_rate"] == 16000

    # Verify the new session was created
    assert result_path.exists()
    assert (result_path / "session.json").exists()
    assert (result_path / ".complete").exists()

    # Load and verify the retranscribed session
    retranscribed = load_session(result_path)
    assert "retranscribe_" in retranscribed.metadata.session_id
    assert "from_original_session" in retranscribed.metadata.session_id
    assert retranscribed.metadata.backend == "whisper"
    assert retranscribed.metadata.total_chunks == 2


def test_retranscribe_session_realtime(temp_session_dir: Path, create_test_session, monkeypatch) -> None:
    """Test retranscribing a session with Realtime backend."""
    # Create a test session
    session_dir = create_test_session(session_id="original_session", duration=5.0, chunks=2)

    # Load the session
    loaded = load_session(session_dir)

    # Mock the backend runner
    from dataclasses import dataclass

    @dataclass
    class FakeRealtimeResult:
        capture_duration: float
        metadata: dict

    called_with = {}

    def fake_realtime_runner(**kwargs):
        called_with.update(kwargs)
        # Call the chunk consumer with some fake chunks
        if "chunk_consumer" in kwargs:
            consumer = kwargs["chunk_consumer"]
            consumer(0, "Realtime chunk 0.", 0.0, 2.0, None)
            consumer(1, "Realtime chunk 1.", 2.0, 4.0, None)
        return FakeRealtimeResult(
            capture_duration=5.0, metadata={"realtime_model": "gpt-realtime-mini"}
        )

    monkeypatch.setattr("transcribe_demo.realtime_backend.run_realtime_transcriber", fake_realtime_runner)

    # Retranscribe with API key
    output_dir = temp_session_dir / "retranscriptions"
    result_path = retranscribe_session(
        loaded_session=loaded,
        output_dir=output_dir,
        backend="realtime",
        backend_kwargs={"api_key": "fake_key", "realtime_model": "gpt-realtime-mini"},
    )

    # Verify the backend was called
    assert "api_key" in called_with
    assert called_with["api_key"] == "fake_key"
    assert called_with["model"] == "gpt-realtime-mini"
    assert called_with["sample_rate"] == 16000

    # Verify the new session was created
    assert result_path.exists()
    assert (result_path / "session.json").exists()
    assert (result_path / ".complete").exists()

    # Load and verify the retranscribed session
    retranscribed = load_session(result_path)
    assert "retranscribe_" in retranscribed.metadata.session_id
    assert "from_original_session" in retranscribed.metadata.session_id
    assert retranscribed.metadata.backend == "realtime"


def test_retranscribe_session_invalid_backend(temp_session_dir: Path, create_test_session) -> None:
    """Test that invalid backend raises error."""
    session_dir = create_test_session(session_id="test_session")
    loaded = load_session(session_dir)

    with pytest.raises(ValueError, match="Invalid backend"):
        retranscribe_session(
            loaded_session=loaded,
            output_dir=temp_session_dir / "retranscriptions",
            backend="invalid_backend",
        )


def test_retranscribe_session_realtime_missing_api_key(temp_session_dir: Path, create_test_session, monkeypatch) -> None:
    """Test that realtime backend without API key raises error."""
    session_dir = create_test_session(session_id="test_session")
    loaded = load_session(session_dir)

    # Ensure OPENAI_API_KEY is not set
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    # Mock the realtime runner to prevent actual execution
    def fake_realtime_runner(**kwargs):
        raise ValueError("Should not be called")

    monkeypatch.setattr("transcribe_demo.realtime_backend.run_realtime_transcriber", fake_realtime_runner)

    with pytest.raises(ValueError, match="OpenAI API key required"):
        retranscribe_session(
            loaded_session=loaded,
            output_dir=temp_session_dir / "retranscriptions",
            backend="realtime",
            backend_kwargs={},  # No API key provided
        )


def test_retranscribe_session_preserves_audio(temp_session_dir: Path, create_test_session, monkeypatch) -> None:
    """Test that retranscription preserves the original audio."""
    # Create a test session with known audio
    session_dir = create_test_session(session_id="test_session", duration=3.0)
    loaded = load_session(session_dir)
    original_audio = loaded.audio.copy()

    # Mock the backend runner
    from dataclasses import dataclass

    @dataclass
    class FakeWhisperResult:
        capture_duration: float
        full_audio_transcription: str | None
        metadata: dict

    def fake_whisper_runner(**kwargs):
        if "chunk_consumer" in kwargs:
            consumer = kwargs["chunk_consumer"]
            consumer(0, "Test chunk.", 0.0, 3.0, 0.1)
        return FakeWhisperResult(
            capture_duration=3.0, full_audio_transcription=None, metadata={"model": "turbo"}
        )

    monkeypatch.setattr("transcribe_demo.whisper_backend.run_whisper_transcriber", fake_whisper_runner)

    # Retranscribe
    output_dir = temp_session_dir / "retranscriptions"
    result_path = retranscribe_session(
        loaded_session=loaded,
        output_dir=output_dir,
        backend="whisper",
    )

    # Load the retranscribed session and verify audio is the same
    retranscribed = load_session(result_path)
    assert retranscribed.audio.shape == original_audio.shape
    # Audio should be very similar (allowing for minor numerical differences from saving/loading)
    assert np.allclose(retranscribed.audio, original_audio, rtol=1e-4, atol=1e-6)


def test_remove_incomplete_sessions(temp_session_dir: Path, create_test_session) -> None:
    """Test removing incomplete sessions."""
    # Create one complete session
    create_test_session(session_id="complete_session")

    # Create two incomplete sessions (without .complete marker)
    incomplete_dir_1 = temp_session_dir / "2025-01-01" / "incomplete_session_1"
    incomplete_dir_1.mkdir(parents=True)
    session_data_1 = {
        "metadata": {
            "session_id": "incomplete_session_1",
            "timestamp": "2025-01-01T10:00:00",
            "backend": "whisper",
            "sample_rate": 16000,
            "channels": 1,
            "capture_duration": 5.0,
            "total_chunks": 0,
        },
        "chunks": [],
    }
    with open(incomplete_dir_1 / "session.json", "w") as f:
        json.dump(session_data_1, f)

    incomplete_dir_2 = temp_session_dir / "2025-01-02" / "incomplete_session_2"
    incomplete_dir_2.mkdir(parents=True)
    session_data_2 = {
        "metadata": {
            "session_id": "incomplete_session_2",
            "timestamp": "2025-01-02T11:00:00",
            "backend": "realtime",
            "sample_rate": 16000,
            "channels": 1,
            "capture_duration": 8.0,
            "total_chunks": 0,
        },
        "chunks": [],
    }
    with open(incomplete_dir_2 / "session.json", "w") as f:
        json.dump(session_data_2, f)

    # Verify all 3 sessions exist
    all_sessions = list_sessions(temp_session_dir, include_incomplete=True)
    assert len(all_sessions) == 3

    # Remove incomplete sessions
    removed_paths = remove_incomplete_sessions(temp_session_dir)

    # Verify 2 sessions were removed
    assert len(removed_paths) == 2

    # Verify the directories were actually deleted
    assert not incomplete_dir_1.exists()
    assert not incomplete_dir_2.exists()

    # Verify only the complete session remains
    remaining_sessions = list_sessions(temp_session_dir, include_incomplete=True)
    assert len(remaining_sessions) == 1
    assert remaining_sessions[0].session_id == "complete_session"


def test_remove_incomplete_sessions_dry_run(temp_session_dir: Path) -> None:
    """Test dry run mode doesn't actually remove sessions."""
    # Create an incomplete session
    incomplete_dir = temp_session_dir / "2025-01-01" / "incomplete_session"
    incomplete_dir.mkdir(parents=True)
    session_data = {
        "metadata": {
            "session_id": "incomplete_session",
            "timestamp": "2025-01-01T10:00:00",
            "backend": "whisper",
            "sample_rate": 16000,
            "channels": 1,
            "capture_duration": 5.0,
            "total_chunks": 0,
        },
        "chunks": [],
    }
    with open(incomplete_dir / "session.json", "w") as f:
        json.dump(session_data, f)

    # Run dry run
    removed_paths = remove_incomplete_sessions(temp_session_dir, dry_run=True)

    # Verify the function returned the path
    assert len(removed_paths) == 1
    assert removed_paths[0] == incomplete_dir

    # Verify the directory still exists
    assert incomplete_dir.exists()
    assert (incomplete_dir / "session.json").exists()


def test_remove_incomplete_sessions_filter_by_backend(temp_session_dir: Path) -> None:
    """Test filtering by backend when removing incomplete sessions."""
    # Create two incomplete sessions with different backends
    whisper_dir = temp_session_dir / "2025-01-01" / "incomplete_whisper"
    whisper_dir.mkdir(parents=True)
    whisper_data = {
        "metadata": {
            "session_id": "incomplete_whisper",
            "timestamp": "2025-01-01T10:00:00",
            "backend": "whisper",
            "sample_rate": 16000,
            "channels": 1,
            "capture_duration": 5.0,
            "total_chunks": 0,
        },
        "chunks": [],
    }
    with open(whisper_dir / "session.json", "w") as f:
        json.dump(whisper_data, f)

    realtime_dir = temp_session_dir / "2025-01-02" / "incomplete_realtime"
    realtime_dir.mkdir(parents=True)
    realtime_data = {
        "metadata": {
            "session_id": "incomplete_realtime",
            "timestamp": "2025-01-02T11:00:00",
            "backend": "realtime",
            "sample_rate": 16000,
            "channels": 1,
            "capture_duration": 8.0,
            "total_chunks": 0,
        },
        "chunks": [],
    }
    with open(realtime_dir / "session.json", "w") as f:
        json.dump(realtime_data, f)

    # Remove only whisper incomplete sessions
    removed_paths = remove_incomplete_sessions(temp_session_dir, backend="whisper")

    # Verify only whisper session was removed
    assert len(removed_paths) == 1
    assert not whisper_dir.exists()
    assert realtime_dir.exists()


def test_remove_incomplete_sessions_filter_by_duration(temp_session_dir: Path) -> None:
    """Test filtering by duration when removing incomplete sessions."""
    # Create two incomplete sessions with different durations
    short_dir = temp_session_dir / "2025-01-01" / "incomplete_short"
    short_dir.mkdir(parents=True)
    short_data = {
        "metadata": {
            "session_id": "incomplete_short",
            "timestamp": "2025-01-01T10:00:00",
            "backend": "whisper",
            "sample_rate": 16000,
            "channels": 1,
            "capture_duration": 3.0,
            "total_chunks": 0,
        },
        "chunks": [],
    }
    with open(short_dir / "session.json", "w") as f:
        json.dump(short_data, f)

    long_dir = temp_session_dir / "2025-01-02" / "incomplete_long"
    long_dir.mkdir(parents=True)
    long_data = {
        "metadata": {
            "session_id": "incomplete_long",
            "timestamp": "2025-01-02T11:00:00",
            "backend": "whisper",
            "sample_rate": 16000,
            "channels": 1,
            "capture_duration": 15.0,
            "total_chunks": 0,
        },
        "chunks": [],
    }
    with open(long_dir / "session.json", "w") as f:
        json.dump(long_data, f)

    # Remove only sessions with duration >= 10.0
    removed_paths = remove_incomplete_sessions(temp_session_dir, min_duration=10.0)

    # Verify only long session was removed
    assert len(removed_paths) == 1
    assert short_dir.exists()
    assert not long_dir.exists()


def test_remove_incomplete_sessions_no_incomplete(temp_session_dir: Path, create_test_session) -> None:
    """Test behavior when no incomplete sessions exist."""
    # Create only complete sessions
    create_test_session(session_id="complete_session_1")
    create_test_session(session_id="complete_session_2")

    # Try to remove incomplete sessions
    removed_paths = remove_incomplete_sessions(temp_session_dir)

    # Verify nothing was removed
    assert len(removed_paths) == 0

    # Verify all sessions still exist
    remaining_sessions = list_sessions(temp_session_dir)
    assert len(remaining_sessions) == 2


def test_remove_incomplete_sessions_empty_directory(temp_session_dir: Path) -> None:
    """Test behavior in an empty directory."""
    removed_paths = remove_incomplete_sessions(temp_session_dir)

    # Verify nothing was removed
    assert len(removed_paths) == 0
