"""Tests for session_logger module."""

from __future__ import annotations

import json
from pathlib import Path


from transcribe_demo.session_logger import ChunkMetadata, SessionLogger, SessionMetadata


def test_chunk_metadata_with_cleaned_text() -> None:
    """Test that ChunkMetadata includes cleaned_text field."""
    chunk = ChunkMetadata(
        index=0,
        text="Hello, world.",
        start_time=0.0,
        end_time=1.0,
        duration=1.0,
        cleaned_text="Hello, world",
    )
    assert chunk.text == "Hello, world."
    assert chunk.cleaned_text == "Hello, world"


def test_session_metadata_with_diff_fields() -> None:
    """Test that SessionMetadata includes transcription diff fields."""
    metadata = SessionMetadata(
        session_id="test_session",
        timestamp="2025-01-01T00:00:00",
        backend="whisper",
        sample_rate=16000,
        channels=1,
        capture_duration=10.0,
        total_chunks=3,
        full_audio_transcription="Hello world",
        stitched_transcription="Hello world",
        transcription_similarity=1.0,
        transcription_diffs=[],
    )
    assert metadata.transcription_similarity == 1.0
    assert metadata.transcription_diffs == []


def test_session_logger_update_chunk_cleaned_text(temp_session_dir: Path) -> None:
    """Test updating chunk cleaned text."""
    logger = SessionLogger(
        output_dir=temp_session_dir,
        sample_rate=16000,
        channels=1,
        backend="whisper",
        save_chunk_audio=False,
    )

    # Log a chunk without cleaned text
    logger.log_chunk(
        index=0,
        text="Hello, world.",
        start_time=0.0,
        end_time=1.0,
    )

    # Update cleaned text
    logger.update_chunk_cleaned_text(0, "Hello, world")

    # Verify update
    assert len(logger.chunks) == 1
    assert logger.chunks[0].text == "Hello, world."
    assert logger.chunks[0].cleaned_text == "Hello, world"


def test_session_logger_finalize_with_diff(temp_session_dir: Path) -> None:
    """Test that session logger saves diff information."""
    logger = SessionLogger(
        output_dir=temp_session_dir,
        sample_rate=16000,
        channels=1,
        backend="whisper",
        save_chunk_audio=False,
    )

    # Log chunks
    logger.log_chunk(
        index=0,
        text="Hello, world.",
        start_time=0.0,
        end_time=1.0,
        cleaned_text="Hello, world",
    )
    logger.log_chunk(
        index=1,
        text="How are you?",
        start_time=1.0,
        end_time=2.0,
        cleaned_text="How are you?",
    )

    # Finalize with diff information
    diff_snippets = [
        {
            "tag": "replace",
            "stitched": "Hello, world How are you?",
            "complete": "Hello world. How are you?",
        }
    ]

    logger.finalize(
        capture_duration=2.0,
        full_audio_transcription="Hello world. How are you?",
        stitched_transcription="Hello world How are you?",
        transcription_similarity=0.95,
        transcription_diffs=diff_snippets,
    )

    # Verify session.json was created and contains diff info
    session_json_path = logger.session_dir / "session.json"
    assert session_json_path.exists()

    with open(session_json_path, encoding="utf-8") as f:
        session_data = json.load(f)

    assert session_data["metadata"]["transcription_similarity"] == 0.95
    assert len(session_data["metadata"]["transcription_diffs"]) == 1
    assert session_data["metadata"]["transcription_diffs"][0]["tag"] == "replace"

    # Verify chunks have cleaned text
    assert len(session_data["chunks"]) == 2
    assert session_data["chunks"][0]["cleaned_text"] == "Hello, world"
    assert session_data["chunks"][1]["cleaned_text"] == "How are you?"

    # Verify README.txt contains diff info
    readme_path = logger.session_dir / "README.txt"
    assert readme_path.exists()

    readme_content = readme_path.read_text(encoding="utf-8")
    assert "TRANSCRIPTION COMPARISON:" in readme_content
    assert "Similarity: 95.00%" in readme_content
    assert "Differences found: 1" in readme_content
    assert "Text (original):" in readme_content
    assert "Text (cleaned):" in readme_content


def test_session_logger_finalize_matching_transcripts(temp_session_dir: Path) -> None:
    """Test that session logger handles matching transcripts correctly."""
    logger = SessionLogger(
        output_dir=temp_session_dir,
        sample_rate=16000,
        channels=1,
        backend="whisper",
        save_chunk_audio=False,
    )

    logger.log_chunk(
        index=0,
        text="Hello world",
        start_time=0.0,
        end_time=1.0,
        cleaned_text="Hello world",
    )

    logger.finalize(
        capture_duration=1.0,
        full_audio_transcription="Hello world",
        stitched_transcription="Hello world",
        transcription_similarity=1.0,
        transcription_diffs=[],
    )

    # Verify README mentions exact match
    readme_path = logger.session_dir / "README.txt"
    readme_content = readme_path.read_text(encoding="utf-8")
    assert "Similarity: 100.00%" in readme_content
    assert "Transcriptions match exactly." in readme_content


def test_session_logger_finalize_without_diff(temp_session_dir: Path) -> None:
    """Test that session logger works without diff information."""
    logger = SessionLogger(
        output_dir=temp_session_dir,
        sample_rate=16000,
        channels=1,
        backend="realtime",
        save_chunk_audio=False,
    )

    logger.log_chunk(
        index=0,
        text="Hello world",
        start_time=0.0,
        end_time=1.0,
    )

    # Finalize without diff information (e.g., comparison disabled)
    logger.finalize(
        capture_duration=1.0,
        stitched_transcription="Hello world",
    )

    # Verify session.json was created
    session_json_path = logger.session_dir / "session.json"
    assert session_json_path.exists()

    with open(session_json_path, encoding="utf-8") as f:
        session_data = json.load(f)

    # Diff fields should be None
    assert session_data["metadata"]["transcription_similarity"] is None
    assert session_data["metadata"]["transcription_diffs"] is None

    # README should not have comparison section
    readme_path = logger.session_dir / "README.txt"
    readme_content = readme_path.read_text(encoding="utf-8")
    assert "TRANSCRIPTION COMPARISON:" not in readme_content
