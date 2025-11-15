"""Tests for backend time limits, stop requests, and session logging."""

from __future__ import annotations

import json
import queue

import numpy as np
import pytest

from test_helpers import FakeWebSocket, create_fake_audio_capture_factory, generate_synthetic_audio
from transcribe_demo import realtime_backend, whisper_backend
from transcribe_demo.backend_protocol import TranscriptionChunk
from transcribe_demo.session_logger import SessionLogger


# --- Whisper Backend Tests ---


def _collect_chunks_from_queue(chunk_queue: queue.Queue[TranscriptionChunk | None]) -> list[dict]:
    """Helper to collect chunks from queue."""
    chunks: list[dict] = []
    while True:
        try:
            chunk = chunk_queue.get(timeout=0.1)
            if chunk is None:
                break
            if not chunk.is_partial:
                chunks.append({
                    "index": chunk.index,
                    "text": chunk.text,
                    "start": chunk.start_time,
                    "end": chunk.end_time,
                })
        except queue.Empty:
            break
    return chunks


def test_whisper_backend_respects_time_limit(monkeypatch):
    """Test that Whisper backend stops after max_capture_duration is reached."""
    audio, sample_rate = generate_synthetic_audio(duration_seconds=4.0)
    time_limit = 2.5  # Stop after 2.5 seconds - enough for at least one chunk

    chunk_queue: queue.Queue[TranscriptionChunk | None] = queue.Queue()
    chunks: list[dict] = []

    class DummyModel:
        def transcribe(self, audio_chunk: np.ndarray, **kwargs):
            return {"text": f"Chunk {len(chunks)}"}

    dummy_model = DummyModel()

    def fake_load_whisper_model(**kwargs):
        return dummy_model, "cpu", False

    monkeypatch.setattr(whisper_backend, "load_whisper_model", fake_load_whisper_model)

    # Create fake audio source with time limit
    from test_helpers import FakeAudioCaptureManager
    audio_source = FakeAudioCaptureManager(
        audio=audio,
        sample_rate=sample_rate,
        channels=1,
        max_capture_duration=time_limit,
        collect_full_audio=True,
        frame_size=480,
    )

    result = whisper_backend.run_whisper_transcriber(
        model_name="fixture",
        sample_rate=sample_rate,
        channels=1,
        temp_file=None,
        ca_cert=None,
        disable_ssl_verify=False,
        device_preference="cpu",
        require_gpu=False,
        chunk_queue=chunk_queue,
        vad_aggressiveness=0,
        vad_min_silence_duration=0.05,  # Very short to allow quick chunking
        vad_min_speech_duration=0.05,
        vad_speech_pad_duration=0.0,
        max_chunk_duration=1.5,  # Short max chunk to get chunks faster
        compare_transcripts=True,
        language="en",
        audio_source=audio_source,
    )

    # Collect chunks from queue
    chunks = _collect_chunks_from_queue(chunk_queue)

    # Verify that capture stopped around the time limit
    assert result.capture_duration <= time_limit * 1.5  # Allow 50% margin
    assert result.capture_duration > 0

    # Verify that we got at least one chunk
    assert len(chunks) > 0

    # Verify that full audio transcription was generated for comparison
    assert result.full_audio_transcription is not None


def test_whisper_backend_transcribes_incomplete_chunk_on_timeout(monkeypatch):
    """Test that incomplete chunks are transcribed immediately when time limit is hit."""
    audio, sample_rate = generate_synthetic_audio(duration_seconds=5.0)
    # Set time limit to cut off mid-speech, creating an incomplete final chunk
    time_limit = 2.3  # This should create a final chunk < 2s (min_chunk_size)

    chunk_queue: queue.Queue[TranscriptionChunk | None] = queue.Queue()
    transcribed_audio_durations: list[float] = []

    class DummyModel:
        def transcribe(self, audio_chunk: np.ndarray, **kwargs):
            # Track the duration of each transcribed audio chunk
            duration = len(audio_chunk) / sample_rate
            transcribed_audio_durations.append(duration)
            return {"text": f"Chunk {len(transcribed_audio_durations) - 1} (duration: {duration:.2f}s)"}

    dummy_model = DummyModel()

    def fake_load_whisper_model(**kwargs):
        return dummy_model, "cpu", False

    monkeypatch.setattr(whisper_backend, "load_whisper_model", fake_load_whisper_model)

    # Create fake audio source with time limit
    from test_helpers import FakeAudioCaptureManager
    audio_source = FakeAudioCaptureManager(
        audio=audio,
        sample_rate=sample_rate,
        channels=1,
        max_capture_duration=time_limit,
        collect_full_audio=True,
        frame_size=480,
    )

    result = whisper_backend.run_whisper_transcriber(
        model_name="fixture",
        sample_rate=sample_rate,
        channels=1,
        temp_file=None,
        ca_cert=None,
        disable_ssl_verify=False,
        device_preference="cpu",
        require_gpu=False,
        chunk_queue=chunk_queue,
        vad_aggressiveness=0,  # Low aggressiveness to ensure speech detection
        vad_min_silence_duration=0.5,  # Long silence requirement - won't be met before timeout
        vad_min_speech_duration=0.05,
        vad_speech_pad_duration=0.0,
        max_chunk_duration=10.0,  # High value - won't trigger before timeout
        compare_transcripts=True,
        language="en",
        audio_source=audio_source,
    )

    # Collect chunks from queue
    chunks = _collect_chunks_from_queue(chunk_queue)

    # Verify that we got at least one chunk
    assert len(chunks) > 0, "Expected at least one chunk to be transcribed"

    # Verify that the last chunk is incomplete (< 2s which is min_chunk_size)
    # Note: Some chunks might be padded, so we check the actual transcribed audio durations
    assert len(transcribed_audio_durations) > 0, "Expected at least one transcribed audio chunk"

    # At least one chunk should be less than min_chunk_size (2.0s), demonstrating
    # that we transcribe incomplete chunks on timeout
    has_incomplete_chunk = any(duration < 2.0 for duration in transcribed_audio_durations)
    assert has_incomplete_chunk, (
        f"Expected at least one incomplete chunk (< 2.0s), but all chunks were >= 2.0s: {transcribed_audio_durations}"
    )

    # Verify capture stopped around the time limit
    assert result.capture_duration <= time_limit * 1.5  # Allow 50% margin


def test_whisper_backend_logs_session(monkeypatch, temp_session_dir):
    """Test that Whisper backend properly logs session with all metadata."""
    audio, sample_rate = generate_synthetic_audio(duration_seconds=3.0)
    full_duration = len(audio) / sample_rate
    time_limit = full_duration * 0.5

    chunk_queue: queue.Queue[TranscriptionChunk | None] = queue.Queue()

    class DummyModel:
        def transcribe(self, audio_chunk: np.ndarray, **kwargs):
            chunk_num = sum(1 for _ in _collect_chunks_from_queue(queue.Queue()))  # Count existing
            return {"text": f"Whisper chunk {chunk_num}"}

    dummy_model = DummyModel()

    def fake_load_whisper_model(**kwargs):
        return dummy_model, "cpu", False

    monkeypatch.setattr(whisper_backend, "load_whisper_model", fake_load_whisper_model)

    # Create fake audio source with time limit
    from test_helpers import FakeAudioCaptureManager
    audio_source = FakeAudioCaptureManager(
        audio=audio,
        sample_rate=sample_rate,
        channels=1,
        max_capture_duration=time_limit,
        collect_full_audio=True,
        frame_size=480,
    )

    # Create session logger with temp directory
    session_logger = SessionLogger(
        output_dir=temp_session_dir,
        sample_rate=sample_rate,
        channels=1,
        backend="whisper",
        save_chunk_audio=True,
        session_id="test_whisper_session",
    )

    result = whisper_backend.run_whisper_transcriber(
        model_name="fixture",
        sample_rate=sample_rate,
        channels=1,
        temp_file=None,
        ca_cert=None,
        disable_ssl_verify=False,
        device_preference="cpu",
        require_gpu=False,
        chunk_queue=chunk_queue,
        vad_aggressiveness=2,
        vad_min_silence_duration=0.2,
        vad_min_speech_duration=0.05,
        vad_speech_pad_duration=0.1,
        max_chunk_duration=5.0,
        compare_transcripts=True,
        language="en",
        session_logger=session_logger,
        min_log_duration=0.0,
        audio_source=audio_source,
    )

    # Collect chunks from queue
    chunks = _collect_chunks_from_queue(chunk_queue)

    # Create stitched transcription
    stitched = " ".join(chunk["text"] for chunk in chunks)

    # Finalize session
    session_logger.finalize(
        capture_duration=result.capture_duration,
        full_audio_transcription=result.full_audio_transcription,
        stitched_transcription=stitched,
        extra_metadata=result.metadata,
        min_duration=0.0,
    )

    # Verify session files were created
    session_dir = session_logger.session_dir
    assert session_dir.exists()
    assert (session_dir / "session.json").exists()
    assert (session_dir / "full_audio.flac").exists()
    assert (session_dir / "README.txt").exists()

    # Verify session.json contains correct data
    with open(session_dir / "session.json", "r") as f:
        session_data = json.load(f)

    assert session_data["metadata"]["backend"] == "whisper"
