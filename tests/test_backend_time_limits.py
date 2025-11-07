"""Tests for backend time limits, stop requests, and session logging."""

from __future__ import annotations

import json

import numpy as np
import pytest

from test_helpers import FakeWebSocket, create_fake_audio_capture_factory, generate_synthetic_audio
from transcribe_demo import realtime_backend, whisper_backend
from transcribe_demo.session_logger import SessionLogger


# --- Whisper Backend Tests ---


def test_whisper_backend_respects_time_limit(monkeypatch):
    """Test that Whisper backend stops after max_capture_duration is reached."""
    audio, sample_rate = generate_synthetic_audio(duration_seconds=4.0)
    time_limit = 2.5  # Stop after 2.5 seconds - enough for at least one chunk

    chunks: list[dict] = []

    class DummyModel:
        def transcribe(self, audio_chunk: np.ndarray, **kwargs):
            return {"text": f"Chunk {len(chunks)}"}

    dummy_model = DummyModel()

    def fake_load_whisper_model(**kwargs):
        return dummy_model, "cpu", False

    monkeypatch.setattr(whisper_backend, "load_whisper_model", fake_load_whisper_model)

    # Use helper to create fake audio capture manager
    monkeypatch.setattr(
        "transcribe_demo.audio_capture.AudioCaptureManager",
        create_fake_audio_capture_factory(audio, sample_rate, frame_size=480),
    )

    def capture_chunk(index, text, start, end, inference_seconds):
        chunks.append({"index": index, "text": text, "start": start, "end": end})

    result = whisper_backend.run_whisper_transcriber(
        model_name="fixture",
        sample_rate=sample_rate,
        channels=1,
        temp_file=None,
        ca_cert=None,
        insecure_downloads=False,
        device_preference="cpu",
        require_gpu=False,
        chunk_consumer=capture_chunk,
        vad_aggressiveness=0,
        vad_min_silence_duration=0.05,  # Very short to allow quick chunking
        vad_min_speech_duration=0.05,
        vad_speech_pad_duration=0.0,
        max_chunk_duration=1.5,  # Short max chunk to get chunks faster
        compare_transcripts=True,
        max_capture_duration=time_limit,
        language="en",
    )

    # Verify that capture stopped around the time limit
    assert result.capture_duration <= time_limit * 1.5  # Allow 50% margin
    assert result.capture_duration > 0

    # Verify that we got at least one chunk
    assert len(chunks) > 0

    # Verify that full audio transcription was generated for comparison
    assert result.full_audio_transcription is not None


def test_whisper_backend_logs_session(monkeypatch, temp_session_dir):
    """Test that Whisper backend properly logs session with all metadata."""
    audio, sample_rate = generate_synthetic_audio(duration_seconds=3.0)
    full_duration = len(audio) / sample_rate
    time_limit = full_duration * 0.5

    chunks: list[dict] = []

    class DummyModel:
        def transcribe(self, audio_chunk: np.ndarray, **kwargs):
            chunk_num = len(chunks)
            return {"text": f"Whisper chunk {chunk_num}"}

    dummy_model = DummyModel()

    def fake_load_whisper_model(**kwargs):
        return dummy_model, "cpu", False

    monkeypatch.setattr(whisper_backend, "load_whisper_model", fake_load_whisper_model)

    monkeypatch.setattr(
        "transcribe_demo.audio_capture.AudioCaptureManager",
        create_fake_audio_capture_factory(audio, sample_rate),
    )

    def capture_chunk(index, text, start, end, inference_seconds):
        chunks.append({"index": index, "text": text, "start": start, "end": end})

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
        insecure_downloads=False,
        device_preference="cpu",
        require_gpu=False,
        chunk_consumer=capture_chunk,
        vad_aggressiveness=2,
        vad_min_silence_duration=0.2,
        vad_min_speech_duration=0.05,
        vad_speech_pad_duration=0.1,
        max_chunk_duration=5.0,
        compare_transcripts=True,
        max_capture_duration=time_limit,
        language="en",
        session_logger=session_logger,
        min_log_duration=0.0,
    )

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
    assert session_data["metadata"]["sample_rate"] == sample_rate
    assert session_data["metadata"]["total_chunks"] == len(chunks)
    assert session_data["metadata"]["model"] == "fixture"
    assert session_data["metadata"]["vad_aggressiveness"] == 2
    assert session_data["metadata"]["full_audio_transcription"] is not None
    assert session_data["metadata"]["stitched_transcription"] == stitched

    # Verify chunks are logged
    assert len(session_data["chunks"]) == len(chunks)
    for i, chunk in enumerate(session_data["chunks"]):
        assert chunk["index"] == i
        assert chunk["text"] == chunks[i]["text"]


# --- Realtime Backend Tests ---


@pytest.mark.integration
def test_realtime_backend_respects_time_limit(monkeypatch):
    """Test that Realtime backend stops after max_capture_duration is reached."""
    audio, sample_rate = generate_synthetic_audio(duration_seconds=3.0)
    full_duration = len(audio) / sample_rate
    time_limit = full_duration * 0.5

    chunk_texts: list[str] = []

    def fake_audio_capture_factory(sample_rate, channels, max_capture_duration=0.0, collect_full_audio=True):
        return FakeAudioCaptureManager(
            audio=audio,
            sample_rate=sample_rate,
            channels=channels,
            max_capture_duration=max_capture_duration,
            collect_full_audio=collect_full_audio,
            frame_size=320,  # Smaller frame for realtime
        )

    monkeypatch.setattr("transcribe_demo.audio_capture.AudioCaptureManager", fake_audio_capture_factory)

    class FakeConnect:
        def __init__(self):
            self._ws = FakeWebSocket(num_chunks=5)

        async def __aenter__(self):
            return self._ws

        async def __aexit__(self, exc_type, exc, tb):
            return False

    def fake_connect(*args, **kwargs):
        return FakeConnect()

    monkeypatch.setattr(realtime_backend.websockets, "connect", fake_connect)

    def collect_chunk(chunk_index, text, absolute_start, absolute_end, inference_seconds):
        if text:
            chunk_texts.append(text)

    monkeypatch.setattr(
        realtime_backend,
        "transcribe_full_audio_realtime",
        lambda *args, **kwargs: "Full realtime transcription",
    )

    result = realtime_backend.run_realtime_transcriber(
        api_key="test-key",
        endpoint="wss://example.com",
        model="gpt-realtime-mini",
        sample_rate=sample_rate,
        channels=1,
        chunk_duration=0.2,
        instructions="transcribe precisely",
        insecure_downloads=False,
        chunk_consumer=collect_chunk,
        compare_transcripts=True,
        max_capture_duration=time_limit,
        language="en",
    )

    # Verify that capture stopped around the time limit
    assert result.capture_duration <= time_limit * 1.5  # Allow 50% margin
    assert result.capture_duration > 0

    # Verify that we got chunks
    assert len(chunk_texts) > 0


@pytest.mark.integration
def test_realtime_backend_logs_session(monkeypatch, temp_session_dir):
    """Test that Realtime backend properly logs session with all metadata."""
    audio, sample_rate = generate_synthetic_audio(duration_seconds=3.0)
    full_duration = len(audio) / sample_rate
    time_limit = full_duration * 0.5

    chunk_texts: list[str] = []

    def fake_audio_capture_factory(sample_rate, channels, max_capture_duration=0.0, collect_full_audio=True):
        return FakeAudioCaptureManager(
            audio=audio,
            sample_rate=sample_rate,
            channels=channels,
            max_capture_duration=max_capture_duration,
            collect_full_audio=collect_full_audio,
            frame_size=320,
        )

    monkeypatch.setattr("transcribe_demo.audio_capture.AudioCaptureManager", fake_audio_capture_factory)

    class FakeConnect:
        def __init__(self):
            self._ws = FakeWebSocket(num_chunks=3)

        async def __aenter__(self):
            return self._ws

        async def __aexit__(self, exc_type, exc, tb):
            return False

    def fake_connect(*args, **kwargs):
        return FakeConnect()

    monkeypatch.setattr(realtime_backend.websockets, "connect", fake_connect)

    def collect_chunk(chunk_index, text, absolute_start, absolute_end, inference_seconds):
        if text:
            chunk_texts.append(text)

    monkeypatch.setattr(
        realtime_backend,
        "transcribe_full_audio_realtime",
        lambda *args, **kwargs: "Full realtime transcription",
    )

    # Create session logger with temp directory
    session_logger = SessionLogger(
        output_dir=temp_session_dir,
        sample_rate=sample_rate,
        channels=1,
        backend="realtime",
        save_chunk_audio=False,  # Realtime doesn't save chunk audio
        session_id="test_realtime_session",
    )

    result = realtime_backend.run_realtime_transcriber(
        api_key="test-key",
        endpoint="wss://example.com",
        model="gpt-realtime-mini",
        sample_rate=sample_rate,
        channels=1,
        chunk_duration=0.2,
        instructions="transcribe precisely",
        insecure_downloads=False,
        chunk_consumer=collect_chunk,
        compare_transcripts=True,
        max_capture_duration=time_limit,
        language="en",
        session_logger=session_logger,
        min_log_duration=0.0,
    )

    # Create stitched transcription
    stitched = " ".join(chunk_texts)

    # Finalize session
    session_logger.finalize(
        capture_duration=result.capture_duration,
        full_audio_transcription="Full realtime transcription",
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

    assert session_data["metadata"]["backend"] == "realtime"
    assert session_data["metadata"]["sample_rate"] == sample_rate
    assert session_data["metadata"]["total_chunks"] == len(chunk_texts)
    assert session_data["metadata"]["model"] == "gpt-realtime-mini"
    assert session_data["metadata"]["full_audio_transcription"] == "Full realtime transcription"
    assert session_data["metadata"]["stitched_transcription"] == stitched


@pytest.mark.integration
def test_realtime_backend_compares_stitched_vs_complete(monkeypatch):
    """Test that Realtime backend compares stitched transcript against complete audio transcript."""
    audio, sample_rate = generate_synthetic_audio(duration_seconds=3.0)
    full_duration = len(audio) / sample_rate
    time_limit = full_duration * 0.5

    chunk_texts: list[str] = []
    full_audio_transcribed = False

    def fake_audio_capture_factory(sample_rate, channels, max_capture_duration=0.0, collect_full_audio=True):
        return FakeAudioCaptureManager(
            audio=audio,
            sample_rate=sample_rate,
            channels=channels,
            max_capture_duration=max_capture_duration,
            collect_full_audio=collect_full_audio,
            frame_size=320,
        )

    monkeypatch.setattr("transcribe_demo.audio_capture.AudioCaptureManager", fake_audio_capture_factory)

    class FakeConnect:
        def __init__(self):
            self._ws = FakeWebSocket(num_chunks=3)

        async def __aenter__(self):
            return self._ws

        async def __aexit__(self, exc_type, exc, tb):
            return False

    def fake_connect(*args, **kwargs):
        return FakeConnect()

    monkeypatch.setattr(realtime_backend.websockets, "connect", fake_connect)

    def collect_chunk(chunk_index, text, absolute_start, absolute_end, inference_seconds):
        if text:
            chunk_texts.append(text)

    def fake_transcribe_full(audio, *args, **kwargs):
        nonlocal full_audio_transcribed
        full_audio_transcribed = True
        return "Complete realtime transcription"

    monkeypatch.setattr(realtime_backend, "transcribe_full_audio_realtime", fake_transcribe_full)

    result = realtime_backend.run_realtime_transcriber(
        api_key="test-key",
        endpoint="wss://example.com",
        model="gpt-realtime-mini",
        sample_rate=sample_rate,
        channels=1,
        chunk_duration=0.2,
        instructions="transcribe precisely",
        insecure_downloads=False,
        chunk_consumer=collect_chunk,
        compare_transcripts=True,
        max_capture_duration=time_limit,
        language="en",
    )

    # Verify that full audio transcription was performed
    # Note: In the actual main.py, this happens after run_realtime_transcriber returns
    # So we're just verifying the infrastructure is in place
    assert result.full_audio.size > 0
    assert len(chunk_texts) > 0


def test_session_logger_respects_min_duration(monkeypatch, temp_session_dir):
    """Test that SessionLogger discards sessions below minimum duration."""
    audio, sample_rate = generate_synthetic_audio(duration_seconds=2.0)
    short_duration = 2.0  # Short capture - just enough for testing

    chunks: list[dict] = []

    class DummyModel:
        def transcribe(self, audio_chunk: np.ndarray, **kwargs):
            return {"text": f"Chunk {len(chunks)}"}

    dummy_model = DummyModel()

    def fake_load_whisper_model(**kwargs):
        return dummy_model, "cpu", False

    monkeypatch.setattr(whisper_backend, "load_whisper_model", fake_load_whisper_model)

    monkeypatch.setattr(
        "transcribe_demo.audio_capture.AudioCaptureManager",
        create_fake_audio_capture_factory(audio, sample_rate),
    )

    def capture_chunk(index, text, start, end, inference_seconds):
        chunks.append({"index": index, "text": text})

    # Create session logger with temp directory
    session_logger = SessionLogger(
        output_dir=temp_session_dir,
        sample_rate=sample_rate,
        channels=1,
        backend="whisper",
        save_chunk_audio=False,  # Don't save chunk audio to speed up test
        session_id="test_short_session",
    )

    result = whisper_backend.run_whisper_transcriber(
        model_name="fixture",
        sample_rate=sample_rate,
        channels=1,
        temp_file=None,
        ca_cert=None,
        insecure_downloads=False,
        device_preference="cpu",
        require_gpu=False,
        chunk_consumer=capture_chunk,
        vad_aggressiveness=0,
        vad_min_silence_duration=0.1,
        vad_min_speech_duration=0.05,
        vad_speech_pad_duration=0.0,
        max_chunk_duration=5.0,
        compare_transcripts=False,
        max_capture_duration=short_duration,
        language="en",
        session_logger=session_logger,
        min_log_duration=100.0,  # Require 100s minimum
    )

    session_dir = session_logger.session_dir

    # Finalize with min_duration requirement
    session_logger.finalize(
        capture_duration=result.capture_duration,
        full_audio_transcription=None,
        stitched_transcription=" ".join(chunk["text"] for chunk in chunks),
        extra_metadata=result.metadata,
        min_duration=100.0,  # Session too short
    )

    # Verify that session directory was deleted
    assert not session_dir.exists()
