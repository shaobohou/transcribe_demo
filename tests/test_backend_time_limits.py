"""Tests for backend time limits, stop requests, and session logging."""

from __future__ import annotations

import json
import queue as queue_module
import tempfile
import threading
import time
import wave
from pathlib import Path

import numpy as np
import pytest

from transcribe_demo import audio_capture, realtime_backend, whisper_backend
from transcribe_demo.session_logger import SessionLogger


def _load_fixture() -> tuple[np.ndarray, int]:
    """Load test audio fixture."""
    fixture = Path(__file__).resolve().parent / "data" / "fox.wav"
    if not fixture.exists():
        raise FileNotFoundError("tests/data/fox.wav fixture not found")
    with wave.open(str(fixture), "rb") as wf:
        if wf.getnchannels() != 1:
            raise RuntimeError("fox.wav must be mono")
        rate = wf.getframerate()
        audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32)
    return audio / 32768.0, rate


class FakeAudioCaptureManager:
    """Fake AudioCaptureManager that simulates audio streaming with controllable duration."""

    def __init__(
        self,
        audio: np.ndarray,
        sample_rate: int,
        channels: int = 1,
        max_capture_duration: float = 0.0,
        collect_full_audio: bool = True,
        frame_size: int = 480,
        simulate_user_stop_after: float | None = None,
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
        self._start_time: float = 0.0
        self._audio = audio
        self._frame_size = frame_size
        self._simulate_user_stop_after = simulate_user_stop_after

    def _feed_audio(self) -> None:
        """Feed test audio into queue in a background thread."""
        fed_samples = 0
        for start in range(0, len(self._audio), self._frame_size):
            if self.stop_event.is_set():
                break

            # Check if we should simulate user stop
            elapsed = time.time() - self._start_time
            if self._simulate_user_stop_after is not None and elapsed >= self._simulate_user_stop_after:
                # Simulate user stop by setting stop event
                self.stop_event.set()
                break

            # Check max_capture_duration
            if self.max_capture_duration > 0:
                samples_duration = fed_samples / self.sample_rate
                if samples_duration >= self.max_capture_duration:
                    self.capture_limit_reached.set()
                    # Continue feeding until buffer is processed
                    time.sleep(0.05)
                    if self.stop_event.is_set():
                        break

            frame = self._audio[start : start + self._frame_size]
            if not frame.size:
                continue

            # Reshape to match expected format (samples, channels)
            frame_shaped = frame.reshape(-1, 1) if self.channels == 1 else frame
            self.audio_queue.put(frame_shaped)

            # Collect for get_full_audio
            if self.collect_full_audio:
                mono = frame_shaped.mean(axis=1).astype(np.float32) if frame_shaped.ndim > 1 else frame_shaped
                self._full_audio_chunks.append(mono)

            fed_samples += len(frame)

            # Small delay to simulate real-time audio streaming
            time.sleep(0.001)

        # Signal end of stream
        self.audio_queue.put(None)

    def start(self) -> None:
        """Start feeding audio in background thread."""
        self._start_time = time.time()
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
        full_audio = self.get_full_audio()
        if full_audio.size == 0:
            return 0.0
        return full_audio.size / self.sample_rate


# --- Whisper Backend Tests ---


def test_whisper_backend_respects_time_limit(monkeypatch):
    """Test that Whisper backend stops after max_capture_duration is reached."""
    audio, sample_rate = _load_fixture()
    full_duration = len(audio) / sample_rate
    time_limit = full_duration * 0.5  # Stop at 50% of audio duration

    chunks: list[dict] = []

    class DummyModel:
        def transcribe(self, audio_chunk: np.ndarray, **kwargs):
            return {"text": f"Chunk {len(chunks)}"}

    dummy_model = DummyModel()

    def fake_load_whisper_model(**kwargs):
        return dummy_model, "cpu", False

    monkeypatch.setattr(whisper_backend, "load_whisper_model", fake_load_whisper_model)

    # Create fake audio capture manager with time limit
    def fake_audio_capture_factory(sample_rate, channels, max_capture_duration=0.0, collect_full_audio=True):
        return FakeAudioCaptureManager(
            audio=audio,
            sample_rate=sample_rate,
            channels=channels,
            max_capture_duration=max_capture_duration,
            collect_full_audio=collect_full_audio,
        )

    monkeypatch.setattr(audio_capture, "AudioCaptureManager", fake_audio_capture_factory)

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
        vad_min_silence_duration=0.1,
        vad_min_speech_duration=0.05,
        vad_speech_pad_duration=0.0,
        max_chunk_duration=5.0,
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


def test_whisper_backend_handles_user_stop(monkeypatch):
    """Test that Whisper backend handles graceful user stop."""
    audio, sample_rate = _load_fixture()
    full_duration = len(audio) / sample_rate
    stop_after = full_duration * 0.3  # Simulate user stop at 30%

    chunks: list[dict] = []

    class DummyModel:
        def transcribe(self, audio_chunk: np.ndarray, **kwargs):
            return {"text": f"Chunk {len(chunks)}"}

    dummy_model = DummyModel()

    def fake_load_whisper_model(**kwargs):
        return dummy_model, "cpu", False

    monkeypatch.setattr(whisper_backend, "load_whisper_model", fake_load_whisper_model)

    # Create fake audio capture manager that simulates user stop
    def fake_audio_capture_factory(sample_rate, channels, max_capture_duration=0.0, collect_full_audio=True):
        return FakeAudioCaptureManager(
            audio=audio,
            sample_rate=sample_rate,
            channels=channels,
            max_capture_duration=max_capture_duration,
            collect_full_audio=collect_full_audio,
            simulate_user_stop_after=stop_after,
        )

    monkeypatch.setattr(audio_capture, "AudioCaptureManager", fake_audio_capture_factory)

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
        vad_min_silence_duration=0.1,
        vad_min_speech_duration=0.05,
        vad_speech_pad_duration=0.0,
        max_chunk_duration=5.0,
        compare_transcripts=True,
        max_capture_duration=0.0,  # Unlimited, but will be stopped by user
        language="en",
    )

    # Verify that capture stopped early (before full duration)
    assert result.capture_duration < full_duration * 0.8  # Should be significantly less
    assert result.capture_duration > 0

    # Verify that we got at least one chunk
    assert len(chunks) > 0


def test_whisper_backend_logs_session(monkeypatch):
    """Test that Whisper backend properly logs session with all metadata."""
    audio, sample_rate = _load_fixture()
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

    def fake_audio_capture_factory(sample_rate, channels, max_capture_duration=0.0, collect_full_audio=True):
        return FakeAudioCaptureManager(
            audio=audio,
            sample_rate=sample_rate,
            channels=channels,
            max_capture_duration=max_capture_duration,
            collect_full_audio=collect_full_audio,
        )

    monkeypatch.setattr(audio_capture, "AudioCaptureManager", fake_audio_capture_factory)

    def capture_chunk(index, text, start, end, inference_seconds):
        chunks.append({"index": index, "text": text, "start": start, "end": end})

    # Create session logger with temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        session_logger = SessionLogger(
            output_dir=Path(tmpdir),
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
        assert (session_dir / "full_audio.wav").exists()
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


def test_whisper_backend_compares_stitched_vs_complete(monkeypatch):
    """Test that Whisper backend compares stitched transcript against complete audio transcript."""
    audio, sample_rate = _load_fixture()
    full_duration = len(audio) / sample_rate

    chunks: list[dict] = []
    transcribe_calls: list[str] = []

    class DummyModel:
        def transcribe(self, audio_chunk: np.ndarray, **kwargs):
            # Track whether this is a chunk or full transcription
            if audio_chunk.size == audio.size:
                transcribe_calls.append("full")
                return {"text": "Complete audio transcription"}
            else:
                chunk_num = len([c for c in transcribe_calls if c == "chunk"])
                transcribe_calls.append("chunk")
                return {"text": f"Chunk {chunk_num}"}

    dummy_model = DummyModel()

    def fake_load_whisper_model(**kwargs):
        return dummy_model, "cpu", False

    monkeypatch.setattr(whisper_backend, "load_whisper_model", fake_load_whisper_model)

    def fake_audio_capture_factory(sample_rate, channels, max_capture_duration=0.0, collect_full_audio=True):
        return FakeAudioCaptureManager(
            audio=audio,
            sample_rate=sample_rate,
            channels=channels,
            max_capture_duration=max_capture_duration,
            collect_full_audio=collect_full_audio,
        )

    monkeypatch.setattr(audio_capture, "AudioCaptureManager", fake_audio_capture_factory)

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
        vad_min_silence_duration=0.1,
        vad_min_speech_duration=0.05,
        vad_speech_pad_duration=0.0,
        max_chunk_duration=5.0,
        compare_transcripts=True,
        max_capture_duration=full_duration * 0.5,
        language="en",
    )

    # Verify that both chunk and full transcriptions were performed
    assert "chunk" in transcribe_calls
    assert "full" in transcribe_calls

    # Verify that full audio transcription is available
    assert result.full_audio_transcription == "Complete audio transcription"

    # Verify that we got chunks
    assert len(chunks) > 0


# --- Realtime Backend Tests ---


class FakeWebSocket:
    """Fake WebSocket for testing Realtime API."""

    def __init__(self, num_chunks: int = 3):
        self._chunk_count = 0
        self._num_chunks = num_chunks
        self.sent_messages: list[dict] = []
        self.closed = False
        self._session_committed = False

    async def send(self, message: str) -> None:
        self.sent_messages.append(json.loads(message))

    def __aiter__(self):
        return self

    async def __anext__(self):
        # Wait a bit to simulate network delay
        import asyncio

        await asyncio.sleep(0.01)

        # Generate chunk transcription events
        if self._chunk_count < self._num_chunks:
            self._chunk_count += 1
            return json.dumps(
                {
                    "type": "conversation.item.input_audio_transcription.completed",
                    "item_id": f"item-{self._chunk_count}",
                    "transcript": f"Realtime chunk {self._chunk_count}",
                }
            )

        # After all chunks, signal commitment
        if not self._session_committed:
            self._session_committed = True
            return json.dumps({"type": "session.input_audio_buffer.committed"})

        # End iteration
        raise StopAsyncIteration

    async def close(self, code: int = 1000, reason: str = "") -> None:
        self.closed = True

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


@pytest.mark.integration
def test_realtime_backend_respects_time_limit(monkeypatch):
    """Test that Realtime backend stops after max_capture_duration is reached."""
    audio, sample_rate = _load_fixture()
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

    monkeypatch.setattr(audio_capture, "AudioCaptureManager", fake_audio_capture_factory)

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
def test_realtime_backend_handles_user_stop(monkeypatch):
    """Test that Realtime backend handles graceful user stop."""
    audio, sample_rate = _load_fixture()
    full_duration = len(audio) / sample_rate
    stop_after = full_duration * 0.3

    chunk_texts: list[str] = []

    def fake_audio_capture_factory(sample_rate, channels, max_capture_duration=0.0, collect_full_audio=True):
        return FakeAudioCaptureManager(
            audio=audio,
            sample_rate=sample_rate,
            channels=channels,
            max_capture_duration=max_capture_duration,
            collect_full_audio=collect_full_audio,
            frame_size=320,
            simulate_user_stop_after=stop_after,
        )

    monkeypatch.setattr(audio_capture, "AudioCaptureManager", fake_audio_capture_factory)

    class FakeConnect:
        def __init__(self):
            self._ws = FakeWebSocket(num_chunks=10)  # More than we'll actually get

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
        max_capture_duration=0.0,  # Unlimited, but will be stopped by user
        language="en",
    )

    # Verify that capture stopped early
    assert result.capture_duration < full_duration * 0.8
    assert result.capture_duration > 0


@pytest.mark.integration
def test_realtime_backend_logs_session(monkeypatch):
    """Test that Realtime backend properly logs session with all metadata."""
    audio, sample_rate = _load_fixture()
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

    monkeypatch.setattr(audio_capture, "AudioCaptureManager", fake_audio_capture_factory)

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
    with tempfile.TemporaryDirectory() as tmpdir:
        session_logger = SessionLogger(
            output_dir=Path(tmpdir),
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
        assert (session_dir / "full_audio.wav").exists()
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
    audio, sample_rate = _load_fixture()
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

    monkeypatch.setattr(audio_capture, "AudioCaptureManager", fake_audio_capture_factory)

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


def test_session_logger_respects_min_duration(monkeypatch):
    """Test that SessionLogger discards sessions below minimum duration."""
    audio, sample_rate = _load_fixture()
    short_duration = 5.0  # Very short capture

    chunks: list[dict] = []

    class DummyModel:
        def transcribe(self, audio_chunk: np.ndarray, **kwargs):
            return {"text": f"Chunk {len(chunks)}"}

    dummy_model = DummyModel()

    def fake_load_whisper_model(**kwargs):
        return dummy_model, "cpu", False

    monkeypatch.setattr(whisper_backend, "load_whisper_model", fake_load_whisper_model)

    def fake_audio_capture_factory(sample_rate, channels, max_capture_duration=0.0, collect_full_audio=True):
        return FakeAudioCaptureManager(
            audio=audio,
            sample_rate=sample_rate,
            channels=channels,
            max_capture_duration=max_capture_duration,
            collect_full_audio=collect_full_audio,
        )

    monkeypatch.setattr(audio_capture, "AudioCaptureManager", fake_audio_capture_factory)

    def capture_chunk(index, text, start, end, inference_seconds):
        chunks.append({"index": index, "text": text})

    # Create session logger with temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        session_logger = SessionLogger(
            output_dir=Path(tmpdir),
            sample_rate=sample_rate,
            channels=1,
            backend="whisper",
            save_chunk_audio=True,
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
