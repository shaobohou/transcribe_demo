"""Pytest configuration and fixtures for transcribe_demo tests.

This module mocks sounddevice at import time to prevent audio device access
in CI environments where no audio hardware is available.
"""

from __future__ import annotations

import sys
import tempfile
import types
from pathlib import Path

import pytest

# Mock sounddevice BEFORE any test modules are imported
# This must happen at module level (not in a fixture) because pytest imports
# test modules during collection, before any fixtures run. When test modules
# import from transcribe_demo, audio_capture.py tries to import sounddevice,
# which fails in CI environments without audio hardware.


# Mock InputStream class - needs to be a class that can be instantiated
class MockInputStream:
    """Mock sounddevice InputStream to prevent audio device access in tests."""

    def __init__(self, *args, **kwargs):
        # Store callback for potential inspection
        self.callback = kwargs.get('callback', None)
        self.channels = kwargs.get('channels', 1)
        self.samplerate = kwargs.get('samplerate', 16000)
        self.dtype = kwargs.get('dtype', 'float32')

    def start(self):
        """Mock start - does nothing."""
        pass

    def stop(self):
        """Mock stop - does nothing."""
        pass

    def close(self):
        """Mock close - does nothing."""
        pass

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        self.close()
        return False


# Mock PortAudioError exception - this needs to be a real exception class
class MockPortAudioError(Exception):
    """Mock exception for PortAudio errors."""
    pass


# Create a simple fake module without using unittest.mock
# This avoids any potential issues with Mock object behavior
class FakeSoundDeviceModule(types.ModuleType):
    """Fake sounddevice module for testing without audio hardware."""

    def __init__(self):
        super().__init__('sounddevice')
        self.InputStream = MockInputStream
        self.PortAudioError = MockPortAudioError

    def query_devices(self, *args, **kwargs):
        """Mock query_devices - returns empty list."""
        return []

    def default_device(self, *args, **kwargs):
        """Mock default device query - returns None."""
        return None


# Inject the mock into sys.modules at module level
# This happens immediately when conftest.py is imported by pytest, before test
# collection begins. This ensures sounddevice is already mocked when test modules
# import backend code (which imports audio_capture, which imports sounddevice).
mock_sd = FakeSoundDeviceModule()
sys.modules['sounddevice'] = mock_sd
sys.modules['sd'] = mock_sd  # Also mock 'sd' in case it's imported differently

# NOTE: We do NOT replace AudioCaptureManager here. Backend modules import the
# audio_capture module (not the class directly) with:
#   from transcribe_demo import audio_capture as audio_capture_lib
# This allows tests to monkeypatch audio_capture_lib.AudioCaptureManager with
# FakeAudioCaptureManager. With sounddevice mocked above, the real AudioCaptureManager
# can be imported safely without triggering audio hardware access.


@pytest.fixture(autouse=True)
def mock_stdin_for_audio_capture(monkeypatch):
    """
    Ensure stdin.isatty() returns False to prevent stdin listener threads.

    This prevents AudioCaptureManager from starting a stdin listener thread
    that would block on input() in CI environments. Applies to all tests
    automatically via autouse=True.
    """
    # Create a mock stdin that always reports it's not a TTY
    class MockStdin:
        def isatty(self):
            return False

    monkeypatch.setattr(sys, 'stdin', MockStdin())


@pytest.fixture
def temp_session_dir():
    """
    Provide a temporary directory for session logging in tests.

    The directory is automatically cleaned up after the test completes,
    ensuring no session logs are left on disk after tests run.

    Usage:
        def test_something(temp_session_dir):
            session_logger = SessionLogger(
                output_dir=temp_session_dir,
                ...
            )
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# Shared test fixtures and helpers


import json
import queue as queue_module
import threading
import time
import wave

import numpy as np


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

    def _feed_audio(self) -> None:
        """Feed test audio into queue in a background thread."""
        fed_samples = 0
        for start in range(0, len(self._audio), self._frame_size):
            if self.stop_event.is_set():
                break

            # Check max_capture_duration and set flag, but continue feeding
            # This matches real AudioCaptureManager behavior - it signals the limit
            # but continues until backend explicitly calls stop()
            if self.max_capture_duration > 0:
                samples_duration = fed_samples / self.sample_rate
                if samples_duration >= self.max_capture_duration:
                    self.capture_limit_reached.set()
                    # Don't break - backend needs time to process queued audio

            frame = self._audio[start : start + self._frame_size]
            if not frame.size:
                continue

            # Reshape to match expected format (samples, channels)
            frame_shaped = frame.reshape(-1, 1) if self.channels == 1 else frame
            self.audio_queue.put(frame_shaped)

            # Collect for get_full_audio (only up to max_capture_duration)
            if self.collect_full_audio and not self.capture_limit_reached.is_set():
                mono = frame_shaped.mean(axis=1).astype(np.float32) if frame_shaped.ndim > 1 else frame_shaped
                self._full_audio_chunks.append(mono)

            fed_samples += len(frame)

            # Small delay to allow backend to process frames
            # Without this, audio feeds too fast and backend doesn't have time to process
            if self.capture_limit_reached.is_set():
                time.sleep(0.001)  # 1ms delay after limit reached to allow backend to stop

        # Signal end of stream
        self.audio_queue.put(None)
        # Set stop_event so wait_until_stopped() can return
        self.stop_event.set()

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


def load_test_fixture() -> tuple[np.ndarray, int]:
    """Load test audio fixture (fox.wav)."""
    fixture = Path(__file__).resolve().parent / "data" / "fox.wav"
    if not fixture.exists():
        raise FileNotFoundError("tests/data/fox.wav fixture not found")
    with wave.open(str(fixture), "rb") as wf:
        if wf.getnchannels() != 1:
            raise RuntimeError("fox.wav must be mono")
        rate = wf.getframerate()
        audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32)
    return audio / 32768.0, rate


def generate_synthetic_audio(duration_seconds: float = 3.0, sample_rate: int = 16000) -> tuple[np.ndarray, int]:
    """
    Generate synthetic audio for faster tests.

    Creates a simple audio signal with varying amplitude to simulate speech-like patterns
    that will trigger VAD detection.

    Args:
        duration_seconds: Length of audio to generate
        sample_rate: Sample rate in Hz

    Returns:
        Tuple of (audio_array, sample_rate)
    """
    num_samples = int(duration_seconds * sample_rate)
    t = np.linspace(0, duration_seconds, num_samples, dtype=np.float32)

    # Create a signal with varying amplitude to simulate speech patterns
    # Mix of low frequency (simulating speech) with amplitude modulation
    carrier = np.sin(2 * np.pi * 200 * t)  # 200 Hz base frequency
    modulation = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)  # 3 Hz amplitude variation
    audio = carrier * modulation * 0.3  # Scale to reasonable amplitude

    return audio.astype(np.float32), sample_rate
