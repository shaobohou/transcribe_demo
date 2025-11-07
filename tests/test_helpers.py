"""Shared test helpers and fixtures for transcribe_demo tests."""

from __future__ import annotations

import json
import queue as queue_module
import threading
import time
import wave
from pathlib import Path

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
        limit_reached_this_iteration = False
        for start in range(0, len(self._audio), self._frame_size):
            if self.stop_event.is_set():
                break

            frame = self._audio[start : start + self._frame_size]
            if not frame.size:
                continue

            # Reshape to match expected format (samples, channels)
            frame_shaped = frame.reshape(-1, 1) if self.channels == 1 else frame

            # Put frame in queue FIRST (matches real AudioCaptureManager behavior)
            self.audio_queue.put(frame_shaped)

            # Collect for get_full_audio
            if self.collect_full_audio:
                mono = frame_shaped.mean(axis=1).astype(np.float32) if frame_shaped.ndim > 1 else frame_shaped
                self._full_audio_chunks.append(mono)

            # Track samples AFTER putting in queue
            fed_samples += len(frame)

            # Check max_capture_duration AFTER feeding frame (matches real behavior)
            # The chunk that causes timeout IS included in the queue
            if self.max_capture_duration > 0 and not self.capture_limit_reached.is_set():
                samples_duration = fed_samples / self.sample_rate
                if samples_duration >= self.max_capture_duration:
                    self.capture_limit_reached.set()
                    limit_reached_this_iteration = True
                    # Signal end of stream with None sentinel (matches real behavior)
                    self.audio_queue.put(None)
                    # Stop further feeding (real AudioCaptureManager stops immediately)
                    self.stop()
                    break

        # Signal end of stream if we finished naturally (not due to timeout)
        if not limit_reached_this_iteration and not self.stop_event.is_set():
            self.audio_queue.put(None)
            # Give backend time to process queued audio before signaling stop
            # This is critical for realtime backend which processes audio asynchronously
            time.sleep(0.5)  # Allow async processing to complete
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


def create_fake_audio_capture_factory(audio: np.ndarray, sample_rate: int, frame_size: int = 480):
    """
    Create a factory function for FakeAudioCaptureManager.

    This helper reduces boilerplate when monkeypatching AudioCaptureManager in tests.

    Args:
        audio: Audio data to feed through the fake manager
        sample_rate: Sample rate of the audio
        frame_size: Frame size for chunking (default 480 for Whisper, use 320 for Realtime)

    Returns:
        Factory function that creates FakeAudioCaptureManager instances

    Example:
        >>> audio, sample_rate = generate_synthetic_audio()
        >>> factory = create_fake_audio_capture_factory(audio, sample_rate)
        >>> monkeypatch.setattr("transcribe_demo.audio_capture.AudioCaptureManager", factory)
    """

    def factory(sample_rate, channels, max_capture_duration=0.0, collect_full_audio=True):
        return FakeAudioCaptureManager(
            audio=audio,
            sample_rate=sample_rate,
            channels=channels,
            max_capture_duration=max_capture_duration,
            collect_full_audio=collect_full_audio,
            frame_size=frame_size,
        )

    return factory
