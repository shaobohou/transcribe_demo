"""Unified audio capture module for both Whisper and Realtime backends."""

from __future__ import annotations

import queue
import sys
import threading
import types
from ctypes import Structure
from typing import TYPE_CHECKING, Any, Self

import numpy as np

if TYPE_CHECKING:
    import sounddevice as sd
else:
    sd = None  # Lazy import when needed


class AudioCaptureManager:
    """
    Manages audio capture from microphone with support for:
    - Queued audio delivery to worker threads
    - Capture duration limits
    - Full audio collection for comparison
    - Thread-safe stop coordination
    """

    def __init__(
        self,
        sample_rate: int,
        channels: int,
        max_capture_duration: float = 0.0,
        collect_full_audio: bool = True,
    ):
        """
        Initialize the audio capture manager.

        Args:
            sample_rate: Audio sample rate in Hz
            channels: Number of audio channels
            max_capture_duration: Maximum duration to capture in seconds (0 = unlimited)
            collect_full_audio: Whether to collect full audio for comparison
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.max_capture_duration = max_capture_duration
        self.collect_full_audio = collect_full_audio

        # Queue for passing audio chunks to worker threads
        self.audio_queue: queue.Queue[np.ndarray | None] = queue.Queue()

        # Synchronization primitives
        self.stop_event = threading.Event()
        self.capture_limit_reached = threading.Event()
        self._lock = threading.Lock()

        # Full audio collection for comparison/logging
        self._full_audio_chunks: list[np.ndarray] = []
        self._total_samples_captured = 0
        self._max_capture_samples = int(sample_rate * max_capture_duration) if max_capture_duration > 0 else 0

        # Input stream reference
        self._stream: Any = None  # sd.InputStream when initialized
        self._stdin_thread: threading.Thread | None = None

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time: Structure,
        status: Any,
    ) -> None:
        """Audio callback invoked by sounddevice for each audio chunk."""
        if status:
            print(f"InputStream status: {status}", file=sys.stderr)

        # Don't capture if we've stopped or reached limit
        if self.stop_event.is_set() or self.capture_limit_reached.is_set():
            return

        # Copy to avoid referencing the sounddevice ring buffer
        audio_chunk = indata.copy()
        self.audio_queue.put(audio_chunk)

        # Convert to mono for tracking and optional storage
        mono = audio_chunk.astype(np.float32, copy=False)
        if mono.ndim > 1:
            mono = mono.mean(axis=1)
        else:
            mono = mono.reshape(-1)

        with self._lock:
            # Always track total samples captured (for duration limit)
            self._total_samples_captured += len(mono)

            # Check if capture duration limit reached
            if self._max_capture_samples > 0 and self._total_samples_captured >= self._max_capture_samples:
                if not self.capture_limit_reached.is_set():
                    self.capture_limit_reached.set()
                    duration_captured = self._total_samples_captured / self.sample_rate
                    print(
                        f"\nCapture duration limit reached ({duration_captured:.1f}s). "
                        f"Finishing current processing and stopping gracefully...",
                        file=sys.stderr,
                    )
                    # Signal end of stream with None sentinel
                    self.audio_queue.put(None)
                    # Stop further capture so waiters unblock
                    self.stop()

            # Preserve full-session audio for post-run transcription comparison (if enabled)
            if self.collect_full_audio:
                self._full_audio_chunks.append(mono.copy())
                # Trim to max_capture_duration by removing oldest chunks
                if self._max_capture_samples > 0:
                    total_samples = sum(len(c) for c in self._full_audio_chunks)
                    while total_samples > self._max_capture_samples and len(self._full_audio_chunks) > 1:
                        removed = self._full_audio_chunks.pop(0)
                        total_samples -= len(removed)

    def _stdin_listener(self) -> None:
        """Wait for user to press Enter, then request shutdown."""
        try:
            input("Press Enter to stop...\n")
        except (EOFError, OSError):
            # stdin unavailable (e.g., running in background); rely on capture limit or Ctrl+C
            return
        print("\nStopping due to user request.", file=sys.stderr)
        self.stop()

    def start(self) -> None:
        """Start audio capture from microphone."""
        # Lazy import sounddevice only when actually needed
        global sd
        if sd is None:
            import sounddevice as sd_module
            sd = sd_module

        # Start stdin listener if available
        if sys.stdin is not None and sys.stdin.isatty():
            self._stdin_thread = threading.Thread(target=self._stdin_listener, name="stdin-listener", daemon=True)
            self._stdin_thread.start()
        else:
            print("stdin is not interactive; press Ctrl+C to stop recording.", file=sys.stderr)

        # Open audio input stream
        self._stream = sd.InputStream(
            callback=self._audio_callback,
            channels=self.channels,
            samplerate=self.sample_rate,
            dtype="float32",
        )
        self._stream.start()

    def stop(self) -> None:
        """Signal that audio capture should stop."""
        self.stop_event.set()
        # Wake up any threads waiting on the queue
        try:
            self.audio_queue.put_nowait(None)
        except queue.Full:
            pass

    def wait_until_stopped(self) -> None:
        """Block until the stop event is set."""
        while not self.stop_event.is_set():
            import time

            time.sleep(0.1)

    def close(self) -> None:
        """Close the audio stream and clean up resources."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def get_full_audio(self) -> np.ndarray:
        """
        Get the complete captured audio as a single numpy array.

        Returns:
            Mono float32 audio array containing all captured audio
        """
        with self._lock:
            if not self._full_audio_chunks:
                return np.zeros(0, dtype=np.float32)
            return np.concatenate(self._full_audio_chunks)

    def get_capture_duration(self) -> float:
        """
        Get the total duration of audio captured so far.

        Returns:
            Duration in seconds
        """
        with self._lock:
            return self._total_samples_captured / self.sample_rate

    def __enter__(self) -> Self:
        """Context manager entry."""
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> bool:
        """Context manager exit."""
        self.stop()
        self.close()
        return False
