"""File-based audio source for simulating live transcription from offline audio files."""

from __future__ import annotations

import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf


class FileAudioSource:
    """
    Simulates live audio capture by reading from an audio file and feeding chunks in real-time.

    This class implements the same interface as AudioCaptureManager, making it a drop-in
    replacement for simulating live transcription from pre-recorded audio files.

    Supported formats: WAV, FLAC, MP3 (if libsndfile has MP3 support), OGG, and more.
    """

    def __init__(
        self,
        audio_file: Path,
        sample_rate: int,
        channels: int,
        max_capture_duration: float = 0.0,
        collect_full_audio: bool = True,
        playback_speed: float = 1.0,
    ):
        """
        Initialize the file audio source.

        Args:
            audio_file: Path to the audio file to read
            sample_rate: Target sample rate (audio will be resampled if needed)
            channels: Number of audio channels
            max_capture_duration: Maximum duration to read from file in seconds (0 = entire file)
            collect_full_audio: Whether to collect full audio for comparison
            playback_speed: Speed multiplier for playback (1.0 = real-time, 2.0 = 2x speed, etc.)
        """
        self.audio_file = Path(audio_file)
        self.sample_rate = sample_rate
        self.channels = channels
        self.max_capture_duration = max_capture_duration
        self.collect_full_audio = collect_full_audio
        self.playback_speed = playback_speed

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

        # Playback thread reference
        self._playback_thread: threading.Thread | None = None

        # Load and prepare audio
        self._loaded_audio: np.ndarray | None = None
        self._loaded_sample_rate: int = 0
        self._load_audio()

    def _load_audio(self) -> None:
        """Load audio file and prepare for playback."""
        if not self.audio_file.exists():
            raise FileNotFoundError(f"Audio file not found: {self.audio_file}")

        try:
            # Load audio using soundfile
            audio, file_sample_rate = sf.read(str(self.audio_file), dtype="float32", always_2d=False)

            print(
                f"Loaded audio file: {self.audio_file.name} "
                f"({len(audio) / file_sample_rate:.2f}s @ {file_sample_rate}Hz)",
                file=sys.stderr,
            )

            # Convert to mono if needed
            if audio.ndim > 1:
                audio = audio.mean(axis=1).astype(np.float32)

            # Resample if needed (simple linear interpolation)
            if file_sample_rate != self.sample_rate:
                print(
                    f"Resampling from {file_sample_rate}Hz to {self.sample_rate}Hz...",
                    file=sys.stderr,
                )
                audio = self._resample(audio, file_sample_rate, self.sample_rate)

            # Limit to max_capture_duration if specified
            if self._max_capture_samples > 0 and len(audio) > self._max_capture_samples:
                audio = audio[: self._max_capture_samples]
                print(
                    f"Limiting audio to {self.max_capture_duration:.1f}s",
                    file=sys.stderr,
                )

            self._loaded_audio = audio
            self._loaded_sample_rate = self.sample_rate

        except Exception as e:
            raise RuntimeError(f"Failed to load audio file {self.audio_file}: {e}") from e

    def _resample(self, audio: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
        """
        Resample audio using linear interpolation.

        Args:
            audio: Input audio array
            from_rate: Original sample rate
            to_rate: Target sample rate

        Returns:
            Resampled audio array
        """
        if from_rate == to_rate:
            return audio

        # Calculate new length
        duration = len(audio) / from_rate
        new_length = int(duration * to_rate)

        # Linear interpolation
        old_indices = np.linspace(0, len(audio) - 1, new_length)
        resampled = np.interp(old_indices, np.arange(len(audio)), audio)

        return resampled.astype(np.float32)

    def _playback_loop(self) -> None:
        """Feed audio chunks into the queue in real-time."""
        if self._loaded_audio is None:
            return

        # Standard frame size (30ms at 16kHz = 480 samples)
        frame_size = 480
        frame_duration = frame_size / self.sample_rate  # Duration of each frame in seconds
        sleep_duration = frame_duration / self.playback_speed  # Adjust for playback speed

        audio = self._loaded_audio
        total_frames = (len(audio) + frame_size - 1) // frame_size

        print(
            f"Starting playback simulation ({total_frames} frames @ {self.playback_speed}x speed)...",
            file=sys.stderr,
        )
        if self.playback_speed != 1.0:
            print(f"Note: Playback speed is {self.playback_speed}x", file=sys.stderr)

        start_time = time.time()

        for frame_idx in range(total_frames):
            if self.stop_event.is_set():
                break

            # Extract frame
            start = frame_idx * frame_size
            end = min(start + frame_size, len(audio))
            frame = audio[start:end]

            if len(frame) == 0:
                break

            # Reshape to match expected format (samples, channels)
            if self.channels == 1:
                frame_shaped = frame.reshape(-1, 1)
            else:
                # For multi-channel, just duplicate the mono signal
                frame_shaped = np.tile(frame.reshape(-1, 1), (1, self.channels))

            # Put frame in queue
            self.audio_queue.put(frame_shaped)

            # Track samples and check duration limit
            with self._lock:
                self._total_samples_captured += len(frame)

                # Collect for get_full_audio
                if self.collect_full_audio:
                    self._full_audio_chunks.append(frame.copy())

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
                        self.stop()
                        return

            # Sleep to simulate real-time playback
            # We sleep AFTER putting the frame to avoid initial delay
            expected_time = (frame_idx + 1) * sleep_duration
            elapsed = time.time() - start_time
            sleep_time = expected_time - elapsed

            if sleep_time > 0:
                time.sleep(sleep_time)

        # Signal end of stream
        print(
            f"\nReached end of audio file ({self._total_samples_captured / self.sample_rate:.1f}s). "
            f"Finishing processing...",
            file=sys.stderr,
        )
        self.audio_queue.put(None)
        self.stop()

    def _stdin_listener(self) -> None:
        """Wait for user to press Enter, then request shutdown."""
        try:
            input("Press Enter to stop...\n")
        except (EOFError, OSError):
            # stdin unavailable (e.g., running in background)
            return
        print("\nStopping due to user request.", file=sys.stderr)
        self.stop()

    def start(self) -> None:
        """Start playback simulation."""
        # Start stdin listener if available
        if sys.stdin is not None and sys.stdin.isatty():
            stdin_thread = threading.Thread(target=self._stdin_listener, name="stdin-listener", daemon=True)
            stdin_thread.start()
        else:
            print("stdin is not interactive; press Ctrl+C to stop playback.", file=sys.stderr)

        # Start playback thread
        self._playback_thread = threading.Thread(target=self._playback_loop, name="file-playback", daemon=True)
        self._playback_thread.start()

    def stop(self) -> None:
        """Signal that playback should stop."""
        self.stop_event.set()
        # Wake up any threads waiting on the queue
        try:
            self.audio_queue.put_nowait(None)
        except queue.Full:
            pass

    def wait_until_stopped(self) -> None:
        """Block until the stop event is set."""
        while not self.stop_event.is_set():
            time.sleep(0.1)

    def close(self) -> None:
        """Clean up resources."""
        if self._playback_thread is not None and self._playback_thread.is_alive():
            self._playback_thread.join(timeout=2.0)

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

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        self.close()
        return False
