"""Tests for file audio source functionality."""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import soundfile as sf

from transcribe_demo.file_audio_source import FileAudioSource


def _create_test_audio_file(path: Path, duration: float, sample_rate: int) -> None:
    """Create a test audio file with sine wave."""
    t = np.linspace(0, duration, int(sample_rate * duration))
    # Generate a 440 Hz sine wave (A4 note)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    sf.write(str(path), audio, sample_rate)


def test_file_audio_source_basic() -> None:
    """Test basic file audio source functionality."""
    with TemporaryDirectory() as tmpdir:
        # Create a 2-second test audio file
        audio_file = Path(tmpdir) / "test.wav"
        sample_rate = 16000
        duration = 2.0
        _create_test_audio_file(audio_file, duration, sample_rate)

        # Create file audio source
        source = FileAudioSource(
            audio_file=audio_file,
            sample_rate=sample_rate,
            channels=1,
            max_capture_duration=0.0,
            collect_full_audio=True,
            playback_speed=10.0,  # Speed up for faster test
        )

        # Start playback
        source.start()

        # Consume audio chunks
        chunks_received = 0
        while not source.stop_event.is_set():
            try:
                chunk = source.audio_queue.get(timeout=1.0)
                if chunk is None:
                    break
                chunks_received += 1
            except Exception:
                break

        source.close()

        # Verify we received chunks
        assert chunks_received > 0, "Should have received at least one audio chunk"

        # Verify full audio collection
        full_audio = source.get_full_audio()
        assert len(full_audio) > 0, "Full audio should be collected"

        # Verify duration is approximately correct (within 10%)
        captured_duration = source.get_capture_duration()
        assert abs(captured_duration - duration) / duration < 0.1, (
            f"Captured duration {captured_duration:.2f}s should be close to {duration:.2f}s"
        )


def test_file_audio_source_duration_limit() -> None:
    """Test that max_capture_duration limit works correctly."""
    with TemporaryDirectory() as tmpdir:
        # Create a 4-second test audio file
        audio_file = Path(tmpdir) / "test.wav"
        sample_rate = 16000
        full_duration = 4.0
        limit_duration = 2.0
        _create_test_audio_file(audio_file, full_duration, sample_rate)

        # Create file audio source with duration limit
        source = FileAudioSource(
            audio_file=audio_file,
            sample_rate=sample_rate,
            channels=1,
            max_capture_duration=limit_duration,
            collect_full_audio=True,
            playback_speed=10.0,  # Speed up for faster test
        )

        # Start playback
        source.start()

        # Wait for stop event
        source.wait_until_stopped()
        source.close()

        # Verify capture limit was respected
        captured_duration = source.get_capture_duration()
        assert captured_duration <= limit_duration * 1.1, (
            f"Captured duration {captured_duration:.2f}s should not exceed limit {limit_duration:.2f}s by much"
        )
        assert source.capture_limit_reached.is_set(), "Capture limit should be reached"


def test_file_audio_source_resampling() -> None:
    """Test that resampling works when file sample rate differs from target."""
    with TemporaryDirectory() as tmpdir:
        # Create a test audio file at 8kHz
        audio_file = Path(tmpdir) / "test.wav"
        file_sample_rate = 8000
        target_sample_rate = 16000
        duration = 1.0
        _create_test_audio_file(audio_file, duration, file_sample_rate)

        # Create file audio source with different sample rate
        source = FileAudioSource(
            audio_file=audio_file,
            sample_rate=target_sample_rate,
            channels=1,
            max_capture_duration=0.0,
            collect_full_audio=True,
            playback_speed=10.0,  # Speed up for faster test
        )

        # Start and consume audio
        source.start()
        source.wait_until_stopped()
        source.close()

        # Verify audio was resampled
        full_audio = source.get_full_audio()
        expected_samples = int(target_sample_rate * duration)

        # Allow for some variation due to resampling and frame boundaries
        assert abs(len(full_audio) - expected_samples) / expected_samples < 0.1, (
            f"Resampled audio length {len(full_audio)} should be close to {expected_samples}"
        )


def test_file_audio_source_nonexistent_file() -> None:
    """Test that nonexistent file raises appropriate error."""
    try:
        FileAudioSource(
            audio_file=Path("/nonexistent/path/to/file.wav"),
            sample_rate=16000,
            channels=1,
        )
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError as e:
        assert "not found" in str(e).lower()
