"""Tests for file audio source functionality."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

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


def test_file_audio_source_url_detection() -> None:
    """Test that URL detection works correctly."""
    with TemporaryDirectory() as tmpdir:
        # Create a test audio file
        audio_file = Path(tmpdir) / "test.wav"
        _create_test_audio_file(audio_file, 1.0, 16000)

        # Use a temporary cache directory
        temp_cache_dir = Path(tmpdir) / "cache"
        temp_cache_dir.mkdir(parents=True, exist_ok=True)

        # Mock urlopen to avoid actual network requests
        with (
            patch("transcribe_demo.file_audio_source.urlopen") as mock_urlopen,
            patch.object(FileAudioSource, "_get_cache_dir", return_value=temp_cache_dir),
        ):
            # Create a mock response
            mock_response = MagicMock()
            mock_response.headers.get.return_value = "0"

            # Read the actual audio file content
            with open(audio_file, "rb") as f:
                audio_content = f.read()

            # Make the mock return the audio content
            mock_response.read.side_effect = [audio_content, b""]
            mock_response.__enter__.return_value = mock_response
            mock_response.__exit__.return_value = False
            mock_urlopen.return_value = mock_response

            # Test with HTTP URL
            source = FileAudioSource(
                audio_file="http://example.com/audio.mp3",
                sample_rate=16000,
                channels=1,
                playback_speed=10.0,
            )

            # Verify URL was detected and download was attempted
            mock_urlopen.assert_called_once()
            call_args = mock_urlopen.call_args[0]
            assert call_args[0] == "http://example.com/audio.mp3"

            # Verify audio was loaded
            assert source._loaded_audio is not None
            assert len(source._loaded_audio) > 0

            # Clean up
            source.close()


def test_file_audio_source_url_with_string_path() -> None:
    """Test that string paths work as well as Path objects."""
    with TemporaryDirectory() as tmpdir:
        # Create a test audio file
        audio_file = Path(tmpdir) / "test.wav"
        _create_test_audio_file(audio_file, 1.0, 16000)

        # Test with string path
        source = FileAudioSource(
            audio_file=str(audio_file),
            sample_rate=16000,
            channels=1,
            playback_speed=10.0,
        )

        # Verify audio was loaded
        assert source._loaded_audio is not None
        assert len(source._loaded_audio) > 0

        # Clean up
        source.close()


def test_url_not_corrupted_by_path() -> None:
    """
    Test that URLs are not corrupted when passed as strings.

    Regression test for bug where Path(url) would normalize 'http://' to 'http:/'
    by collapsing the double slash.
    """
    # Test case 1: Verify that Path() corrupts URLs (the bug)
    test_url = "http://public.npr.org/test.mp3"
    corrupted = str(Path(test_url))
    assert "//" not in corrupted or corrupted == test_url, f"Path() should corrupt double-slash in URL: {corrupted}"
    # On most systems, Path normalizes http:// to http:/
    if corrupted != test_url:
        assert "http:/" in corrupted and "http://" not in corrupted

    # Test case 2: Verify FileAudioSource._is_url works correctly
    with TemporaryDirectory() as tmpdir:
        audio_file = Path(tmpdir) / "test.wav"
        _create_test_audio_file(audio_file, 1.0, 16000)

        # Use a temporary cache directory
        temp_cache_dir = Path(tmpdir) / "cache"
        temp_cache_dir.mkdir(parents=True, exist_ok=True)

        with (
            patch("transcribe_demo.file_audio_source.urlopen") as mock_urlopen,
            patch.object(FileAudioSource, "_get_cache_dir", return_value=temp_cache_dir),
        ):
            # Create a mock response
            mock_response = MagicMock()
            mock_response.headers.get.return_value = "0"

            # Read the actual audio file content
            with open(audio_file, "rb") as f:
                audio_content = f.read()

            mock_response.read.side_effect = [audio_content, b""]
            mock_response.__enter__.return_value = mock_response
            mock_response.__exit__.return_value = False
            mock_urlopen.return_value = mock_response

            # Test with URL as string (correct approach)
            source = FileAudioSource(
                audio_file=test_url,  # Pass as string, NOT Path(test_url)
                sample_rate=16000,
                channels=1,
                playback_speed=10.0,
            )

            # Verify URL detection worked
            assert source._is_url(path=test_url), "URL should be detected"

            # Verify the exact URL was passed to urlopen (not corrupted)
            mock_urlopen.assert_called_once()
            actual_url = mock_urlopen.call_args[0][0]
            assert actual_url == test_url, f"URL should not be corrupted: expected '{test_url}', got '{actual_url}'"
            assert "http://" in actual_url, "Double slash should be preserved"

            source.close()


def test_url_caching_cache_miss() -> None:
    """Test that first URL download creates a cache entry."""
    with TemporaryDirectory() as tmpdir:
        # Create a test audio file
        audio_file = Path(tmpdir) / "test.wav"
        _create_test_audio_file(audio_file, 1.0, 16000)

        # Use a temporary cache directory
        temp_cache_dir = Path(tmpdir) / "cache"
        temp_cache_dir.mkdir(parents=True, exist_ok=True)

        with (
            patch("transcribe_demo.file_audio_source.urlopen") as mock_urlopen,
            patch.object(FileAudioSource, "_get_cache_dir", return_value=temp_cache_dir),
        ):
            # Create a mock response
            mock_response = MagicMock()
            mock_response.headers.get.return_value = "0"

            # Read the actual audio file content
            with open(audio_file, "rb") as f:
                audio_content = f.read()

            mock_response.read.side_effect = [audio_content, b""]
            mock_response.__enter__.return_value = mock_response
            mock_response.__exit__.return_value = False
            mock_urlopen.return_value = mock_response

            # Create file audio source with URL
            test_url = "http://example.com/test.mp3"
            source = FileAudioSource(
                audio_file=test_url,
                sample_rate=16000,
                channels=1,
                playback_speed=10.0,
            )

            # Verify URL was downloaded (cache miss)
            mock_urlopen.assert_called_once()

            # Verify cache file was created
            cache_key = source._get_cache_key(url=test_url)
            cache_file = temp_cache_dir / f"{cache_key}.mp3"
            assert cache_file.exists(), "Cache file should be created"

            # Verify audio was loaded
            assert source._loaded_audio is not None
            assert len(source._loaded_audio) > 0

            source.close()


def test_url_caching_cache_hit() -> None:
    """Test that second URL download uses cache and doesn't re-download."""
    with TemporaryDirectory() as tmpdir:
        # Create a test audio file
        audio_file = Path(tmpdir) / "test.wav"
        _create_test_audio_file(audio_file, 1.0, 16000)

        # Use a temporary cache directory
        temp_cache_dir = Path(tmpdir) / "cache"
        temp_cache_dir.mkdir(parents=True)

        test_url = "http://example.com/test.mp3"

        # First download - populate cache
        with (
            patch("transcribe_demo.file_audio_source.urlopen") as mock_urlopen,
            patch.object(FileAudioSource, "_get_cache_dir", return_value=temp_cache_dir),
        ):
            # Create a mock response
            mock_response = MagicMock()
            mock_response.headers.get.return_value = "0"

            # Read the actual audio file content
            with open(audio_file, "rb") as f:
                audio_content = f.read()

            mock_response.read.side_effect = [audio_content, b""]
            mock_response.__enter__.return_value = mock_response
            mock_response.__exit__.return_value = False
            mock_urlopen.return_value = mock_response

            # Create first source
            source1 = FileAudioSource(
                audio_file=test_url,
                sample_rate=16000,
                channels=1,
                playback_speed=10.0,
            )

            # Verify first download happened
            assert mock_urlopen.call_count == 1
            source1.close()

        # Second download - should use cache
        with (
            patch("transcribe_demo.file_audio_source.urlopen") as mock_urlopen_2,
            patch.object(FileAudioSource, "_get_cache_dir", return_value=temp_cache_dir),
        ):
            # Create second source with same URL
            source2 = FileAudioSource(
                audio_file=test_url,
                sample_rate=16000,
                channels=1,
                playback_speed=10.0,
            )

            # Verify NO download happened (cache hit)
            mock_urlopen_2.assert_not_called()

            # Verify audio was loaded from cache
            assert source2._loaded_audio is not None
            assert len(source2._loaded_audio) > 0

            source2.close()


def test_url_caching_different_urls_different_cache() -> None:
    """Test that different URLs create different cache entries."""
    with TemporaryDirectory() as tmpdir:
        # Create test audio files
        audio_file1 = Path(tmpdir) / "test1.wav"
        audio_file2 = Path(tmpdir) / "test2.wav"
        _create_test_audio_file(audio_file1, 1.0, 16000)
        _create_test_audio_file(audio_file2, 2.0, 16000)

        # Use a temporary cache directory
        temp_cache_dir = Path(tmpdir) / "cache"
        temp_cache_dir.mkdir(parents=True, exist_ok=True)

        test_url1 = "http://example.com/test1.mp3"
        test_url2 = "http://example.com/test2.mp3"

        with (
            patch("transcribe_demo.file_audio_source.urlopen") as mock_urlopen,
            patch.object(FileAudioSource, "_get_cache_dir", return_value=temp_cache_dir),
        ):

            def mock_urlopen_side_effect(url, timeout=None):
                """Return different content based on URL."""
                mock_response = MagicMock()
                mock_response.headers.get.return_value = "0"

                if "test1" in url:
                    with open(audio_file1, "rb") as f:
                        content = f.read()
                else:
                    with open(audio_file2, "rb") as f:
                        content = f.read()

                mock_response.read.side_effect = [content, b""]
                mock_response.__enter__.return_value = mock_response
                mock_response.__exit__.return_value = False
                return mock_response

            mock_urlopen.side_effect = mock_urlopen_side_effect

            # Download first URL
            source1 = FileAudioSource(
                audio_file=test_url1,
                sample_rate=16000,
                channels=1,
                playback_speed=10.0,
            )

            # Download second URL
            source2 = FileAudioSource(
                audio_file=test_url2,
                sample_rate=16000,
                channels=1,
                playback_speed=10.0,
            )

            # Verify both URLs were downloaded
            assert mock_urlopen.call_count == 2

            # Verify different cache files were created
            cache_key1 = source1._get_cache_key(url=test_url1)
            cache_key2 = source2._get_cache_key(url=test_url2)
            assert cache_key1 != cache_key2, "Different URLs should have different cache keys"

            cache_file1 = temp_cache_dir / f"{cache_key1}.mp3"
            cache_file2 = temp_cache_dir / f"{cache_key2}.mp3"
            assert cache_file1.exists(), "First cache file should exist"
            assert cache_file2.exists(), "Second cache file should exist"

            source1.close()
            source2.close()
