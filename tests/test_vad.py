"""Tests for the WebRTC VAD functionality."""

import numpy as np
import pytest

from test_helpers import load_test_fixture
from transcribe_demo.whisper_backend import WebRTCVAD


class TestWebRTCVADInitialization:
    """Tests for WebRTCVAD initialization and validation."""

    def test_valid_sample_rates(self):
        """Test that valid sample rates are accepted."""
        for rate in [8000, 16000, 32000, 48000]:
            vad = WebRTCVAD(sample_rate=rate, frame_duration_ms=30, aggressiveness=2)
            assert vad.sample_rate == rate

    def test_invalid_sample_rate(self):
        """Test that invalid sample rates raise an error."""
        with pytest.raises(ValueError, match="Sample rate must be"):
            WebRTCVAD(sample_rate=44100, frame_duration_ms=30, aggressiveness=2)

    def test_valid_frame_durations(self):
        """Test that valid frame durations are accepted."""
        for duration in [10, 20, 30]:
            vad = WebRTCVAD(sample_rate=16000, frame_duration_ms=duration, aggressiveness=2)
            assert vad.frame_duration_ms == duration

    def test_invalid_frame_duration(self):
        """Test that invalid frame durations raise an error."""
        with pytest.raises(ValueError, match="Frame duration must be"):
            WebRTCVAD(sample_rate=16000, frame_duration_ms=25, aggressiveness=2)

    def test_valid_aggressiveness_levels(self):
        """Test that all aggressiveness levels (0-3) are accepted."""
        for level in [0, 1, 2, 3]:
            vad = WebRTCVAD(sample_rate=16000, frame_duration_ms=30, aggressiveness=level)
            assert vad.vad is not None

    def test_frame_size_calculation(self):
        """Test that frame size is calculated correctly."""
        vad = WebRTCVAD(sample_rate=16000, frame_duration_ms=30, aggressiveness=2)
        expected_frame_size = int(16000 * 30 / 1000)  # 480 samples
        assert vad.frame_size == expected_frame_size


class TestWebRTCVADSpeechDetection:
    """Tests for the is_speech function."""

    def test_silence_detection(self):
        """Test that silence (zeros) is detected as non-speech."""
        vad = WebRTCVAD(sample_rate=16000, frame_duration_ms=30, aggressiveness=3)
        frame_size = vad.frame_size

        # Create a silent frame (all zeros)
        silent_frame = np.zeros(frame_size, dtype=np.float32)

        # Silence should not be detected as speech
        assert not vad.is_speech(audio=silent_frame)

    def test_wrong_frame_size(self):
        """Test that wrong frame size returns False."""
        vad = WebRTCVAD(sample_rate=16000, frame_duration_ms=30, aggressiveness=2)

        # Create a frame with wrong size
        wrong_size_frame = np.zeros(100, dtype=np.float32)

        # Should return False for wrong frame size
        assert not vad.is_speech(audio=wrong_size_frame)

    def test_empty_frame(self):
        """Test that empty frame returns False."""
        vad = WebRTCVAD(sample_rate=16000, frame_duration_ms=30, aggressiveness=2)

        # Empty frame
        empty_frame = np.array([], dtype=np.float32)

        # Should return False
        assert not vad.is_speech(audio=empty_frame)


class TestWebRTCVADIntegration:
    """Integration tests for VAD in realistic scenarios."""

    def test_continuous_silence_frames(self):
        """Test multiple consecutive silent frames."""
        vad = WebRTCVAD(sample_rate=16000, frame_duration_ms=30, aggressiveness=2)
        frame_size = vad.frame_size

        # Create 10 silent frames - all should be detected as non-speech
        for _ in range(10):
            silent_frame = np.zeros(frame_size, dtype=np.float32)
            assert not vad.is_speech(audio=silent_frame)

    def test_clipped_audio(self):
        """Test that clipped audio at max amplitude is likely detected as speech."""
        vad = WebRTCVAD(sample_rate=16000, frame_duration_ms=30, aggressiveness=2)
        frame_size = vad.frame_size

        # Create audio at max amplitude (likely speech given the high energy)
        clipped_frame = np.ones(frame_size, dtype=np.float32)

        # Max amplitude should be detected as speech with moderate aggressiveness
        assert vad.is_speech(audio=clipped_frame)

    def test_different_sample_rates(self):
        """Test VAD with different sample rates."""
        for sample_rate in [8000, 16000, 32000, 48000]:
            vad = WebRTCVAD(sample_rate=sample_rate, frame_duration_ms=30, aggressiveness=2)
            frame_size = vad.frame_size

            # Create silent frame for this sample rate
            silent_frame = np.zeros(frame_size, dtype=np.float32)

            # Should detect as non-speech
            assert not vad.is_speech(audio=silent_frame)


class TestVADEdgeCases:
    """Tests for edge cases and error handling."""

    def test_nan_values(self):
        """Test handling of NaN values in audio doesn't crash."""
        vad = WebRTCVAD(sample_rate=16000, frame_duration_ms=30, aggressiveness=2)
        frame_size = vad.frame_size

        # Create frame with NaN values
        nan_frame = np.full(frame_size, np.nan, dtype=np.float32)

        # Should handle gracefully and return False (invalid audio)
        result = vad.is_speech(audio=nan_frame)
        assert result is False

    def test_inf_values(self):
        """Test handling of inf values in audio doesn't crash."""
        vad = WebRTCVAD(sample_rate=16000, frame_duration_ms=30, aggressiveness=2)
        frame_size = vad.frame_size

        # Create frame with inf values
        inf_frame = np.full(frame_size, np.inf, dtype=np.float32)

        # Should handle gracefully and return False (invalid audio)
        result = vad.is_speech(audio=inf_frame)
        assert result is False

    def test_very_large_values(self):
        """Test handling of very large values warns and clips properly."""
        vad = WebRTCVAD(sample_rate=16000, frame_duration_ms=30, aggressiveness=2)
        frame_size = vad.frame_size

        # Create frame with very large values
        large_frame = np.full(frame_size, 100.0, dtype=np.float32)

        # Should warn about clipping and still process
        with pytest.warns(UserWarning, match="Audio values exceeded"):
            result = vad.is_speech(audio=large_frame)
        # After clipping to max amplitude, should likely be detected as speech
        assert result is True

    def test_negative_values(self):
        """Test handling of negative audio values (valid for audio)."""
        vad = WebRTCVAD(sample_rate=16000, frame_duration_ms=30, aggressiveness=2)
        frame_size = vad.frame_size

        # Create frame with negative values (valid - audio has positive and negative)
        negative_frame = np.full(frame_size, -0.5, dtype=np.float32)

        # Negative values are valid audio, high amplitude should be detected as speech
        result = vad.is_speech(audio=negative_frame)
        assert result is True


@pytest.mark.integration
def test_webrtc_vad_detects_speech_in_sample_audio():
    """Validate VAD detects speech frames in the provided sample clip."""
    audio, sample_rate = load_test_fixture()

    if sample_rate != 16000:
        pytest.fail("fox.wav must be 16kHz for VAD test.")

    vad = WebRTCVAD(sample_rate=16000, frame_duration_ms=30, aggressiveness=2)
    frame_size = vad.frame_size
    if audio.size < frame_size:
        pytest.skip("Sample audio shorter than a single VAD frame.")

    trimmed = audio[: (audio.size // frame_size) * frame_size]
    frames = trimmed.reshape(-1, frame_size)

    max_frames = 500  # Limit processing time
    frames = frames[:max_frames]
    detections = sum(1 for frame in frames if vad.is_speech(audio=frame))

    assert detections > 0, "VAD failed to detect speech in sample audio."
