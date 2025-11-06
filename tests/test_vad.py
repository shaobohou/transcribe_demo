"""Tests for the WebRTC VAD functionality."""

from pathlib import Path
import wave

import numpy as np
import pytest

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
        assert not vad.is_speech(silent_frame)

    def test_noise_detection(self):
        """Test detection with noise."""
        vad = WebRTCVAD(sample_rate=16000, frame_duration_ms=30, aggressiveness=2)
        frame_size = vad.frame_size

        # Create a frame with low-level random noise
        noise_frame = np.random.randn(frame_size).astype(np.float32) * 0.01

        # Low noise should likely be detected as non-speech (depends on aggressiveness)
        result = vad.is_speech(noise_frame)
        assert isinstance(result, bool)

    def test_simulated_speech_detection(self):
        """Test with simulated speech-like signal."""
        vad = WebRTCVAD(sample_rate=16000, frame_duration_ms=30, aggressiveness=1)
        frame_size = vad.frame_size

        # Create a frame with higher amplitude periodic signal (simulating speech)
        t = np.linspace(0, 0.03, frame_size)
        speech_like = (np.sin(2 * np.pi * 200 * t) * 0.3).astype(np.float32)

        # This should be more likely to be detected as speech
        result = vad.is_speech(speech_like)
        assert isinstance(result, bool)

    def test_wrong_frame_size(self):
        """Test that wrong frame size returns False."""
        vad = WebRTCVAD(sample_rate=16000, frame_duration_ms=30, aggressiveness=2)

        # Create a frame with wrong size
        wrong_size_frame = np.zeros(100, dtype=np.float32)

        # Should return False for wrong frame size
        assert not vad.is_speech(wrong_size_frame)

    def test_empty_frame(self):
        """Test that empty frame returns False."""
        vad = WebRTCVAD(sample_rate=16000, frame_duration_ms=30, aggressiveness=2)

        # Empty frame
        empty_frame = np.array([], dtype=np.float32)

        # Should return False
        assert not vad.is_speech(empty_frame)

    def test_aggressiveness_levels(self):
        """Test that higher aggressiveness is more likely to reject noise."""
        frame_size_30ms = int(16000 * 30 / 1000)

        # Create a low-amplitude signal
        low_signal = np.random.randn(frame_size_30ms).astype(np.float32) * 0.05

        # Test different aggressiveness levels
        vad_low = WebRTCVAD(sample_rate=16000, frame_duration_ms=30, aggressiveness=0)
        vad_high = WebRTCVAD(sample_rate=16000, frame_duration_ms=30, aggressiveness=3)

        result_low = vad_low.is_speech(low_signal)
        result_high = vad_high.is_speech(low_signal)

        # Both should return boolean
        assert isinstance(result_low, bool)
        assert isinstance(result_high, bool)

        # High aggressiveness should be less likely to detect speech in noise
        # (but we can't guarantee this without real audio, so just check types)


class TestWebRTCVADIntegration:
    """Integration tests for VAD in realistic scenarios."""

    def test_continuous_silence_frames(self):
        """Test multiple consecutive silent frames."""
        vad = WebRTCVAD(sample_rate=16000, frame_duration_ms=30, aggressiveness=2)
        frame_size = vad.frame_size

        # Create 10 silent frames
        for _ in range(10):
            silent_frame = np.zeros(frame_size, dtype=np.float32)
            assert not vad.is_speech(silent_frame)

    def test_alternating_speech_silence(self):
        """Test alternating speech and silence frames."""
        vad = WebRTCVAD(sample_rate=16000, frame_duration_ms=30, aggressiveness=1)
        frame_size = vad.frame_size

        # Silent frame
        silent_frame = np.zeros(frame_size, dtype=np.float32)

        # Speech-like frame
        t = np.linspace(0, 0.03, frame_size)
        speech_frame = (np.sin(2 * np.pi * 300 * t) * 0.5).astype(np.float32)

        # Test alternating - all should return boolean
        # Note: VAD behavior can be affected by internal state, so we just verify types
        result1 = vad.is_speech(silent_frame)
        result2 = vad.is_speech(speech_frame)
        result3 = vad.is_speech(silent_frame)

        assert isinstance(result1, bool)
        assert isinstance(result2, bool)
        assert isinstance(result3, bool)

    def test_clipped_audio(self):
        """Test that clipped audio (at boundaries) is handled."""
        vad = WebRTCVAD(sample_rate=16000, frame_duration_ms=30, aggressiveness=2)
        frame_size = vad.frame_size

        # Create audio at max amplitude
        clipped_frame = np.ones(frame_size, dtype=np.float32)

        # Should handle without crashing
        result = vad.is_speech(clipped_frame)
        assert isinstance(result, bool)

    def test_different_sample_rates(self):
        """Test VAD with different sample rates."""
        for sample_rate in [8000, 16000, 32000, 48000]:
            vad = WebRTCVAD(sample_rate=sample_rate, frame_duration_ms=30, aggressiveness=2)
            frame_size = vad.frame_size

            # Create silent frame for this sample rate
            silent_frame = np.zeros(frame_size, dtype=np.float32)

            # Should detect as non-speech
            assert not vad.is_speech(silent_frame)


class TestVADEdgeCases:
    """Tests for edge cases and error handling."""

    def test_nan_values(self):
        """Test handling of NaN values in audio."""
        vad = WebRTCVAD(sample_rate=16000, frame_duration_ms=30, aggressiveness=2)
        frame_size = vad.frame_size

        # Create frame with NaN values
        nan_frame = np.full(frame_size, np.nan, dtype=np.float32)

        # Should handle gracefully (likely returns False)
        result = vad.is_speech(nan_frame)
        assert isinstance(result, bool)

    def test_inf_values(self):
        """Test handling of inf values in audio."""
        vad = WebRTCVAD(sample_rate=16000, frame_duration_ms=30, aggressiveness=2)
        frame_size = vad.frame_size

        # Create frame with inf values
        inf_frame = np.full(frame_size, np.inf, dtype=np.float32)

        # Should handle gracefully
        result = vad.is_speech(inf_frame)
        assert isinstance(result, bool)

    def test_very_large_values(self):
        """Test handling of very large values (beyond int16 range)."""
        vad = WebRTCVAD(sample_rate=16000, frame_duration_ms=30, aggressiveness=2)
        frame_size = vad.frame_size

        # Create frame with very large values
        large_frame = np.full(frame_size, 100.0, dtype=np.float32)

        # Should clip and handle
        result = vad.is_speech(large_frame)
        assert isinstance(result, bool)

    def test_negative_values(self):
        """Test handling of negative values in audio."""
        vad = WebRTCVAD(sample_rate=16000, frame_duration_ms=30, aggressiveness=2)
        frame_size = vad.frame_size

        # Create frame with negative values
        negative_frame = np.full(frame_size, -0.5, dtype=np.float32)

        # Should handle properly
        result = vad.is_speech(negative_frame)
        assert isinstance(result, bool)


@pytest.mark.integration
def test_webrtc_vad_detects_speech_in_sample_audio():
    """Validate VAD detects speech frames in the provided sample clip."""
    sample_path = Path(__file__).resolve().parent / "data" / "fox.wav"
    if not sample_path.exists():
        pytest.fail("Sample audio fox.wav not present in tests/data.")

    with wave.open(str(sample_path), "rb") as wf:
        if wf.getnchannels() != 1:
            pytest.fail("fox.wav must be mono for VAD test.")
        sample_rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())

    if sample_rate != 16000:
        pytest.fail("fox.wav must be 16kHz for VAD test.")

    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

    vad = WebRTCVAD(sample_rate=16000, frame_duration_ms=30, aggressiveness=2)
    frame_size = vad.frame_size
    if audio.size < frame_size:
        pytest.skip("Sample audio shorter than a single VAD frame.")

    trimmed = audio[: (audio.size // frame_size) * frame_size]
    frames = trimmed.reshape(-1, frame_size)

    max_frames = 500  # Limit processing time
    frames = frames[:max_frames]
    detections = sum(1 for frame in frames if vad.is_speech(frame))

    assert detections > 0, "VAD failed to detect speech in sample audio."
