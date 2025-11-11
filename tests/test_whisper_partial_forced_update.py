"""Tests for partial transcription forced update fix."""

from __future__ import annotations

import time

import numpy as np
import pytest

from test_helpers import create_fake_audio_capture_factory, generate_synthetic_audio
from transcribe_demo import whisper_backend


class DummyWhisperModel:
    """Dummy Whisper model that tracks transcription calls."""

    def __init__(self, fast: bool = False):
        self.calls: list[dict] = []
        self._counter = 0
        self._fast = fast

    def transcribe(self, audio_chunk: np.ndarray, **kwargs):
        call_info = {
            "call_number": self._counter,
            "audio_size": audio_chunk.size,
            "timestamp": time.perf_counter(),
        }
        self.calls.append(call_info)
        self._counter += 1

        prefix = "partial" if self._fast else "chunk"
        return {"text": f"{prefix}-{self._counter}"}


@pytest.mark.integration
def test_partial_transcription_continues_at_max_buffer_size(monkeypatch):
    """
    Test that partial transcriptions continue updating even when buffer reaches
    max_partial_buffer_seconds and becomes a sliding window.

    This is a regression test for the issue where partial transcriptions would
    stop updating once the buffer reached max size because the 10% size change
    threshold would never be met.
    """
    # Generate longer audio (15 seconds) so buffer will hit max_partial_buffer_seconds
    audio, sample_rate = generate_synthetic_audio(duration_seconds=15.0, sample_rate=16000)

    main_model = DummyWhisperModel(fast=False)
    partial_model = DummyWhisperModel(fast=True)

    # Track which model to return
    model_call_count = [0]

    def fake_load_whisper_model(**kwargs):
        model_call_count[0] += 1
        if model_call_count[0] == 1:
            # First call is for main model
            return main_model, "cpu", False
        else:
            # Second call is for partial model
            return partial_model, "cpu", False

    monkeypatch.setattr(whisper_backend, "load_whisper_model", fake_load_whisper_model)

    monkeypatch.setattr(
        "transcribe_demo.audio_capture.AudioCaptureManager",
        create_fake_audio_capture_factory(audio, sample_rate, frame_size=480),
    )

    partial_chunks: list[dict] = []

    def capture_chunk(index, text, start, end, inference_seconds, is_partial=False):
        if is_partial:
            partial_chunks.append({
                "index": index,
                "text": text,
                "timestamp": time.perf_counter(),
            })

    duration_seconds = len(audio) / sample_rate

    # Use large max_chunk_duration to prevent VAD from splitting
    # Use large min_silence_duration to prevent early chunking
    # This ensures buffer accumulates for partial transcription
    whisper_backend.run_whisper_transcriber(
        model_name="fixture",
        sample_rate=sample_rate,
        channels=1,
        temp_file=None,
        ca_cert=None,
        disable_ssl_verify=False,
        device_preference="cpu",
        require_gpu=False,
        chunk_consumer=capture_chunk,
        vad_aggressiveness=0,
        vad_min_silence_duration=20.0,  # Very high to prevent chunking
        vad_min_speech_duration=0.05,
        vad_speech_pad_duration=0.0,
        max_chunk_duration=60.0,  # Large enough to not trigger splits
        compare_transcripts=False,
        max_capture_duration=duration_seconds,
        language="en",
        enable_partial_transcription=True,
        partial_model="fixture-partial",
        partial_interval=0.5,  # Update every 0.5s
        max_partial_buffer_seconds=3.0,  # Buffer caps at 3 seconds
    )

    # Verify we got multiple partial transcriptions
    assert len(partial_chunks) >= 10, (
        f"Expected at least 10 partial transcriptions for 15s audio with 0.5s interval, "
        f"got {len(partial_chunks)}"
    )

    # Verify partial transcriptions continue throughout the session
    # Check that we have partials in the later part of the audio (after 8 seconds)
    later_partials = [p for p in partial_chunks if p["timestamp"] - partial_chunks[0]["timestamp"] > 8.0]
    assert len(later_partials) >= 5, (
        f"Expected at least 5 partial transcriptions after 8 seconds, "
        f"got {len(later_partials)}. This suggests partials stopped updating."
    )


@pytest.mark.integration
def test_partial_transcription_respects_interval(monkeypatch):
    """
    Test that forced updates respect the partial_interval parameter.

    Verifies that partials are generated at approximately the specified interval,
    not faster or slower.
    """
    # Generate 10 seconds of audio
    audio, sample_rate = generate_synthetic_audio(duration_seconds=10.0, sample_rate=16000)

    main_model = DummyWhisperModel(fast=False)
    partial_model = DummyWhisperModel(fast=True)

    model_call_count = [0]

    def fake_load_whisper_model(**kwargs):
        model_call_count[0] += 1
        if model_call_count[0] == 1:
            return main_model, "cpu", False
        else:
            return partial_model, "cpu", False

    monkeypatch.setattr(whisper_backend, "load_whisper_model", fake_load_whisper_model)

    monkeypatch.setattr(
        "transcribe_demo.audio_capture.AudioCaptureManager",
        create_fake_audio_capture_factory(audio, sample_rate, frame_size=480),
    )

    partial_timestamps: list[float] = []
    start_time = [0.0]

    def capture_chunk(index, text, start, end, inference_seconds, is_partial=False):
        if is_partial:
            if not start_time[0]:
                start_time[0] = time.perf_counter()
            partial_timestamps.append(time.perf_counter() - start_time[0])

    duration_seconds = len(audio) / sample_rate
    partial_interval = 1.0  # 1 second interval

    whisper_backend.run_whisper_transcriber(
        model_name="fixture",
        sample_rate=sample_rate,
        channels=1,
        temp_file=None,
        ca_cert=None,
        disable_ssl_verify=False,
        device_preference="cpu",
        require_gpu=False,
        chunk_consumer=capture_chunk,
        vad_aggressiveness=0,
        vad_min_silence_duration=20.0,  # Very high to prevent chunking
        vad_min_speech_duration=0.05,
        vad_speech_pad_duration=0.0,
        max_chunk_duration=60.0,
        compare_transcripts=False,
        max_capture_duration=duration_seconds,
        language="en",
        enable_partial_transcription=True,
        partial_model="fixture-partial",
        partial_interval=partial_interval,
        max_partial_buffer_seconds=3.0,
    )

    # Should have at least 5 partial transcriptions (10s audio / 1s interval, accounting for startup)
    assert len(partial_timestamps) >= 5, (
        f"Expected at least 5 partial transcriptions for 10s audio with 1s interval, "
        f"got {len(partial_timestamps)}"
    )

    # Check intervals between consecutive partials
    if len(partial_timestamps) >= 2:
        intervals = [
            partial_timestamps[i] - partial_timestamps[i - 1]
            for i in range(1, len(partial_timestamps))
        ]

        # Average interval should be close to partial_interval (within 0.5s tolerance)
        avg_interval = sum(intervals) / len(intervals)
        assert abs(avg_interval - partial_interval) < 0.5, (
            f"Average partial interval {avg_interval:.2f}s does not match "
            f"expected {partial_interval}s (tolerance ±0.5s)"
        )


@pytest.mark.integration
def test_partial_transcription_without_forced_update_would_fail(monkeypatch):
    """
    Test that demonstrates what would happen without forced updates.

    This test uses a very short partial_interval (0.1s) and verifies that we still
    get regular updates even when the buffer is at max size. Without the forced
    update fix, we would see very few partials after the buffer reaches max size.
    """
    # Generate 8 seconds of audio
    audio, sample_rate = generate_synthetic_audio(duration_seconds=8.0, sample_rate=16000)

    main_model = DummyWhisperModel(fast=False)
    partial_model = DummyWhisperModel(fast=True)

    model_call_count = [0]

    def fake_load_whisper_model(**kwargs):
        model_call_count[0] += 1
        if model_call_count[0] == 1:
            return main_model, "cpu", False
        else:
            return partial_model, "cpu", False

    monkeypatch.setattr(whisper_backend, "load_whisper_model", fake_load_whisper_model)

    monkeypatch.setattr(
        "transcribe_demo.audio_capture.AudioCaptureManager",
        create_fake_audio_capture_factory(audio, sample_rate, frame_size=480),
    )

    partial_chunks: list[dict] = []

    def capture_chunk(index, text, start, end, inference_seconds, is_partial=False):
        if is_partial:
            partial_chunks.append({"text": text, "timestamp": time.perf_counter()})

    duration_seconds = len(audio) / sample_rate

    whisper_backend.run_whisper_transcriber(
        model_name="fixture",
        sample_rate=sample_rate,
        channels=1,
        temp_file=None,
        ca_cert=None,
        disable_ssl_verify=False,
        device_preference="cpu",
        require_gpu=False,
        chunk_consumer=capture_chunk,
        vad_aggressiveness=0,
        vad_min_silence_duration=20.0,  # Very high to prevent chunking
        vad_min_speech_duration=0.05,
        vad_speech_pad_duration=0.0,
        max_chunk_duration=60.0,
        compare_transcripts=False,
        max_capture_duration=duration_seconds,
        language="en",
        enable_partial_transcription=True,
        partial_model="fixture-partial",
        partial_interval=0.1,  # Very frequent updates
        max_partial_buffer_seconds=2.0,  # Small buffer (2s)
    )

    # With forced updates, we should get many partials even after buffer reaches max
    # Expected: ~80 partials (8s / 0.1s interval)
    # Without forced updates, we'd get only ~20 partials (first 2s before buffer caps)
    assert len(partial_chunks) >= 40, (
        f"Expected at least 40 partial transcriptions with forced updates, "
        f"got {len(partial_chunks)}. The forced update fix may not be working."
    )

    # Verify continuous updates by checking we have partials throughout the session
    if partial_chunks:
        timestamps = [p["timestamp"] for p in partial_chunks]
        time_span = timestamps[-1] - timestamps[0]

        # Should span most of the audio duration
        assert time_span >= 6.0, (
            f"Partial transcriptions only span {time_span:.1f}s of the 8s audio. "
            f"They may have stopped updating after buffer reached max size."
        )
