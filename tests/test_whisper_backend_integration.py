from __future__ import annotations

import queue
from typing import Any

import numpy as np
import pytest

from test_helpers import create_fake_audio_capture_factory, load_test_fixture
from transcribe_demo import whisper_backend
from transcribe_demo.backend_protocol import TranscriptionChunk


@pytest.mark.integration
def test_run_whisper_transcriber_processes_audio(monkeypatch):
    audio, sample_rate = load_test_fixture()
    chunks: list[dict[str, float | str | None]] = []

    class DummyModel:
        def __init__(self):
            self.calls: list[np.ndarray] = []
            self._counter = 0

        def transcribe(self, audio_chunk: np.ndarray, **kwargs):
            self.calls.append(np.array(audio_chunk))
            self._counter += 1
            return {"text": f"chunk-{self._counter}"}

    dummy_model = DummyModel()

    def fake_load_whisper_model(**kwargs):
        return dummy_model, "cpu", False

    monkeypatch.setattr(whisper_backend, "load_whisper_model", fake_load_whisper_model)

    # Create fake audio source
    from test_helpers import FakeAudioCaptureManager
    duration_seconds = len(audio) / sample_rate
    audio_source = FakeAudioCaptureManager(
        audio=audio,
        sample_rate=sample_rate,
        channels=1,
        max_capture_duration=duration_seconds,
        collect_full_audio=False,
        frame_size=480,
    )

    chunk_queue: queue.Queue[TranscriptionChunk | None] = queue.Queue()

    result = whisper_backend.run_whisper_transcriber(
        model_name="fixture",
        sample_rate=sample_rate,
        channels=1,
        temp_file=None,
        ca_cert=None,
        disable_ssl_verify=False,
        device_preference="cpu",
        require_gpu=False,
        audio_source=audio_source,
        chunk_queue=chunk_queue,
        vad_aggressiveness=0,
        vad_min_silence_duration=0.2,
        vad_min_speech_duration=0.05,
        vad_speech_pad_duration=0.0,
        max_chunk_duration=5.0,
        compare_transcripts=False,
        language="en",
    )

    # Collect chunks from queue
    while True:
        chunk = chunk_queue.get()
        if chunk is None:
            break
        if not chunk.is_partial:
            chunks.append(
                {
                    "index": chunk.index,
                    "text": chunk.text,
                    "start": chunk.start_time,
                    "end": chunk.end_time,
                    "inference": chunk.inference_seconds,
                }
            )

    assert result.full_audio_transcription is None
    assert dummy_model.calls, "Model was never invoked"
    assert chunks, "No chunks were produced"
    texts = [entry["text"] for entry in chunks]
    assert "chunk-1" in texts


@pytest.mark.integration
def test_whisper_backend_full_audio_matches_input(monkeypatch):
    """Test that full audio returned from audio capture matches the original input audio."""
    from typing import TYPE_CHECKING

    audio, sample_rate = load_test_fixture()
    audio_capture_holder: dict[str, Any] = {}

    if TYPE_CHECKING:
        from test_helpers import FakeAudioCaptureManager

    class DummyModel:
        def transcribe(self, audio_chunk: np.ndarray, **kwargs):
            return {"text": "test"}

    def fake_load_whisper_model(**kwargs):
        return DummyModel(), "cpu", False

    monkeypatch.setattr(whisper_backend, "load_whisper_model", fake_load_whisper_model)

    # Create fake audio source directly
    from test_helpers import FakeAudioCaptureManager
    duration_seconds = len(audio) / sample_rate
    audio_source = FakeAudioCaptureManager(
        audio=audio,
        sample_rate=sample_rate,
        channels=1,
        max_capture_duration=duration_seconds,
        collect_full_audio=True,  # Must be True for full audio comparison
        frame_size=480,
    )
    audio_capture_holder["manager"] = audio_source

    chunk_queue: queue.Queue[TranscriptionChunk | None] = queue.Queue()

    whisper_backend.run_whisper_transcriber(
        model_name="fixture",
        sample_rate=sample_rate,
        channels=1,
        temp_file=None,
        ca_cert=None,
        disable_ssl_verify=False,
        device_preference="cpu",
        require_gpu=False,
        audio_source=audio_source,
        chunk_queue=chunk_queue,
        vad_aggressiveness=0,
        vad_min_silence_duration=0.2,
        vad_min_speech_duration=0.05,
        vad_speech_pad_duration=0.0,
        max_chunk_duration=5.0,
        compare_transcripts=True,  # Must be True to enable full audio collection
        language="en",
    )

    # Drain the queue
    while True:
        chunk = chunk_queue.get()
        if chunk is None:
            break

    # Verify that the full audio captured matches the original input
    if TYPE_CHECKING:
        manager: FakeAudioCaptureManager = audio_capture_holder["manager"]  # type: ignore[assignment]
    else:
        manager = audio_capture_holder["manager"]
    captured_audio = manager.get_full_audio()
    assert captured_audio.size > 0, "No audio was captured"
    # Audio should match the original input (possibly with minor floating point differences)
    assert np.allclose(captured_audio, audio, rtol=1e-5, atol=1e-6), "Captured audio does not match original input"
