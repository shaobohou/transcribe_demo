from __future__ import annotations

import numpy as np
import pytest

from conftest import create_fake_audio_capture_factory, load_test_fixture
from transcribe_demo import whisper_backend


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

    # Use helper to create fake audio capture manager
    monkeypatch.setattr(
        "transcribe_demo.audio_capture.AudioCaptureManager",
        create_fake_audio_capture_factory(audio, sample_rate, frame_size=480),
    )

    def capture_chunk(index, text, start, end, inference_seconds):
        chunks.append(
            {
                "index": index,
                "text": text,
                "start": start,
                "end": end,
                "inference": inference_seconds,
            }
        )

    duration_seconds = len(audio) / sample_rate
    result = whisper_backend.run_whisper_transcriber(
        model_name="fixture",
        sample_rate=sample_rate,
        channels=1,
        temp_file=None,
        ca_cert=None,
        insecure_downloads=False,
        device_preference="cpu",
        require_gpu=False,
        chunk_consumer=capture_chunk,
        vad_aggressiveness=0,
        vad_min_silence_duration=0.2,
        vad_min_speech_duration=0.05,
        vad_speech_pad_duration=0.0,
        max_chunk_duration=5.0,
        compare_transcripts=False,
        max_capture_duration=duration_seconds,
        language="en",
    )

    assert result.full_audio_transcription is None
    assert dummy_model.calls, "Model was never invoked"
    assert chunks, "No chunks were produced"
    texts = [entry["text"] for entry in chunks]
    assert "chunk-1" in texts
