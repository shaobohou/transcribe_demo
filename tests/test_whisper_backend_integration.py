from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from transcribe_demo import whisper_backend


def _load_fixture() -> tuple[np.ndarray, int]:
    fixture = Path(__file__).resolve().parent / "data" / "fox.wav"
    if not fixture.exists():
        raise FileNotFoundError("tests/data/fox.wav fixture not found")
    with wave.open(str(fixture), "rb") as wf:
        if wf.getnchannels() != 1:
            raise RuntimeError("fox.wav must be mono")
        rate = wf.getframerate()
        audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32)
    return audio / 32768.0, rate


def test_run_whisper_transcriber_processes_audio(monkeypatch):
    audio, sample_rate = _load_fixture()
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

    frame_size = 480  # 30ms for 16kHz

    class FakeInputStream:
        def __init__(self, callback, channels, samplerate, dtype):
            self.callback = callback
            assert channels == 1
            assert samplerate == sample_rate

        def __enter__(self):
            for start in range(0, len(audio), frame_size):
                frame = audio[start:start + frame_size]
                if not frame.size:
                    continue
                self.callback(frame.reshape(-1, 1), len(frame), None, None)
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(whisper_backend.sd, "InputStream", FakeInputStream)
    monkeypatch.setattr(whisper_backend.time, "sleep", lambda _=0.0: None)

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
