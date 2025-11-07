from __future__ import annotations

import threading
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


@pytest.mark.integration
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

    # Monkeypatch AudioCaptureManager to feed test data
    class FakeAudioCaptureManager:
        def __init__(self, sample_rate, channels, max_capture_duration=0.0, collect_full_audio=True):
            self.sample_rate = sample_rate
            self.channels = channels
            self.max_capture_duration = max_capture_duration
            self.collect_full_audio = collect_full_audio
            self.audio_queue = whisper_backend.queue.Queue()
            self.stop_event = threading.Event()
            self.capture_limit_reached = threading.Event()
            self._full_audio_chunks = []
            self._feeder_thread = None

        def _feed_audio(self):
            # Feed test audio into queue in a background thread
            for start in range(0, len(audio), frame_size):
                if self.stop_event.is_set():
                    break
                frame = audio[start : start + frame_size]
                if not frame.size:
                    continue
                self.audio_queue.put(frame.reshape(-1, 1))
                # Also collect for get_full_audio
                self._full_audio_chunks.append(frame.copy())
            # Signal end of stream
            self.audio_queue.put(None)

        def start(self):
            # Start feeding audio in background thread
            self._feeder_thread = threading.Thread(target=self._feed_audio, daemon=True)
            self._feeder_thread.start()

        def wait_until_stopped(self):
            # Wait until stop event is set
            self.stop_event.wait()

        def stop(self):
            self.stop_event.set()

        def close(self):
            if self._feeder_thread and self._feeder_thread.is_alive():
                self._feeder_thread.join(timeout=1.0)

        def get_full_audio(self):
            if not self._full_audio_chunks:
                return np.zeros(0, dtype=np.float32)
            return np.concatenate(self._full_audio_chunks)

        def get_capture_duration(self):
            return len(audio) / sample_rate

    monkeypatch.setattr(whisper_backend, "AudioCaptureManager", FakeAudioCaptureManager)

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
