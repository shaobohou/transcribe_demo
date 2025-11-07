from __future__ import annotations

import json
import queue as queue_module
import threading
import wave
from pathlib import Path

import numpy as np
import pytest

from transcribe_demo import realtime_backend
from transcribe_demo import audio_capture


def _load_fixture() -> tuple[np.ndarray, int]:
    fixture = Path(__file__).resolve().parent / "data" / "fox.wav"
    if not fixture.exists():
        raise FileNotFoundError("tests/data/fox.wav fixture not found")
    with wave.open(str(fixture), "rb") as wf:
        if wf.getnchannels() != 1:
            raise RuntimeError("fox.wav must be mono")
        sample_rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    return audio, sample_rate


@pytest.mark.integration
def test_run_realtime_transcriber_processes_audio(monkeypatch):
    audio, sample_rate = _load_fixture()
    frame_size = 320  # 20ms at 16kHz

    chunk_texts: list[str] = []
    fake_ws_holder: dict[str, FakeWebSocket] = {}

    # Monkeypatch AudioCaptureManager to feed test data
    class FakeAudioCaptureManager:
        def __init__(self, sample_rate, channels, max_capture_duration=0.0, collect_full_audio=True):
            self.sample_rate = sample_rate
            self.channels = channels
            self.max_capture_duration = max_capture_duration
            self.collect_full_audio = collect_full_audio
            self.audio_queue = queue_module.Queue()
            self.stop_event = threading.Event()
            self.capture_limit_reached = threading.Event()
            self._full_audio_chunks = []
            self._feeder_thread = None
            self._total_samples_fed = 0

        def _feed_audio(self):
            # Feed test audio into queue in a background thread
            samples_fed = 0
            max_samples = int(self.sample_rate * self.max_capture_duration) if self.max_capture_duration > 0 else float('inf')

            for start in range(0, len(audio), frame_size):
                if self.stop_event.is_set():
                    break
                chunk = audio[start : start + frame_size]
                if not chunk.size:
                    continue
                indata = chunk.astype(np.float32).reshape(-1, 1)
                self.audio_queue.put(indata)
                # Also collect for get_full_audio
                mono = indata.mean(axis=1).astype(np.float32)
                self._full_audio_chunks.append(mono)
                samples_fed += len(mono)
                self._total_samples_fed = samples_fed

                # Check if capture duration limit reached
                if samples_fed >= max_samples:
                    self.capture_limit_reached.set()
                    self.audio_queue.put(None)
                    self.stop()
                    break
            else:
                # Signal end of stream if we finished naturally
                self.audio_queue.put(None)
                self.stop()

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
            return self._total_samples_fed / self.sample_rate

    monkeypatch.setattr(audio_capture, "AudioCaptureManager", FakeAudioCaptureManager)

    class FakeWebSocket:
        def __init__(self, events):
            self._events = list(events)
            self.sent_messages: list[dict[str, object]] = []
            self.closed = False

        async def send(self, message: str) -> None:
            self.sent_messages.append(json.loads(message))

        def __aiter__(self):
            return self

        async def __anext__(self):
            if not self._events:
                raise StopAsyncIteration
            return json.dumps(self._events.pop(0))

        async def close(self, code: int = 1000, reason: str = "") -> None:  # pragma: no cover - simple setter
            self.closed = True

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    def make_events():
        return [
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": "item-1",
                "transcript": "hello fox",
            },
            {"type": "session.input_audio_buffer.committed"},
        ]

    class FakeConnect:
        def __init__(self, ws):
            self._ws = ws

        async def __aenter__(self):
            return self._ws

        async def __aexit__(self, exc_type, exc, tb):
            return False

    def fake_connect(*args, **kwargs):
        ws = FakeWebSocket(make_events())
        fake_ws_holder["ws"] = ws
        return FakeConnect(ws)

    monkeypatch.setattr(realtime_backend.websockets, "connect", fake_connect)

    def collect_chunk(chunk_index, text, absolute_start, absolute_end, inference_seconds):
        if text:
            chunk_texts.append(text)

    monkeypatch.setattr(
        realtime_backend,
        "transcribe_full_audio_realtime",
        lambda *args, **kwargs: "full hello fox",
    )

    result = realtime_backend.run_realtime_transcriber(
        api_key="test-key",
        endpoint="wss://example.com",
        model="gpt-realtime-mini",
        sample_rate=sample_rate,
        channels=1,
        chunk_duration=0.2,
        instructions="transcribe precisely",
        insecure_downloads=False,
        chunk_consumer=collect_chunk,
        compare_transcripts=True,
        max_capture_duration=len(audio) / sample_rate,
        language="en",
    )

    assert result.chunks == ["hello fox"]
    assert result.full_audio.size > 0

    ws = fake_ws_holder["ws"]
    sent_types = {msg["type"] for msg in ws.sent_messages}
    assert "session.update" in sent_types
    assert "input_audio_buffer.append" in sent_types
    assert "input_audio_buffer.commit" in sent_types
