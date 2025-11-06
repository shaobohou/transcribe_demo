from __future__ import annotations

import json
import queue as queue_module
import wave
from pathlib import Path

import numpy as np
import pytest

from transcribe_demo import realtime_backend


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
    queue_holder: dict[str, queue_module.Queue[np.ndarray | None]] = {}
    fake_ws_holder: dict[str, object] = {}

    class TrackingQueue(queue_module.Queue):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            queue_holder["queue"] = self

    monkeypatch.setattr(realtime_backend.queue, "Queue", TrackingQueue)

    class FakeInputStream:
        def __init__(self, callback, channels, samplerate, dtype):
            assert channels == 1
            assert samplerate == sample_rate
            self.callback = callback

        def __enter__(self):
            for start in range(0, len(audio), frame_size):
                chunk = audio[start : start + frame_size]
                if not chunk.size:
                    continue
                indata = chunk.astype(np.float32).reshape(-1, 1)
                self.callback(indata, len(indata), None, None)
            # Signal end of stream to audio sender
            queue = queue_holder.get("queue")
            if queue is not None:
                queue.put(None)
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(realtime_backend.sd, "InputStream", FakeInputStream)

    async def fast_sleep(
        _duration: float,
    ) -> None:  # pragma: no cover - trivial awaitable
        return None

    monkeypatch.setattr(realtime_backend.asyncio, "sleep", fast_sleep)

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
