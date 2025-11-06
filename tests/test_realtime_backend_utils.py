import asyncio
import json

import numpy as np
import pytest

from transcribe_demo import realtime_backend


def test_float_to_pcm16_clips_and_scales():
    samples = np.array([-1.2, -0.5, 0.0, 0.5, 1.5], dtype=np.float32)
    pcm = realtime_backend.float_to_pcm16(samples)
    ints = np.frombuffer(pcm, dtype=np.int16)
    # Values should be clipped to int16 range and scaled appropriately.
    assert ints[0] == -32767
    assert ints[-1] == 32767
    assert ints[2] == 0


def test_resample_audio_adjusts_length():
    original = np.linspace(-1.0, 1.0, num=4, dtype=np.float32)
    resampled = realtime_backend.resample_audio(original, from_rate=4000, to_rate=8000)
    assert resampled.size > original.size
    assert np.isclose(resampled[0], original[0])
    assert np.isclose(resampled[-1], original[-1])


def test_send_json_uses_lock():
    sent = []

    class StubWS:
        async def send(self, message):
            sent.append(json.loads(message))

    payload = {"type": "ping"}

    async def run_test():
        lock = asyncio.Lock()
        await realtime_backend.send_json(StubWS(), payload, lock)

    asyncio.run(run_test())
    assert sent == [payload]


def test_run_async_falls_back_to_manual_loop(monkeypatch):
    original_new_loop = realtime_backend.asyncio.new_event_loop

    async def coro():
        await asyncio.sleep(0)
        return "ok"

    def failing_run(coro):
        coro.close()
        raise RuntimeError("loop running")

    def new_loop_wrapper():
        loop = original_new_loop()
        new_loop_wrapper.loop = loop
        return loop

    monkeypatch.setattr(realtime_backend.asyncio, "run", failing_run)
    monkeypatch.setattr(realtime_backend.asyncio, "new_event_loop", new_loop_wrapper)

    result = realtime_backend._run_async(lambda: coro())
    assert result == "ok"
    assert hasattr(new_loop_wrapper, "loop")


class FakeWebSocket:
    def __init__(self, events):
        self.events = asyncio.Queue()
        for event in events:
            self.events.put_nowait(json.dumps(event))
        self.sent_messages = []
        self.closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def send(self, message):
        self.sent_messages.append(json.loads(message))

    async def recv(self):
        if self.events.empty():
            await asyncio.sleep(0.01)
            raise asyncio.TimeoutError  # Should not be reached; loop closes earlier.
        return await self.events.get()

    async def close(self):
        self.closed = True


def test_transcribe_full_audio_realtime_collects_chunks(monkeypatch):
    audio = np.ones(8000, dtype=np.float32) * 0.1

    events = [
        {
            "type": "conversation.item.input_audio_transcription.delta",
            "item_id": "item-1",
            "delta": "partial ",
        },
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": "item-1",
            "transcript": "partial final",
        },
        {"type": "session.input_audio_buffer.committed"},
        {"type": "session.closed"},
    ]

    fake_ws = FakeWebSocket(events)

    class FakeConnect:
        def __init__(self, ws):
            self.ws = ws

        def __await__(self):
            async def inner():
                return self.ws
            return inner().__await__()

        async def __aenter__(self):
            return await asyncio.sleep(0, self.ws)

        async def __aexit__(self, exc_type, exc, tb):
            return False

    async def fake_send_json(ws, payload, lock):
        async with lock:
            await ws.send(json.dumps(payload))

    monkeypatch.setattr(realtime_backend, "send_json", fake_send_json)
    monkeypatch.setattr(realtime_backend.websockets, "connect", lambda *args, **kwargs: FakeConnect(fake_ws))

    transcript = realtime_backend.transcribe_full_audio_realtime(
        audio=audio,
        sample_rate=8000,
        chunk_duration=0.5,
        api_key="key",
        endpoint="wss://example.com",
        model="model",
        instructions="instructions",
        insecure_downloads=False,
        language="en",
    )

    assert transcript == "partial final"
    # Ensure audio payloads were sent (session update + audio + commit)
    message_types = [msg["type"] for msg in fake_ws.sent_messages]
    assert "session.update" in message_types
    assert "input_audio_buffer.append" in message_types
    assert "input_audio_buffer.commit" in message_types
