from __future__ import annotations

import json

import numpy as np
import pytest

from test_helpers import create_fake_audio_capture_factory, load_test_fixture
from transcribe_demo import realtime_backend


@pytest.mark.integration
def test_run_realtime_transcriber_processes_audio(monkeypatch):
    audio, sample_rate = load_test_fixture()

    chunk_texts: list[str] = []
    fake_ws_holder: dict[str, "FakeWebSocket"] = {}

    # Use helper to create fake audio capture manager
    monkeypatch.setattr(
        "transcribe_demo.audio_capture.AudioCaptureManager",
        create_fake_audio_capture_factory(audio, sample_rate, frame_size=320),  # 20ms for realtime
    )

    # Custom FakeWebSocket for this test with specific events
    class FakeWebSocket:
        def __init__(self, events):
            self._events = list(events)
            self.sent_messages: list[dict[str, object]] = []
            self.closed = False

        async def send(self, message: str) -> None:
            self.sent_messages.append(json.loads(message))

        async def recv(self):
            """Receive method compatible with websockets.recv() API."""
            import asyncio

            if not self._events:
                # Block forever to trigger timeout
                await asyncio.sleep(1000)
            return json.dumps(self._events.pop(0))

        def __aiter__(self):
            return self

        async def __anext__(self):
            if not self._events:
                raise StopAsyncIteration
            return json.dumps(self._events.pop(0))

        async def close(self, code: int = 1000, reason: str = "") -> None:
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

    def collect_chunk(chunk):
        if chunk.text:
            chunk_texts.append(chunk.text)

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
        disable_ssl_verify=False,
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


@pytest.mark.integration
def test_realtime_backend_full_audio_matches_input(monkeypatch):
    """Test that full audio returned from realtime backend matches the original input audio."""
    audio, sample_rate = load_test_fixture()

    # Use helper to create fake audio capture manager
    monkeypatch.setattr(
        "transcribe_demo.audio_capture.AudioCaptureManager",
        create_fake_audio_capture_factory(audio, sample_rate, frame_size=320),
    )

    # Custom FakeWebSocket for this test
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

        async def close(self, code: int = 1000, reason: str = "") -> None:
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
                "transcript": "test audio",
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
        return FakeConnect(ws)

    monkeypatch.setattr(realtime_backend.websockets, "connect", fake_connect)

    monkeypatch.setattr(
        realtime_backend,
        "transcribe_full_audio_realtime",
        lambda *args, **kwargs: "test audio full",
    )

    result = realtime_backend.run_realtime_transcriber(
        api_key="test-key",
        endpoint="wss://example.com",
        model="gpt-realtime-mini",
        sample_rate=sample_rate,
        channels=1,
        chunk_duration=0.2,
        instructions="transcribe precisely",
        disable_ssl_verify=False,
        chunk_consumer=lambda *, chunk_index, text, absolute_start, absolute_end, inference_seconds: None,
        compare_transcripts=True,  # Must be True to enable full audio collection
        max_capture_duration=len(audio) / sample_rate,
        language="en",
    )

    # Verify that the full audio in the result matches the original input
    assert result.full_audio.size > 0, "No audio was captured"
    # Audio should match the original input (possibly with minor floating point differences)
    assert np.allclose(result.full_audio, audio, rtol=1e-5, atol=1e-6), "Captured audio does not match original input"
