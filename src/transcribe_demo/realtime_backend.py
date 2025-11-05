from __future__ import annotations

import asyncio
import base64
import json
import queue
import ssl
import sys
import threading
import time
from typing import Dict, Optional

import numpy as np
import sounddevice as sd
import websockets


def float_to_pcm16(audio: np.ndarray) -> bytes:
    mono = audio.astype(np.float32, copy=False)
    np.clip(mono, -1.0, 1.0, out=mono)
    ints = (mono * 32767.0).astype(np.int16, copy=False)
    return ints.tobytes()


def resample_audio(audio: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    if from_rate == to_rate or audio.size == 0:
        return audio
    duration = audio.size / float(from_rate)
    target_length = int(round(duration * to_rate))
    if target_length <= 1:
        return np.zeros(0, dtype=np.float32)
    source_positions = np.linspace(0, audio.size - 1, audio.size, dtype=np.float32)
    target_positions = np.linspace(0, audio.size - 1, target_length, dtype=np.float32)
    return np.interp(target_positions, source_positions, audio).astype(np.float32, copy=False)


async def send_json(
    ws: websockets.WebSocketClientProtocol,
    payload: Dict[str, object],
    lock: asyncio.Lock,
) -> None:
    message = json.dumps(payload)
    async with lock:
        await ws.send(message)


def run_realtime_transcriber(
    api_key: str,
    endpoint: str,
    model: str,
    sample_rate: int,
    channels: int,
    chunk_duration: float,
    instructions: str,
    insecure_downloads: bool = False,
    chunk_consumer: Optional[object] = None,
) -> None:
    audio_queue: queue.Queue[np.ndarray] = queue.Queue()
    stop_event = threading.Event()
    session_sample_rate = 24000
    chunk_counter = 0
    session_start_time = time.perf_counter()
    cumulative_time = 0.0

    def callback(indata: np.ndarray, frames: int, time, status) -> None:
        if status:
            print(f"InputStream status: {status}", file=sys.stderr)
        if stop_event.is_set():
            return
        audio_queue.put(indata.copy())

    async def wait_for_stop() -> None:
        try:
            await asyncio.to_thread(input, "Press Enter to stop...\n")
        except (EOFError, OSError):
            # If stdin is not available (e.g., running in background), just wait
            while not stop_event.is_set():
                await asyncio.sleep(0.1)
        stop_event.set()

    async def audio_sender(
        ws: websockets.WebSocketClientProtocol,
        lock: asyncio.Lock,
    ) -> None:
        buffer = np.zeros(0, dtype=np.float32)
        chunk_size = max(int(sample_rate * chunk_duration), 1)
        while not stop_event.is_set():
            if buffer.size < chunk_size:
                try:
                    chunk = await asyncio.to_thread(audio_queue.get, True, 0.1)
                except queue.Empty:
                    if stop_event.is_set():
                        break
                    continue
                if chunk.ndim > 1:
                    chunk = chunk.mean(axis=1)
                buffer = np.concatenate((buffer, chunk.astype(np.float32, copy=False)))
                continue

            window = buffer[:chunk_size]
            resampled = resample_audio(window, sample_rate, session_sample_rate)

            # Only send if we have enough audio (at least 100ms as required by API)
            if len(resampled) == 0:
                buffer = buffer[chunk_size:]
                continue

            pcm_payload = base64.b64encode(float_to_pcm16(resampled)).decode("ascii")
            await send_json(
                ws,
                {
                    "type": "input_audio_buffer.append",
                    "audio": pcm_payload,
                },
                lock,
            )
            # Don't manually commit when using server_vad - the server handles it automatically
            buffer = buffer[chunk_size:]

    async def receiver(
        ws: websockets.WebSocketClientProtocol,
    ) -> None:
        nonlocal chunk_counter, cumulative_time
        partials: Dict[str, str] = {}
        try:
            async for message in ws:
                payload = json.loads(message)
                event_type = payload.get("type")
                if event_type == "conversation.item.input_audio_transcription.delta":
                    item_id = payload.get("item_id")
                    delta = payload.get("delta") or ""
                    if not delta:
                        continue
                    if item_id:
                        partials[item_id] = partials.get(item_id, "") + delta
                elif event_type == "conversation.item.input_audio_transcription.completed":
                    item_id = payload.get("item_id")
                    transcript = payload.get("transcript") or ""
                    had_partials = bool(item_id and partials.get(item_id))
                    final_text = transcript if transcript else ""
                    if not final_text and had_partials and item_id:
                        final_text = partials.get(item_id, "")
                    final_text = final_text.strip()

                    # Track absolute timestamp from session start
                    chunk_start = cumulative_time
                    chunk_end = time.perf_counter() - session_start_time
                    cumulative_time = chunk_end

                    if chunk_consumer:
                        # For realtime mode: show absolute timestamps from session start
                        # These represent when chunks completed, not actual audio duration
                        chunk_consumer(
                            chunk_index=chunk_counter,
                            text=final_text,
                            absolute_start=chunk_start,
                            absolute_end=chunk_end,
                            inference_seconds=None,  # Signals realtime mode (show timestamp)
                        )
                    else:
                        # Fallback to old behavior if no consumer provided
                        label = f"[chunk {chunk_counter:03d} | {chunk_end:.2f}s]"
                        if final_text:
                            print(f"{label} {final_text}", flush=True)
                        else:
                            print(label, flush=True)

                    chunk_counter += 1
                    if item_id:
                        partials.pop(item_id, None)
                elif event_type == "error":
                    message_text = payload.get("error") or payload.get("message") or payload
                    print(f"\nRealtime error: {message_text}", file=sys.stderr)
                elif event_type == "error.session":
                    message_text = payload.get("message") or payload
                    print(f"\nRealtime session error: {message_text}", file=sys.stderr)
                if stop_event.is_set():
                    break
        except websockets.ConnectionClosed:
            stop_event.set()

    async def runtime() -> None:
        uri = f"{endpoint}?model={model}"
        headers = [
            ("Authorization", f"Bearer {api_key}"),
            ("OpenAI-Beta", "realtime=v1"),
        ]
        lock = asyncio.Lock()

        ssl_context: Optional[ssl.SSLContext] = None
        if insecure_downloads:
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE

        async with websockets.connect(uri, additional_headers=headers, max_size=None, ssl=ssl_context) as ws:
            await send_json(
                ws,
                {
                    "type": "session.update",
                    "session": {
                        "modalities": ["text"],
                        "instructions": instructions,
                        "input_audio_format": "pcm16",
                        "input_audio_transcription": {
                            "model": "whisper-1",
                        },
                        "turn_detection": {
                            "type": "server_vad",
                            "threshold": 0.3,
                            "prefix_padding_ms": 200,
                            "silence_duration_ms": 300,
                        },
                        "temperature": 0.6,
                        "max_response_output_tokens": 4096,
                    },
                },
                lock,
            )
            tasks = [
                asyncio.create_task(audio_sender(ws, lock)),
                asyncio.create_task(receiver(ws)),
                asyncio.create_task(wait_for_stop()),
            ]
            done, pending = await asyncio.wait(
                tasks,
                return_when=asyncio.FIRST_COMPLETED,
            )
            stop_event.set()
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            for task in done:
                if task.exception():
                    raise task.exception()

    with sd.InputStream(
        callback=callback,
        channels=channels,
        samplerate=sample_rate,
        dtype="float32",
    ):
        try:
            asyncio.run(runtime())
        finally:
            stop_event.set()
