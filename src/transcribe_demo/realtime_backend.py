from __future__ import annotations

import asyncio
import base64
import json
import queue
import ssl
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Coroutine, Dict, Optional, List

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


@dataclass
class RealtimeTranscriptionResult:
    """Aggregate data returned after a realtime transcription session."""
    full_audio: np.ndarray
    sample_rate: int
    chunks: List[str] | None = None


def _run_async(coro: Coroutine[Any, Any, str]) -> str:
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()


def transcribe_full_audio_realtime(
    audio: np.ndarray,
    sample_rate: int,
    chunk_duration: float,
    api_key: str,
    endpoint: str,
    model: str,
    instructions: str,
    insecure_downloads: bool = False,
) -> str:
    """
    Transcribe the entire audio buffer using the realtime backend.
    Returns full transcript text produced by the realtime model.
    """

    if audio.size == 0:
        return ""

    audio = np.asarray(audio, dtype=np.float32)
    session_sample_rate = 24000
    chunk_size = max(int(sample_rate * chunk_duration), sample_rate // 2 or 1)

    async def _transcribe() -> str:
        uri = f"{endpoint}?model={model}"
        headers = [
            ("Authorization", f"Bearer {api_key}"),
            ("OpenAI-Beta", "realtime=v1"),
        ]
        ssl_context: Optional[ssl.SSLContext] = None
        if insecure_downloads:
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE

        lock = asyncio.Lock()
        partials: Dict[str, str] = {}
        completed: list[str] = []
        processed_items: set[str] = set()

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

            cursor = 0
            total_samples = audio.size
            while cursor < total_samples:
                window = audio[cursor: cursor + chunk_size]
                cursor += chunk_size
                resampled = resample_audio(window, sample_rate, session_sample_rate)
                if resampled.size == 0:
                    continue
                payload = base64.b64encode(float_to_pcm16(resampled)).decode("ascii")
                await send_json(
                    ws,
                    {
                        "type": "input_audio_buffer.append",
                        "audio": payload,
                    },
                    lock,
                )
            await send_json(ws, {"type": "input_audio_buffer.commit"}, lock)

            while True:
                try:
                    message = await asyncio.wait_for(ws.recv(), timeout=2.0)
                except asyncio.TimeoutError:
                    if completed and not partials:
                        break
                    continue

                payload = json.loads(message)
                event_type = payload.get("type")

                if event_type == "conversation.item.input_audio_transcription.delta":
                    item_id = payload.get("item_id")
                    delta = payload.get("delta") or ""
                    if item_id and delta:
                        partials[item_id] = partials.get(item_id, "") + delta
                elif event_type == "conversation.item.input_audio_transcription.completed":
                    item_id = payload.get("item_id")
                    if item_id and item_id in processed_items:
                        continue
                    transcript = payload.get("transcript") or ""
                    final_text = transcript.strip()
                    if not final_text and item_id:
                        final_text = (partials.get(item_id) or "").strip()
                    if final_text and (not completed or final_text != completed[-1]):
                        completed.append(final_text)
                    if item_id:
                        processed_items.add(item_id)
                        partials.pop(item_id, None)
                elif event_type == "session.input_audio_buffer.committed":
                    if completed and not partials:
                        break
                elif event_type == "session.closed":
                    break
                elif event_type and event_type.startswith("error"):
                    raise RuntimeError(f"Realtime error: {payload}")

            await ws.close()
        return " ".join(completed).strip()

    return _run_async(_transcribe())


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
    compare_transcripts: bool = True,
    max_capture_duration: float = 120.0,
) -> RealtimeTranscriptionResult:
    audio_queue: queue.Queue[np.ndarray] = queue.Queue()
    stop_event = threading.Event()
    capture_limit_reached = threading.Event()
    session_sample_rate = 24000
    chunk_counter = 0
    session_start_time = time.perf_counter()
    cumulative_time = 0.0
    full_audio_chunks: list[np.ndarray] = []
    full_audio_lock = threading.Lock()
    chunk_texts: list[str] = []
    # Always track capture duration, but only collect full audio for comparison if enabled
    max_capture_samples = int(sample_rate * max_capture_duration) if max_capture_duration > 0 else 0
    total_samples_captured = 0

    def callback(indata: np.ndarray, frames: int, time, status) -> None:
        nonlocal total_samples_captured
        if status:
            print(f"InputStream status: {status}", file=sys.stderr)
        if stop_event.is_set() or capture_limit_reached.is_set():
            return
        chunk = indata.copy()
        audio_queue.put(chunk)

        # Convert to mono for tracking and optional storage
        mono = chunk.astype(np.float32, copy=False)
        if mono.ndim > 1:
            mono = mono.mean(axis=1)
        else:
            mono = mono.reshape(-1)

        with full_audio_lock:
            # Always track total samples captured (for duration limit)
            total_samples_captured += len(mono)

            # Check if capture duration limit reached
            if max_capture_samples > 0 and total_samples_captured >= max_capture_samples:
                if not capture_limit_reached.is_set():
                    capture_limit_reached.set()
                    duration_captured = total_samples_captured / sample_rate
                    print(
                        f"\nCapture duration limit reached ({duration_captured:.1f}s). "
                        f"Finishing current transcriptions and stopping gracefully...",
                        file=sys.stderr,
                    )

            # Preserve full-session audio for post-run transcription comparison (if enabled)
            if compare_transcripts:
                full_audio_chunks.append(mono.copy())
                # Trim to max_capture_duration by removing oldest chunks
                if max_capture_samples > 0:
                    total_samples = sum(len(c) for c in full_audio_chunks)
                    while total_samples > max_capture_samples and len(full_audio_chunks) > 1:
                        removed = full_audio_chunks.pop(0)
                        total_samples -= len(removed)

    async def wait_for_stop() -> None:
        """Wait for either user input or capture limit, whichever comes first."""

        async def wait_for_input() -> None:
            """Wait for user to press Enter."""
            try:
                await asyncio.to_thread(input, "Press Enter to stop...\n")
            except (EOFError, OSError):
                # stdin not available (e.g., background execution)
                # Wait indefinitely - capture limit will stop us
                await asyncio.Event().wait()

        async def wait_for_capture_limit() -> None:
            """Wait for capture duration limit to be reached."""
            while not capture_limit_reached.is_set():
                await asyncio.sleep(0.1)
            print("Stopping due to capture duration limit.", file=sys.stderr)

        # Create tasks for both conditions
        input_task = asyncio.create_task(wait_for_input())
        limit_task = asyncio.create_task(wait_for_capture_limit())

        # Wait for whichever completes first
        done, pending = await asyncio.wait(
            [input_task, limit_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        # Cancel the task that didn't finish
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

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

                    if final_text:
                        chunk_texts.append(final_text)
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
    with full_audio_lock:
        full_audio = np.concatenate(full_audio_chunks) if full_audio_chunks else np.zeros(0, dtype=np.float32)
    return RealtimeTranscriptionResult(full_audio=full_audio, sample_rate=sample_rate, chunks=chunk_texts)
