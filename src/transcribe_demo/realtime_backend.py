"""Realtime transcription backend using OpenAI Realtime API."""

from __future__ import annotations

import asyncio
import base64
import json
import queue
import ssl
import sys
import threading
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import websockets

if TYPE_CHECKING:
    import websockets.asyncio.client

from transcribe_demo import audio_capture as audio_capture_lib
from transcribe_demo.file_audio_source import FileAudioSource
from transcribe_demo.session_logger import SessionLogger


class ChunkConsumer(Protocol):
    def __call__(
        self,
        *,
        chunk_index: int,
        text: str,
        absolute_start: float,
        absolute_end: float,
        inference_seconds: float | None,
    ) -> None: ...


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
    chunks: list[str] | None = None
    capture_duration: float = 0.0
    metadata: dict[str, Any] | None = None


async def _send_json(
    ws: websockets.asyncio.client.ClientConnection,
    payload: dict[str, Any],
    lock: asyncio.Lock,
) -> None:
    """Send a JSON payload through the websocket using the provided lock."""

    message = json.dumps(payload)
    async with lock:
        await ws.send(message)


def _create_session_update(
    instructions: str,
    transcription_config: dict[str, Any],
    *,
    include_turn_detection: bool,
    vad_threshold: float = 0.3,
    vad_silence_duration_ms: int = 200,
) -> dict[str, Any]:
    """Build the session.update payload for realtime websocket communication."""

    session: dict[str, Any] = {
        "modalities": ["text"],
        "instructions": instructions,
        "input_audio_format": "pcm16",
        "input_audio_transcription": transcription_config,
        "temperature": 0.6,
        "max_response_output_tokens": 4096,
    }
    if include_turn_detection:
        session["turn_detection"] = {
            "type": "server_vad",
            "threshold": vad_threshold,
            "prefix_padding_ms": 200,
            "silence_duration_ms": vad_silence_duration_ms,
        }
    return {"type": "session.update", "session": session}


def _run_async(coro_factory: Callable[[], Coroutine[Any, Any, str]]) -> str:
    try:
        return asyncio.run(coro_factory())
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro_factory())
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
    disable_ssl_verify: bool = False,
    language: str = "en",
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
        ssl_context: ssl.SSLContext | None = None
        if disable_ssl_verify:
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE

        lock = asyncio.Lock()
        partials: dict[str, str] = {}
        completed: list[str] = []
        processed_items: set[str] = set()

        language_value = (language or "").strip()
        transcription_config: dict[str, Any] = {"model": "whisper-1"}
        if language_value and language_value.lower() != "auto":
            transcription_config["language"] = language_value

        async with websockets.connect(uri, additional_headers=headers, max_size=None, ssl=ssl_context) as ws:
            await _send_json(
                ws,
                _create_session_update(
                    instructions,
                    transcription_config,
                    include_turn_detection=False,
                ),
                lock,
            )

            cursor = 0
            total_samples = audio.size
            chunks_sent = 0
            while cursor < total_samples:
                window = audio[cursor : cursor + chunk_size]
                cursor += chunk_size
                resampled = resample_audio(window, sample_rate, session_sample_rate)
                if resampled.size == 0:
                    continue
                payload = base64.b64encode(float_to_pcm16(resampled)).decode("ascii")
                await _send_json(
                    ws,
                    {
                        "type": "input_audio_buffer.append",
                        "audio": payload,
                    },
                    lock,
                )
                chunks_sent += 1

            # Only commit if we actually sent audio chunks
            if chunks_sent == 0:
                return ""

            await _send_json(ws, {"type": "input_audio_buffer.commit"}, lock)

            committed_received = False
            post_commit_timeout = 1.0  # Wait 1 second after commit for final transcriptions

            while True:
                try:
                    # Use timeout after commit to detect when server is done sending transcriptions
                    timeout = post_commit_timeout if committed_received else 2.0
                    message = await asyncio.wait_for(ws.recv(), timeout=timeout)
                except asyncio.TimeoutError:
                    # After commit, if we timeout and have partials but no completed, flush partials
                    if committed_received and partials:
                        # Server VAD didn't complete these chunks, use partial transcriptions
                        for item_id, partial_text in partials.items():
                            final_text = partial_text.strip()
                            if final_text and (not completed or final_text != completed[-1]):
                                completed.append(final_text)
                        break
                    elif completed and not partials:
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
                    # Mark that commit was received, start timeout for final transcriptions
                    committed_received = True
                    if completed and not partials:
                        break
                elif event_type == "session.closed":
                    break
                elif event_type and event_type.startswith("error"):
                    raise RuntimeError(f"Realtime error: {payload}")

            await ws.close()
        return " ".join(completed).strip()

    return _run_async(_transcribe)


def run_realtime_transcriber(
    api_key: str,
    endpoint: str,
    model: str,
    sample_rate: int,
    channels: int,
    chunk_duration: float,
    instructions: str,
    disable_ssl_verify: bool = False,
    chunk_consumer: ChunkConsumer | None = None,
    compare_transcripts: bool = True,
    max_capture_duration: float = 120.0,
    language: str = "en",
    session_logger: SessionLogger | None = None,
    min_log_duration: float = 0.0,
    audio_file: Path | None = None,
    playback_speed: float = 1.0,
    vad_threshold: float = 0.3,
    vad_silence_duration_ms: int = 200,
    debug: bool = False,
) -> RealtimeTranscriptionResult:
    # Initialize audio source (either from file or microphone)
    if audio_file is not None:
        audio_capture = FileAudioSource(
            audio_file=audio_file,
            sample_rate=sample_rate,
            channels=channels,
            max_capture_duration=max_capture_duration,
            collect_full_audio=compare_transcripts or (session_logger is not None),
            playback_speed=playback_speed,
        )
    else:
        audio_capture = audio_capture_lib.AudioCaptureManager(
            sample_rate=sample_rate,
            channels=channels,
            max_capture_duration=max_capture_duration,
            collect_full_audio=compare_transcripts or (session_logger is not None),
        )

    session_sample_rate = 24000
    chunk_texts: list[str] = []
    language_value = (language or "").strip()

    # Shared state between threads
    chunk_counter_lock = threading.Lock()
    chunk_counter = [0]  # Use list for mutability across threads
    session_start_time = time.perf_counter()
    cumulative_time = [0.0]  # Use list for mutability across threads

    def websocket_worker() -> None:
        """Worker thread that runs the async websocket communication."""

        async def run_websocket() -> None:
            uri = f"{endpoint}?model={model}"
            headers = [
                ("Authorization", f"Bearer {api_key}"),
                ("OpenAI-Beta", "realtime=v1"),
            ]

            ssl_context: ssl.SSLContext | None = None
            if disable_ssl_verify:
                ssl_context = ssl.create_default_context()
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE

            transcription_config: dict[str, Any] = {"model": "whisper-1"}
            if language_value and language_value.lower() != "auto":
                transcription_config["language"] = language_value

            lock = asyncio.Lock()
            partials: dict[str, str] = {}

            try:
                async with websockets.connect(uri, additional_headers=headers, max_size=None, ssl=ssl_context) as ws:
                    # Session setup
                    await _send_json(
                        ws,
                        _create_session_update(
                            instructions,
                            transcription_config,
                            include_turn_detection=True,
                            vad_threshold=vad_threshold,
                            vad_silence_duration_ms=vad_silence_duration_ms,
                        ),
                        lock,
                    )

                    # Create tasks for sending and receiving
                    async def audio_sender() -> None:
                        buffer = np.zeros(0, dtype=np.float32)
                        chunk_size = max(int(sample_rate * chunk_duration), 1)
                        sentinel_received = False

                        try:
                            while True:
                                if audio_capture.stop_event.is_set() and sentinel_received:
                                    break

                                # Get audio from queue
                                force_flush = False
                                got_item = False
                                item: np.ndarray | None = None

                                if not sentinel_received and buffer.size < chunk_size:
                                    try:
                                        timeout = 0.1 if not audio_capture.stop_event.is_set() else 0.01
                                        item = await asyncio.to_thread(audio_capture.audio_queue.get, True, timeout)
                                        got_item = True
                                    except queue.Empty:
                                        got_item = False

                                if got_item:
                                    if item is None:
                                        sentinel_received = True
                                        force_flush = True
                                    else:
                                        chunk = item
                                        if chunk.ndim > 1:
                                            chunk = chunk.mean(axis=1)
                                        buffer = np.concatenate((buffer, chunk.astype(np.float32, copy=False)))
                                else:
                                    if (
                                        audio_capture.stop_event.is_set()
                                        or audio_capture.capture_limit_reached.is_set()
                                    ):
                                        force_flush = buffer.size > 0 or sentinel_received
                                    elif buffer.size < chunk_size:
                                        await asyncio.sleep(0.05)
                                        continue

                                if buffer.size == 0:
                                    if (
                                        sentinel_received
                                        or audio_capture.stop_event.is_set()
                                        or audio_capture.capture_limit_reached.is_set()
                                    ):
                                        break
                                    continue

                                send_count = buffer.size if force_flush or buffer.size < chunk_size else chunk_size
                                if send_count == 0:
                                    if (
                                        sentinel_received
                                        or audio_capture.stop_event.is_set()
                                        or audio_capture.capture_limit_reached.is_set()
                                    ):
                                        break
                                    continue

                                window = buffer[:send_count]
                                buffer = buffer[send_count:]

                                resampled = resample_audio(window, sample_rate, session_sample_rate)
                                if len(resampled) == 0:
                                    if force_flush and buffer.size == 0:
                                        break
                                    continue

                                pcm_payload = base64.b64encode(float_to_pcm16(resampled)).decode("ascii")
                                await _send_json(
                                    ws,
                                    {
                                        "type": "input_audio_buffer.append",
                                        "audio": pcm_payload,
                                    },
                                    lock,
                                )

                                if force_flush and buffer.size == 0:
                                    break

                            # Send any remaining buffer
                            if buffer.size > 0:
                                resampled = resample_audio(buffer, sample_rate, session_sample_rate)
                                if len(resampled) > 0:
                                    pcm_payload = base64.b64encode(float_to_pcm16(resampled)).decode("ascii")
                                    await _send_json(
                                        ws,
                                        {
                                            "type": "input_audio_buffer.append",
                                            "audio": pcm_payload,
                                        },
                                        lock,
                                    )

                            await _send_json(ws, {"type": "input_audio_buffer.commit"}, lock)
                        except Exception as e:
                            print(f"Audio sender error: {e}", file=sys.stderr)
                            raise

                    async def receiver() -> None:
                        committed_received = False
                        post_commit_timeout = 1.0  # Wait 1 second after commit for final transcriptions

                        def flush_remaining_partials() -> None:
                            """Flush any remaining partial transcriptions that haven't been completed."""
                            for item_id, partial_text in list(partials.items()):
                                final_text = partial_text.strip()
                                if not final_text:
                                    continue

                                # Track absolute timestamp from session start
                                with chunk_counter_lock:
                                    chunk_start = cumulative_time[0]
                                    chunk_end = time.perf_counter() - session_start_time
                                    cumulative_time[0] = chunk_end
                                    current_chunk_index = chunk_counter[0]
                                    chunk_counter[0] += 1

                                if chunk_consumer:
                                    chunk_consumer(
                                        chunk_index=current_chunk_index,
                                        text=final_text,
                                        absolute_start=chunk_start,
                                        absolute_end=chunk_end,
                                        inference_seconds=None,  # Signals realtime mode
                                    )
                                else:
                                    label = f"[chunk {current_chunk_index:03d} | {chunk_end:.2f}s]"
                                    print(f"{label} {final_text}", flush=True)

                                chunk_texts.append(final_text)

                                # Log to session logger if enabled
                                if session_logger is not None:
                                    session_logger.log_chunk(
                                        index=current_chunk_index,
                                        text=final_text,
                                        start_time=chunk_start,
                                        end_time=chunk_end,
                                        inference_seconds=None,
                                        audio=None,
                                    )

                        try:
                            while True:
                                try:
                                    # Use timeout after commit to detect when server is done sending transcriptions
                                    timeout = post_commit_timeout if committed_received else None
                                    message = await asyncio.wait_for(ws.recv(), timeout=timeout)
                                except asyncio.TimeoutError:
                                    # After commit, if we timeout waiting for more messages, flush any remaining partials
                                    if committed_received and partials:
                                        flush_remaining_partials()
                                    break
                                except websockets.ConnectionClosed:
                                    # Flush any remaining partials on connection close
                                    if partials:
                                        flush_remaining_partials()
                                    break

                                payload = json.loads(message)
                                event_type = payload.get("type")

                                if event_type == "conversation.item.input_audio_transcription.delta":
                                    item_id = payload.get("item_id")
                                    delta = payload.get("delta") or ""
                                    if delta and item_id:
                                        partials[item_id] = partials.get(item_id, "") + delta
                                        if debug:
                                            current_partial = partials[item_id]
                                            print(
                                                f"[DEBUG] Delta event for {item_id}: partial length={len(current_partial)}, delta='{delta}'",
                                                file=sys.stderr,
                                                flush=True,
                                            )

                                elif event_type == "conversation.item.input_audio_transcription.completed":
                                    if debug:
                                        print(
                                            f"[DEBUG] Completed event received: item_id={payload.get('item_id')}, "
                                            f"transcript_length={len(payload.get('transcript') or '')}",
                                            file=sys.stderr,
                                            flush=True,
                                        )
                                    item_id = payload.get("item_id")
                                    transcript = payload.get("transcript") or ""
                                    had_partials = bool(item_id and partials.get(item_id))
                                    final_text = transcript if transcript else ""
                                    if not final_text and had_partials and item_id:
                                        final_text = partials.get(item_id, "")
                                    final_text = final_text.strip()

                                    # Track absolute timestamp from session start
                                    with chunk_counter_lock:
                                        chunk_start = cumulative_time[0]
                                        chunk_end = time.perf_counter() - session_start_time
                                        cumulative_time[0] = chunk_end
                                        current_chunk_index = chunk_counter[0]
                                        chunk_counter[0] += 1

                                    if chunk_consumer:
                                        chunk_consumer(
                                            chunk_index=current_chunk_index,
                                            text=final_text,
                                            absolute_start=chunk_start,
                                            absolute_end=chunk_end,
                                            inference_seconds=None,  # Signals realtime mode
                                        )
                                    else:
                                        label = f"[chunk {current_chunk_index:03d} | {chunk_end:.2f}s]"
                                        if final_text:
                                            print(f"{label} {final_text}", flush=True)
                                        else:
                                            print(label, flush=True)

                                    if final_text:
                                        chunk_texts.append(final_text)

                                    # Log to session logger if enabled
                                    if session_logger is not None:
                                        session_logger.log_chunk(
                                            index=current_chunk_index,
                                            text=final_text,
                                            start_time=chunk_start,
                                            end_time=chunk_end,
                                            inference_seconds=None,
                                            audio=None,
                                        )

                                    if item_id:
                                        partials.pop(item_id, None)

                                elif event_type == "session.input_audio_buffer.committed":
                                    # Mark that commit was received, start timeout for final transcriptions
                                    committed_received = True

                                elif event_type == "error":
                                    message_text = payload.get("error") or payload.get("message") or payload
                                    print(f"\nRealtime error: {message_text}", file=sys.stderr)
                                elif event_type == "error.session":
                                    message_text = payload.get("message") or payload
                                    print(f"\nRealtime session error: {message_text}", file=sys.stderr)

                                # Check stop_event after processing message
                                if audio_capture.stop_event.is_set():
                                    # Flush any remaining partials before stopping
                                    if partials:
                                        flush_remaining_partials()
                                    break
                        except Exception as e:
                            print(f"Receiver error: {e}", file=sys.stderr)
                            raise

                    # Run sender and receiver concurrently
                    sender_task = asyncio.create_task(audio_sender())
                    receiver_task = asyncio.create_task(receiver())

                    # Wait for both to complete or stop event
                    done, pending = await asyncio.wait(
                        [sender_task, receiver_task], return_when=asyncio.FIRST_EXCEPTION
                    )

                    # Cancel any pending tasks
                    for task in pending:
                        task.cancel()

                    # Wait for cancelled tasks to finish
                    await asyncio.gather(*pending, return_exceptions=True)

                    # Check for exceptions in completed tasks
                    for task in done:
                        exception = task.exception()
                        if exception is not None:
                            raise exception

                    # Close websocket gracefully
                    try:
                        await ws.close()
                    except Exception:
                        pass

            except Exception as e:
                print(f"Websocket error: {e}", file=sys.stderr)
                audio_capture.stop()

        # Run the async websocket in this thread
        try:
            asyncio.run(run_websocket())
        except KeyboardInterrupt:
            pass
        finally:
            audio_capture.stop()

    # Start websocket worker thread
    ws_thread = threading.Thread(target=websocket_worker, daemon=True)
    ws_thread.start()

    # Start audio capture
    try:
        audio_capture.start()
        audio_capture.wait_until_stopped()
    except KeyboardInterrupt:
        pass
    finally:
        audio_capture.stop()
        ws_thread.join(timeout=2.0)
        audio_capture.close()

    # Get full audio
    full_audio = audio_capture.get_full_audio()

    # Save full audio for session logging (finalization happens in main.py with stitched transcription)
    if session_logger is not None:
        session_logger.save_full_audio(full_audio, audio_capture.get_capture_duration())

    return RealtimeTranscriptionResult(
        full_audio=full_audio,
        sample_rate=sample_rate,
        chunks=chunk_texts,
        capture_duration=audio_capture.get_capture_duration(),
        metadata={
            "model": model,
            "realtime_endpoint": endpoint,
            "realtime_instructions": instructions,
            "language": language_value or "auto",
        },
    )
