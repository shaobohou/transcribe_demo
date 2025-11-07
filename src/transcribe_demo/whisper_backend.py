from __future__ import annotations

import asyncio
import os
import queue
import ssl
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import webrtcvad
import whisper

from transcribe_demo import audio_capture as audio_capture_lib
from transcribe_demo.session_logger import SessionLogger


@dataclass
class TranscriptionChunk:
    """Stores a transcription chunk with its timing information."""

    index: int
    text: str
    start_time: float
    end_time: float
    overlap_start: float  # Start of overlap region from previous chunk
    inference_seconds: float | None = None


@dataclass
class WhisperTranscriptionResult:
    """Aggregate result returned after a Whisper transcription session."""

    full_audio_transcription: str | None
    capture_duration: float = 0.0
    metadata: dict[str, Any] | None = None


def _mps_available() -> bool:
    mps_backend = getattr(torch.backends, "mps", None)
    if not mps_backend or not hasattr(mps_backend, "is_available"):
        return False
    try:
        return bool(mps_backend.is_available())
    except Exception:
        return False


def load_whisper_model(
    model_name: str,
    device_preference: str,
    require_gpu: bool,
    ca_cert: Path | None,
    insecure_downloads: bool,
) -> tuple[whisper.Whisper, str, bool]:
    cuda_available = torch.cuda.is_available()
    apple_mps_available = _mps_available()

    if device_preference == "auto":
        if cuda_available:
            device = "cuda"
        elif apple_mps_available:
            device = "mps"
        else:
            device = "cpu"
    elif device_preference == "cuda":
        if not cuda_available:
            raise RuntimeError("CUDA GPU requested (--device=cuda) but none is available.")
        device = "cuda"
    elif device_preference == "mps":
        if not apple_mps_available:
            raise RuntimeError("Apple Metal GPU requested (--device=mps) but none is available.")
        device = "mps"
    else:
        device = "cpu"

    if require_gpu and device not in {"cuda", "mps"}:
        raise RuntimeError("GPU expected (--require-gpu supplied) but none is available.")

    if device == "cuda":
        print("Running transcription on CUDA GPU.", file=sys.stderr)
    elif device == "mps":
        print("Running transcription on Apple Metal (MPS) GPU.", file=sys.stderr)
    else:
        print(
            "GPU not detected; running on CPU will be significantly slower.",
            file=sys.stderr,
        )

    if ca_cert is not None:
        if not ca_cert.exists():
            raise FileNotFoundError(f"CA bundle not found: {ca_cert}")
        os.environ["SSL_CERT_FILE"] = str(ca_cert)
        os.environ["REQUESTS_CA_BUNDLE"] = str(ca_cert)

    original_https_context = None
    if insecure_downloads:
        original_https_context = ssl._create_default_https_context
        ssl._create_default_https_context = ssl._create_unverified_context

    try:
        preferred_models = {"tiny": "tiny.en", "base": "base.en"}
        effective_model_name = preferred_models.get(model_name, model_name)
        if effective_model_name != model_name:
            print(
                f"Using English-only checkpoint '{effective_model_name}' for faster decoding.",
                file=sys.stderr,
            )
        print(f"Loading Whisper model: {effective_model_name}", file=sys.stderr)
        model = whisper.load_model(effective_model_name, device=device)
    finally:
        if original_https_context is not None:
            ssl._create_default_https_context = original_https_context

    fp16 = device == "cuda"
    return model, device, fp16


# TODO: Add Silero VAD as alternative backend for better noise/music robustness
#
# Problem: WebRTC VAD can only distinguish silence vs voice, causing false positives
# with background music/noise. Silero VAD can distinguish voice vs music vs noise vs silence.
#
# Benefits:
# - 10-30x more accurate with background noise/music
# - Probabilistic scores (adjustable threshold 0.0-1.0)
# - Deep learning-based vs WebRTC's simple GMM
# - Processing: <1ms per 30ms chunk (real-time capable)
# - Supports any sample rate (WebRTC limited to 8k/16k/32k/48k)
#
# Implementation:
# 1. Add dependency: `uv add silero-vad torch`
#
# 2. Create SileroVAD class:
#    class SileroVAD:
#        def __init__(self, sample_rate: int = 16000, threshold: float = 0.5):
#            self.sample_rate = sample_rate
#            self.threshold = threshold
#            self.model, _ = torch.hub.load('snakers4/silero-vad', model='silero_vad')
#            self.frame_size = 512  # samples
#
#        def is_speech(self, audio: np.ndarray) -> bool:
#            if len(audio) < self.frame_size:
#                return False
#            audio_tensor = torch.from_numpy(audio[:self.frame_size])
#            with torch.no_grad():
#                speech_prob = self.model(audio_tensor, self.sample_rate).item()
#            return speech_prob > self.threshold
#
# 3. Add CLI arguments in main.py:
#    --vad-backend {webrtc,silero}  (default: webrtc)
#    --vad-threshold FLOAT          (default: 0.5, range: 0.0-1.0)
#                                    Recommended: 0.7-0.9 for noisy environments
#
# 4. Update run_whisper_transcriber() to accept vad_backend parameter and
#    instantiate appropriate VAD class
#
# 5. Usage for noisy environments:
#    uv run transcribe-demo --vad-backend silero --vad-threshold 0.8
#
# Additional improvements for noise robustness:
# - Add initial_prompt to model.transcribe(): "Ignore background music and noise."
# - Set condition_on_previous_text=False to prevent hallucination loops
#
# References:
# - https://github.com/snakers4/silero-vad
# - https://github.com/snakers4/silero-vad/blob/master/examples/pyaudio-streaming/


class WebRTCVAD:
    """WebRTC VAD wrapper for speech detection."""

    def __init__(self, sample_rate: int, frame_duration_ms: int = 30, aggressiveness: int = 2):
        """
        Initialize WebRTC VAD.

        Args:
            sample_rate: Must be 8000, 16000, 32000, or 48000 Hz
            frame_duration_ms: Frame duration in ms (10, 20, or 30)
            aggressiveness: VAD aggressiveness (0-3, higher = more aggressive filtering)
        """
        if sample_rate not in (8000, 16000, 32000, 48000):
            raise ValueError(f"Sample rate must be 8000, 16000, 32000, or 48000 Hz, got {sample_rate}")
        if frame_duration_ms not in (10, 20, 30):
            raise ValueError(f"Frame duration must be 10, 20, or 30 ms, got {frame_duration_ms}")

        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)
        self.vad = webrtcvad.Vad(aggressiveness)

    def is_speech(self, audio: np.ndarray) -> bool:
        """
        Detect if audio frame contains speech.

        Args:
            audio: Audio samples (float32, mono). Must be exactly one frame length.

        Returns:
            True if speech is detected, False otherwise
        """
        if len(audio) != self.frame_size:
            return False

        # Sanitize audio before conversion; replace non-finite values and clip to int16 range
        clean_audio = np.nan_to_num(
            np.asarray(audio, dtype=np.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
            copy=True,
        )

        # Warn if clipping is needed (indicates potential audio configuration issues)
        if np.any(np.abs(clean_audio) > 1.0):
            import warnings

            warnings.warn(
                "Audio values exceeded [-1.0, 1.0] range, clipping applied. "
                "This may indicate incorrect gain settings or audio driver issues.",
                UserWarning,
                stacklevel=2,
            )
        np.clip(clean_audio, -1.0, 1.0, out=clean_audio)

        # Convert float32 [-1.0, 1.0] to int16 PCM
        audio_int16 = (clean_audio * 32768.0).astype(np.int16)

        # Convert to bytes
        audio_bytes = struct.pack(f"{len(audio_int16)}h", *audio_int16)

        # Run VAD
        try:
            return self.vad.is_speech(audio_bytes, self.sample_rate)
        except Exception:
            return False


def run_whisper_transcriber(
    model_name: str,
    sample_rate: int,
    channels: int,
    temp_file: Path | None,
    ca_cert: Path | None,
    insecure_downloads: bool,
    device_preference: str,
    require_gpu: bool,
    chunk_consumer: Callable[[int, str, float, float, float | None], None] | None = None,
    vad_aggressiveness: int = 2,
    vad_min_silence_duration: float = 0.2,
    vad_min_speech_duration: float = 0.25,
    vad_speech_pad_duration: float = 0.2,
    max_chunk_duration: float = 60.0,
    compare_transcripts: bool = True,
    max_capture_duration: float = 120.0,
    language: str = "en",
    session_logger: SessionLogger | None = None,
    min_log_duration: float = 0.0,
) -> WhisperTranscriptionResult:
    model, device, fp16 = load_whisper_model(
        model_name=model_name,
        device_preference=device_preference,
        require_gpu=require_gpu,
        ca_cert=ca_cert,
        insecure_downloads=insecure_downloads,
    )
    normalized_language = None
    if language and language.lower() != "auto":
        normalized_language = language

    # Track wall-clock start of the transcription session for absolute timestamps
    session_start_time = time.perf_counter()

    # Initialize audio capture manager
    audio_capture = audio_capture_lib.AudioCaptureManager(
        sample_rate=sample_rate,
        channels=channels,
        max_capture_duration=max_capture_duration,
        collect_full_audio=compare_transcripts or (session_logger is not None),
    )

    buffer = np.zeros(0, dtype=np.float32)

    # VAD configuration
    vad = WebRTCVAD(sample_rate=sample_rate, frame_duration_ms=30, aggressiveness=vad_aggressiveness)
    min_chunk_size = int(sample_rate * 2.0)  # Minimum 2 seconds to avoid transcribing short noise
    max_chunk_size = int(sample_rate * max_chunk_duration)
    silence_frames_threshold = int(sample_rate * vad_min_silence_duration)
    min_speech_frames = int(sample_rate * vad_min_speech_duration)
    speech_pad_samples = int(sample_rate * vad_speech_pad_duration)

    if max_capture_duration > 0:
        print(
            f"Using WebRTC VAD-based chunking (min: 2.0s, max: {max_chunk_duration}s, "
            f"aggressiveness: {vad_aggressiveness}, min_speech: {vad_min_speech_duration}s, "
            f"pad: {vad_speech_pad_duration}s, max_capture: {max_capture_duration}s)",
            file=sys.stderr,
        )
    else:
        print(
            f"Using WebRTC VAD-based chunking (min: 2.0s, max: {max_chunk_duration}s, "
            f"aggressiveness: {vad_aggressiveness}, min_speech: {vad_min_speech_duration}s, "
            f"pad: {vad_speech_pad_duration}s)",
            file=sys.stderr,
        )

    transcription_queue: asyncio.Queue[tuple[int, np.ndarray, float] | None] = asyncio.Queue()
    transcriber_error: list[BaseException] = []

    async def vad_worker() -> None:
        nonlocal buffer
        chunk_index = 0
        silence_frames = 0
        speech_frames = 0
        speech_pad_buffer = np.zeros(0, dtype=np.float32)
        vad_frame_buffer = np.zeros(0, dtype=np.float32)
        next_chunk_start = 0.0

        try:
            while not audio_capture.stop_event.is_set():
                try:
                    chunk = await asyncio.to_thread(audio_capture.audio_queue.get, True, 0.1)
                except queue.Empty:
                    await asyncio.sleep(0.05)
                    continue

                # Reduce to mono when multiple channels are present.
                force_transcribe_now = False
                if chunk is None:
                    force_transcribe_now = True
                    mono = np.zeros(0, dtype=np.float32)
                else:
                    mono = chunk.mean(axis=1).astype(np.float32, copy=False)

                buffer = np.concatenate((buffer, mono))
                speech_pad_buffer = np.concatenate((speech_pad_buffer, mono))

                # Process audio in VAD frames (30ms = 480 samples at 16kHz)
                vad_frame_buffer = np.concatenate((vad_frame_buffer, mono))

                # Process complete VAD frames
                while len(vad_frame_buffer) >= vad.frame_size:
                    frame = vad_frame_buffer[: vad.frame_size]
                    vad_frame_buffer = vad_frame_buffer[vad.frame_size :]

                    # Check if frame contains speech
                    if vad.is_speech(frame):
                        # Reset silence counter and accumulate speech
                        silence_frames = 0
                        speech_frames += vad.frame_size
                    else:
                        # Accumulate silence
                        silence_frames += vad.frame_size

                # Determine if we should transcribe
                should_transcribe = False
                force_flush = force_transcribe_now
                max_duration_exceeded = False

                # When time limit is reached, transcribe immediately without waiting for VAD pause
                if audio_capture.capture_limit_reached.is_set() and buffer.size > 0:
                    should_transcribe = True
                    force_flush = True
                # Normal VAD-based chunking
                elif buffer.size >= min_chunk_size and speech_frames >= min_speech_frames:
                    if silence_frames >= silence_frames_threshold:
                        should_transcribe = True
                    elif buffer.size >= max_chunk_size:
                        should_transcribe = True
                        max_duration_exceeded = True

                if force_transcribe_now:
                    should_transcribe = True
                    force_flush = True

                if not should_transcribe:
                    continue

                # Trim trailing silence
                trim_samples = min(silence_frames, buffer.size)
                if trim_samples > 0:
                    window = buffer[:-trim_samples] if trim_samples < buffer.size else buffer
                else:
                    window = buffer

                # Add speech padding from the circular buffer
                if len(speech_pad_buffer) > speech_pad_samples:
                    pad_start_idx = max(0, len(speech_pad_buffer) - len(window) - speech_pad_samples)
                    padding = speech_pad_buffer[pad_start_idx : len(speech_pad_buffer) - len(window)]
                    if len(padding) > 0:
                        window = np.concatenate((padding, window))

                if len(window) < min_chunk_size:
                    if not force_flush:
                        continue
                    # Allow final partial chunk; avoid empty buffers
                    if len(window) == 0:
                        if audio_capture.capture_limit_reached.is_set():
                            audio_capture.stop()
                        continue

                chunk_audio_duration = len(window) / float(sample_rate)
                current_end = next_chunk_start + chunk_audio_duration

                # Log warning if max chunk duration was exceeded
                if max_duration_exceeded:
                    print(
                        f"WARNING: Chunk split due to max duration exceeded "
                        f"({chunk_audio_duration:.2f}s >= {max_chunk_duration}s). "
                        f"Consider increasing --max-chunk-duration or check for continuous speech without pauses.",
                        file=sys.stderr,
                    )

                if temp_file is not None:
                    np.save(temp_file, window)

                # Enqueue chunk for transcription
                try:
                    await transcription_queue.put((chunk_index, window.copy(), chunk_audio_duration))
                except Exception as exc:
                    transcriber_error.append(exc)
                    audio_capture.stop()
                    break

                # Clear buffer completely (no overlap with VAD)
                buffer = np.zeros(0, dtype=np.float32)
                silence_frames = 0
                speech_frames = 0
                next_chunk_start = current_end

                # Keep speech_pad_buffer size bounded to avoid excessive memory usage
                max_pad_buffer_size = speech_pad_samples * 4  # Keep 4x padding size as history
                if len(speech_pad_buffer) > max_pad_buffer_size:
                    speech_pad_buffer = speech_pad_buffer[-max_pad_buffer_size:]

                chunk_index += 1

                # After chunk completes, check if capture limit was reached and stop gracefully
                if audio_capture.capture_limit_reached.is_set():
                    print(
                        "Current chunk finished. Stopping transcription...",
                        file=sys.stderr,
                    )
                    audio_capture.stop()
                    break
        finally:
            await transcription_queue.put(None)

    async def transcriber_worker() -> None:
        try:
            while True:
                item = await transcription_queue.get()
                if item is None:
                    break

                chunk_index, window, chunk_audio_duration = item

                inference_start = time.perf_counter()
                result: dict[str, Any] = await asyncio.to_thread(
                    model.transcribe,
                    window,
                    fp16=fp16,
                    temperature=0.0,
                    beam_size=1,
                    best_of=1,
                    language=normalized_language,
                )
                inference_duration = time.perf_counter() - inference_start
                raw_text = result.get("text", "")
                if isinstance(raw_text, str):
                    text = raw_text
                elif isinstance(raw_text, list):
                    text = " ".join(str(part) for part in raw_text)
                else:
                    text = str(raw_text)

                # Compute absolute timestamps relative to session start (approximate real-time)
                chunk_absolute_end = max(0.0, inference_start - session_start_time)
                chunk_absolute_start = max(0.0, chunk_absolute_end - chunk_audio_duration)

                if chunk_consumer is not None:
                    chunk_consumer(
                        chunk_index,
                        text,
                        chunk_absolute_start,
                        chunk_absolute_end,
                        inference_duration,
                    )
                else:
                    print(
                        f"[chunk {chunk_index:03d} | t={chunk_absolute_end:.2f}s | "
                        f"audio: {chunk_audio_duration:.2f}s | inference: {inference_duration:.2f}s] {text}",
                        flush=True,
                    )

                if session_logger is not None:
                    session_logger.log_chunk(
                        index=chunk_index,
                        text=text,
                        start_time=chunk_absolute_start,
                        end_time=chunk_absolute_end,
                        inference_seconds=inference_duration,
                        audio=window.copy() if session_logger.save_chunk_audio else None,
                    )
        except Exception as exc:  # pragma: no cover - defensive guard
            transcriber_error.append(exc)
            audio_capture.stop()
            await transcription_queue.put(None)

    async def orchestrate() -> None:
        builder_task = asyncio.create_task(vad_worker())
        transcriber_task = asyncio.create_task(transcriber_worker())
        try:
            audio_capture.start()
            await asyncio.to_thread(audio_capture.wait_until_stopped)
        finally:
            audio_capture.stop()
            await asyncio.gather(builder_task, transcriber_task, return_exceptions=True)
            audio_capture.close()
            if temp_file is not None and temp_file.exists():
                temp_file.unlink()

    def run_asyncio_pipeline() -> None:
        try:
            asyncio.run(orchestrate())
        except RuntimeError:
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(orchestrate())
            finally:
                loop.close()

    run_asyncio_pipeline()

    if transcriber_error:
        raise RuntimeError(f"Transcription worker error: {transcriber_error[0]}") from transcriber_error[0]

    # Get full audio for comparison
    full_audio_transcription: str | None = None
    full_audio = audio_capture.get_full_audio()
    if compare_transcripts and full_audio.size:
        full_audio_result: dict[str, Any] = model.transcribe(
            full_audio,
            fp16=fp16,
            temperature=0.0,
            beam_size=1,
            best_of=1,
            language=normalized_language,
        )
        full_audio_raw = full_audio_result.get("text", "")
        if isinstance(full_audio_raw, str):
            full_audio_transcription = full_audio_raw
        elif isinstance(full_audio_raw, list):
            full_audio_transcription = " ".join(str(part) for part in full_audio_raw)
        else:
            full_audio_transcription = str(full_audio_raw)

    # Save full audio for session logging (finalization happens in main.py with stitched transcription)
    if session_logger is not None:
        session_logger.save_full_audio(full_audio, audio_capture.get_capture_duration())

    return WhisperTranscriptionResult(
        full_audio_transcription=full_audio_transcription,
        capture_duration=audio_capture.get_capture_duration(),
        metadata={
            "model": model_name,
            "device": device,
            "language": normalized_language or "auto",
            "vad_aggressiveness": vad_aggressiveness,
            "vad_min_silence_duration": vad_min_silence_duration,
            "vad_min_speech_duration": vad_min_speech_duration,
            "vad_speech_pad_duration": vad_speech_pad_duration,
            "max_chunk_duration": max_chunk_duration,
        },
    )


def transcribe_full_audio(
    audio: np.ndarray,
    sample_rate: int,
    model_name: str,
    device_preference: str,
    require_gpu: bool,
    ca_cert: Path | None,
    insecure_downloads: bool,
    language: str = "en",
) -> str:
    """
    Run a single transcription pass over the provided audio buffer.

    Args:
        audio: Mono float32 audio samples.
        sample_rate: Sample rate of the provided audio (Hz).
        model_name: Whisper checkpoint name (e.g., turbo, small.en).
        device_preference: Preferred device to run on (auto/cpu/cuda/mps).
        require_gpu: If True, raise when GPU unavailable.
        ca_cert: Optional custom CA bundle for downloads.
        insecure_downloads: Disable SSL verification for downloads when True.

    Returns:
        Full transcription text (may be empty when no audio).
    """
    if audio.size == 0:
        return ""

    # Whisper handles resampling internally; sample_rate retained for interface symmetry.
    _ = sample_rate

    model, _, fp16 = load_whisper_model(
        model_name=model_name,
        device_preference=device_preference,
        require_gpu=require_gpu,
        ca_cert=ca_cert,
        insecure_downloads=insecure_downloads,
    )

    normalized_language = None
    if language and language.lower() != "auto":
        normalized_language = language

    result: dict[str, Any] = model.transcribe(
        audio,
        fp16=fp16,
        temperature=0.0,
        beam_size=1,
        best_of=1,
        language=normalized_language,
    )
    raw_text = result.get("text", "")
    if isinstance(raw_text, str):
        return raw_text
    if isinstance(raw_text, list):
        return " ".join(str(part) for part in raw_text)
    return str(raw_text)
