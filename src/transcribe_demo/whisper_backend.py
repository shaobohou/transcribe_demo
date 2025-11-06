from __future__ import annotations

import os
import queue
import ssl
import struct
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional
from types import ModuleType

import numpy as np
import torch
import webrtcvad
import whisper


_SOUNDDEVICE_MODULE: ModuleType | None = None


def _get_sounddevice() -> ModuleType:
    global _SOUNDDEVICE_MODULE
    if _SOUNDDEVICE_MODULE is None:
        import sounddevice as sd  # pylint: disable=import-outside-toplevel

        _SOUNDDEVICE_MODULE = sd
    return _SOUNDDEVICE_MODULE


@dataclass
class TranscriptionChunk:
    """Stores a transcription chunk with its timing information."""
    index: int
    text: str
    start_time: float
    end_time: float
    overlap_start: float  # Start of overlap region from previous chunk
    inference_seconds: float | None = None


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
            nan=0.0, posinf=0.0, neginf=0.0, copy=True
        )

        # Warn if clipping is needed (indicates potential audio configuration issues)
        if np.any(np.abs(clean_audio) > 1.0):
            import warnings
            warnings.warn(
                "Audio values exceeded [-1.0, 1.0] range, clipping applied. "
                "This may indicate incorrect gain settings or audio driver issues.",
                UserWarning,
                stacklevel=2
            )
        np.clip(clean_audio, -1.0, 1.0, out=clean_audio)

        # Convert float32 [-1.0, 1.0] to int16 PCM
        audio_int16 = (clean_audio * 32768.0).astype(np.int16)

        # Convert to bytes
        audio_bytes = struct.pack(f'{len(audio_int16)}h', *audio_int16)

        # Run VAD
        try:
            return self.vad.is_speech(audio_bytes, self.sample_rate)
        except Exception:
            return False


def run_whisper_transcriber(
    model_name: str,
    sample_rate: int,
    channels: int,
    temp_file: Optional[Path],
    ca_cert: Optional[Path],
    insecure_downloads: bool,
    device_preference: str,
    require_gpu: bool,
    chunk_consumer: Optional[Callable[[int, str, float, float, list | None, float], None]] = None,
    vad_aggressiveness: int = 2,
    vad_min_silence_duration: float = 0.2,
    vad_min_speech_duration: float = 0.25,
    vad_speech_pad_duration: float = 0.2,
    max_chunk_duration: float = 60.0,
) -> None:
    def mps_available() -> bool:
        mps_backend = getattr(torch.backends, "mps", None)
        if not mps_backend or not hasattr(mps_backend, "is_available"):
            return False
        try:
            return bool(mps_backend.is_available())
        except Exception:
            return False

    cuda_available = torch.cuda.is_available()
    apple_mps_available = mps_available()

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

    # Track wall-clock start of the transcription session for absolute timestamps
    session_start_time = time.perf_counter()

    audio_queue: queue.Queue[np.ndarray] = queue.Queue()
    buffer = np.zeros(0, dtype=np.float32)
    lock = threading.Lock()

    # VAD configuration
    vad = WebRTCVAD(sample_rate=sample_rate, frame_duration_ms=30, aggressiveness=vad_aggressiveness)
    min_chunk_size = int(sample_rate * 2.0)  # Minimum 2 seconds to avoid transcribing short noise
    max_chunk_size = int(sample_rate * max_chunk_duration)
    silence_frames_threshold = int(sample_rate * vad_min_silence_duration)
    min_speech_frames = int(sample_rate * vad_min_speech_duration)
    speech_pad_samples = int(sample_rate * vad_speech_pad_duration)
    print(f"Using WebRTC VAD-based chunking (min: 2.0s, max: {max_chunk_duration}s, aggressiveness: {vad_aggressiveness}, min_speech: {vad_min_speech_duration}s, pad: {vad_speech_pad_duration}s)", file=sys.stderr)

    def callback(indata: np.ndarray, frames: int, time, status) -> None:
        if status:
            print(f"InputStream status: {status}", file=sys.stderr)
        # Copy to avoid referencing the sounddevice ring buffer.
        audio_queue.put(indata.copy())

    def worker(stop_event: threading.Event) -> None:
        nonlocal buffer
        chunk_index = 0
        next_chunk_start = 0.0
        silence_frames = 0
        speech_frames = 0
        speech_pad_buffer = np.zeros(0, dtype=np.float32)
        vad_frame_buffer = np.zeros(0, dtype=np.float32)

        while not stop_event.is_set():
            try:
                chunk = audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # Reduce to mono when multiple channels are present.
            mono = chunk.mean(axis=1).astype(np.float32, copy=False)

            with lock:
                buffer = np.concatenate((buffer, mono))
                speech_pad_buffer = np.concatenate((speech_pad_buffer, mono))

                # Process audio in VAD frames (30ms = 480 samples at 16kHz)
                vad_frame_buffer = np.concatenate((vad_frame_buffer, mono))

                # Process complete VAD frames
                while len(vad_frame_buffer) >= vad.frame_size:
                    frame = vad_frame_buffer[:vad.frame_size]
                    vad_frame_buffer = vad_frame_buffer[vad.frame_size:]

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
                max_duration_exceeded = False
                if buffer.size >= min_chunk_size and speech_frames >= min_speech_frames:
                    if silence_frames >= silence_frames_threshold:
                        should_transcribe = True
                    elif buffer.size >= max_chunk_size:
                        should_transcribe = True
                        max_duration_exceeded = True

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
                    padding = speech_pad_buffer[pad_start_idx:len(speech_pad_buffer) - len(window)]
                    if len(padding) > 0:
                        window = np.concatenate((padding, window))

                if len(window) < min_chunk_size:
                    continue

                current_start = next_chunk_start
                current_end = next_chunk_start + len(window) / float(sample_rate)
                chunk_audio_duration = current_end - current_start

                # Log warning if max chunk duration was exceeded
                if max_duration_exceeded:
                    print(
                        f"WARNING: Chunk split due to max duration exceeded ({chunk_audio_duration:.2f}s >= {max_chunk_duration}s). "
                        f"Consider increasing --max-chunk-duration or check for continuous speech without pauses.",
                        file=sys.stderr,
                    )

                if temp_file is not None:
                    np.save(temp_file, window)

                inference_start = time.perf_counter()
                result = model.transcribe(
                    window,
                    fp16=fp16,
                    temperature=0.0,
                    beam_size=1,
                    best_of=1,
                    language="en",  # Force English to prevent hallucinations in other languages
                )
                inference_duration = time.perf_counter() - inference_start
                text = result["text"]

                # Compute absolute timestamps relative to session start
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
                        f"[chunk {chunk_index:03d} | t={chunk_absolute_end:.2f}s | audio: {chunk_audio_duration:.2f}s | inference: {inference_duration:.2f}s] {text}",
                        flush=True,
                    )

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

    stop_event = threading.Event()
    thread = threading.Thread(target=worker, args=(stop_event,), daemon=True)
    thread.start()

    try:
        sd = _get_sounddevice()
        with sd.InputStream(
            callback=callback,
            channels=channels,
            samplerate=sample_rate,
            dtype="float32",
        ):
            try:
                input("Press Enter to stop...\n")
            except (EOFError, OSError):
                # If stdin is not available (e.g., running in background), just wait
                while not stop_event.is_set():
                    time.sleep(0.1)
    finally:
        stop_event.set()
        thread.join(timeout=1.0)
        if temp_file is not None:
            with lock:
                if temp_file.exists():
                    temp_file.unlink()
