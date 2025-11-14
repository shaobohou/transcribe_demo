from __future__ import annotations

import difflib
import os
import queue
import re
import sys
import threading
from collections.abc import Generator
from pathlib import Path
from typing import TextIO

from absl import app
from absl import flags

from transcribe_demo.audio_capture import AudioCaptureManager
from transcribe_demo.backend_protocol import AudioSource, TranscriptionChunk
from transcribe_demo.file_audio_source import FileAudioSource
from transcribe_demo.realtime_backend import (
    RealtimeBackend,
    RealtimeTranscriptionResult,
    run_realtime_transcriber,
    transcribe_full_audio_realtime,
)
from transcribe_demo.session_logger import SessionLogger
from transcribe_demo.whisper_backend import WhisperBackend, WhisperTranscriptionResult, run_whisper_transcriber


REALTIME_CHUNK_DURATION = 2.0

FLAGS = flags.FLAGS

# Backend configuration
flags.DEFINE_enum(
    "backend",
    "whisper",
    ["whisper", "realtime"],
    "Transcription backend to use.",
)

# API configuration
flags.DEFINE_string(
    "api_key",
    None,
    "OpenAI API key for realtime transcription. Defaults to OPENAI_API_KEY.",
)

# Whisper model configuration
flags.DEFINE_string(
    "model",
    "turbo",
    "Whisper checkpoint to load. Recommended: 'turbo' (default, GPU, multilingual) or 'base.en' (CPU-friendly, English-only).",
)
flags.DEFINE_enum(
    "device",
    "auto",
    ["auto", "cpu", "cuda", "mps"],
    "Device to run Whisper on. 'auto' prefers CUDA, then MPS, otherwise CPU.",
)
flags.DEFINE_boolean(
    "require_gpu",
    False,
    "Exit immediately if CUDA is unavailable instead of falling back to CPU.",
)

flags.DEFINE_string(
    "language",
    "en",
    "Preferred language code for transcription (e.g., en, es). Use 'auto' to let the model detect.",
)

# Partial transcription configuration (Whisper backend only)
flags.DEFINE_boolean(
    "enable_partial_transcription",
    False,
    "Enable real-time partial transcription of accumulating audio chunks using a fast model. "
    "Only applies to Whisper backend.",
)
flags.DEFINE_string(
    "partial_model",
    "base.en",
    "Whisper model for partial transcription (should be faster than main model, e.g., base.en, tiny.en).",
)
flags.DEFINE_float(
    "partial_interval",
    1.0,
    "Interval (seconds) between partial transcription updates.",
    lower_bound=0.1,
    upper_bound=10.0,
)
flags.DEFINE_float(
    "max_partial_buffer_seconds",
    10.0,
    "Segment duration (in seconds) for partial transcription. "
    "Audio is divided into fixed-duration segments. Each segment is continuously "
    "transcribed as new audio accumulates, with updates printed on separate lines.",
    lower_bound=1.0,
    upper_bound=60.0,
)

# Audio configuration
flags.DEFINE_integer(
    "samplerate",
    16000,
    "Input sample rate expected by the model.",
)
flags.DEFINE_integer(
    "channels",
    1,
    "Number of microphone input channels.",
)
flags.DEFINE_string(
    "audio_file",
    None,
    "Path or URL to audio file for simulating live transcription (MP3, WAV, FLAC, etc.). "
    "Supports local files and HTTP/HTTPS URLs. "
    "If provided, audio will be read from file/URL instead of microphone.",
)
flags.DEFINE_float(
    "playback_speed",
    1.0,
    "Playback speed multiplier when using --audio-file (1.0 = real-time, 2.0 = 2x speed).",
    lower_bound=0.1,
    upper_bound=10.0,
)

# File configuration
flags.DEFINE_string(
    "temp_file",
    None,
    "Optional path to persist audio chunks for inspection.",
)

# SSL/Certificate configuration
flags.DEFINE_string(
    "ca_cert",
    None,
    "Custom certificate bundle to trust when downloading Whisper models.",
)
flags.DEFINE_boolean(
    "disable_ssl_verify",
    False,
    "Disable SSL certificate verification for all network operations, including model downloads "
    "and Realtime API connections. Use this to bypass certificate issues in restricted networks. "
    "WARNING: This is insecure and not recommended for production use.",
)

# VAD configuration
flags.DEFINE_integer(
    "vad_aggressiveness",
    2,
    "WebRTC VAD aggressiveness level: 0=least aggressive, 3=most aggressive.",
    lower_bound=0,
    upper_bound=3,
)
flags.DEFINE_float(
    "vad_min_silence_duration",
    0.2,
    "Minimum duration of silence (seconds) to trigger chunk split.",
)
flags.DEFINE_float(
    "vad_min_speech_duration",
    0.25,
    "Minimum duration of speech (seconds) required before transcribing.",
)
flags.DEFINE_float(
    "vad_speech_pad_duration",
    0.2,
    "Padding duration (seconds) added before speech to avoid cutting words.",
)
flags.DEFINE_float(
    "max_chunk_duration",
    60.0,
    "Maximum chunk duration in seconds when using VAD.",
)

# Feature flags
flags.DEFINE_boolean(
    "refine_with_context",
    False,
    "[NOT YET IMPLEMENTED] Use 3-chunk sliding window to refine middle chunk transcription.",
)

# Realtime API configuration
flags.DEFINE_string(
    "realtime_model",
    "gpt-realtime-mini",
    "Realtime model to use with the OpenAI Realtime API.",
)
flags.DEFINE_string(
    "realtime_endpoint",
    "wss://api.openai.com/v1/realtime",
    "Realtime websocket endpoint (advanced).",
)
flags.DEFINE_string(
    "realtime_instructions",
    (
        "You are a high-accuracy transcription service. "
        "Return a concise verbatim transcript of the most recent audio buffer. "
        "Do not add commentary or speaker labels."
    ),
    "Instruction prompt sent to the realtime model.",
)
flags.DEFINE_float(
    "realtime_vad_threshold",
    0.2,
    "Server VAD threshold for turn detection (0.0-1.0). Lower = more sensitive. "
    "Default 0.2 works well for continuous speech (news, podcasts). "
    "Only applies when using realtime backend.",
    lower_bound=0.0,
    upper_bound=1.0,
)
flags.DEFINE_integer(
    "realtime_vad_silence_duration_ms",
    100,
    "Silence duration in milliseconds required to detect turn boundary. "
    "Lower values = more frequent chunks. Default 100ms works well for fast-paced content. "
    "Only applies when using realtime backend.",
    lower_bound=100,
    upper_bound=2000,
)
flags.DEFINE_boolean(
    "realtime_debug",
    False,
    "Enable debug logging for realtime transcription events (shows delta and completed events).",
)

# Comparison and capture configuration
flags.DEFINE_boolean(
    "compare_transcripts",
    True,
    "Compare chunked transcription with full-audio transcription at session end. "
    "Note: For Realtime API, this doubles API usage cost.",
)
flags.DEFINE_float(
    "max_capture_duration",
    120.0,
    "Maximum duration (seconds) to run the transcription session. "
    "Program will gracefully stop after this duration. Set to 0 for unlimited duration.",
)

# Session logging configuration (always enabled)
flags.DEFINE_string(
    "session_log_dir",
    "./session_logs",
    "Directory to save session logs. All sessions are logged with full audio, chunk audio, and metadata.",
)
flags.DEFINE_float(
    "min_log_duration",
    10.0,
    "Minimum session duration (seconds) required to save logs. Sessions shorter than this are discarded.",
)
flags.DEFINE_enum(
    "audio_format",
    "flac",
    ["wav", "flac"],
    "Audio format for saved session files. 'flac' provides lossless compression (~50-60% smaller), 'wav' is uncompressed.",
)

# Validators
flags.register_validator(
    "max_capture_duration",
    lambda value: value >= 0.0,
    message="--max_capture_duration must be >= 0 (set to 0 for unlimited duration)",
)


class ChunkCollectorWithStitching:
    """
    Collects transcription chunks and displays them.
    With VAD-based chunking, chunks are simply stitched (no overlap).
    """

    def __init__(self, stream: TextIO) -> None:
        self._stream = stream
        self._last_time = float("-inf")
        self._chunks: list[TranscriptionChunk] = []
        self._last_partial_chunk_index: int | None = None

    @staticmethod
    def _clean_chunk_text(text: str, is_final_chunk: bool = False) -> str:
        """
        Clean trailing punctuation from chunk text for better stitching.

        Whisper adds punctuation as if each chunk is a complete sentence,
        but VAD splits can occur mid-sentence. We strip trailing commas and
        periods (but keep ? and ! as they're more intentional).
        """
        text = text.strip()
        if not is_final_chunk and text:
            # Strip trailing commas and periods, but not question marks or exclamation points
            while text and text[-1] in ".,":
                text = text[:-1].rstrip()
        return text

    def _display_partial_chunk(self, chunk: TranscriptionChunk) -> None:
        """Display a partial transcription, overwriting same segment on TTY."""
        import sys

        # Check if stdout is a TTY for color formatting
        is_tty = sys.stdout.isatty()

        # Format the partial transcription
        if chunk.inference_seconds is not None:
            timing_suffix = f" | t={chunk.end_time:.2f}s | inference: {chunk.inference_seconds:.2f}s"
        else:
            timing_suffix = f" | t={chunk.end_time:.2f}s"

        # Display full accumulated text (no truncation)
        text_display = chunk.text.strip()

        if is_tty:
            yellow = "\x1b[33m"
            reset = "\x1b[0m"
            bold = "\x1b[1m"
            dim = "\x1b[2m"
            label = f"{bold}{yellow}[PARTIAL {chunk.index:03d}{timing_suffix}]{reset}"
            line = f"{label} {dim}{text_display}{reset}"

            # Handle line clearing/newlines based on segment changes
            if self._last_partial_chunk_index == chunk.index:
                # Same segment: overwrite with \r and clear line
                self._stream.write(f"\r\x1b[2K{line}")
            else:
                # Different segment: finalize previous and start new line
                if self._last_partial_chunk_index is not None:
                    self._stream.write("\n")
                self._stream.write(line)
        else:
            # Non-TTY: always print on new line for logs
            label = f"[PARTIAL {chunk.index:03d}{timing_suffix}]"
            line = f"{label} {text_display}"
            self._stream.write(line + "\n")

        # Update tracking
        self._last_partial_chunk_index = chunk.index
        self._stream.flush()

    def __call__(self, chunk: TranscriptionChunk) -> None:
        """Process a transcription chunk (implements ChunkConsumer protocol)."""
        if not chunk.text:
            return

        # Handle partial transcription (don't store, just display)
        if chunk.is_partial:
            self._display_partial_chunk(chunk)
            return

        # Store the chunk
        self._chunks.append(chunk)

        # TODO: Sliding window refinement feature (--refine-with-context)
        # When enabled, use a 3-chunk sliding window to refine the middle chunk:
        # 1. Store raw audio buffers for the last 3 chunks
        # 2. When chunk N arrives, stitch audio from chunks N-2, N-1, N
        # 3. Re-transcribe the stitched audio with Whisper
        # 4. Use word-level timestamps to extract refined text for chunk N-1 (middle)
        # 5. Display refined version of chunk N-1 after chunk N processing
        #
        # Benefits:
        # - Better context reduces boundary errors
        # - Improved accuracy for cross-chunk phrases
        # - More natural linguistic flow
        #
        # Considerations:
        # - Adds 1-chunk latency (chunk N-1 displayed after N arrives)
        # - Requires ~3x inference time per chunk
        # - Needs word timestamps (not available on MPS/Apple Metal)
        # - Requires storing raw audio buffers
        #
        # Implementation notes:
        # - Add audio buffer storage to whisper_backend.py
        # - Pass raw audio along with transcription in chunk_consumer
        # - Only refine chunks >= 2 (need 3-chunk window)
        # - Extract middle chunk text using word timestamps that fall within N-1 time range
        # - Display both immediate (chunk N) and refined (chunk N-1) with different labels

        # Finalize any partial line before displaying final chunk
        if self._last_partial_chunk_index is not None:
            self._stream.write("\n")
            self._last_partial_chunk_index = None

        # Display the individual chunk
        if chunk.inference_seconds is not None:
            # Whisper mode: show actual audio duration and inference time
            chunk_audio_duration = chunk.end_time - chunk.start_time
            timing_suffix = (
                f" | t={chunk.end_time:.2f}s | audio: {chunk_audio_duration:.2f}s | inference: {chunk.inference_seconds:.2f}s"
            )
            label = f"[chunk {chunk.index:03d}{timing_suffix}]"
        else:
            # Realtime mode: show absolute timestamp from session start
            timing_suffix = f" | t={chunk.end_time:.2f}s"
            label = f"[chunk {chunk.index:03d}{timing_suffix}]"
        use_color = bool(getattr(self._stream, "isatty", lambda: False)())

        cyan = ""
        green = ""
        reset = ""
        bold = ""
        if use_color:
            cyan = "\x1b[36m"
            green = "\x1b[32m"
            reset = "\x1b[0m"
            bold = "\x1b[1m"
            label_colored = f"{bold}{cyan}{label}{reset}"
            line = f"{label_colored} {chunk.text.strip()}"
        else:
            line = f"{label} {chunk.text.strip()}"

        self._stream.write(line + "\n")
        self._stream.flush()
        self._last_time = max(self._last_time, chunk.end_time)

        # Show stitched result every few chunks
        if (chunk.index + 1) % 3 == 0:
            # Clean trailing punctuation from all chunks except the last one
            cleaned_chunks = [
                self._clean_chunk_text(c.text, is_final_chunk=(i == len(self._chunks) - 1))
                for i, c in enumerate(self._chunks)
            ]
            stitched_text = " ".join(chunk for chunk in cleaned_chunks if chunk)

            if use_color:
                stitched_label = f"\n{bold}{green}[STITCHED]{reset}"
            else:
                stitched_label = "\n[STITCHED]"
            self._stream.write(f"{stitched_label} {stitched_text}\n\n")
            self._stream.flush()

    def get_final_stitched(self) -> str:
        """Get the final stitched transcription of all chunks."""
        # Clean trailing punctuation from all chunks except the last one
        cleaned_chunks = [
            self._clean_chunk_text(c.text, is_final_chunk=(i == len(self._chunks) - 1))
            for i, c in enumerate(self._chunks)
        ]
        return " ".join(chunk for chunk in cleaned_chunks if chunk)

    def get_cleaned_chunks(self) -> list[tuple[int, str]]:
        """
        Get cleaned text for each chunk.

        Returns:
            List of (chunk_index, cleaned_text) tuples
        """
        cleaned_chunks = [
            (
                c.index,
                self._clean_chunk_text(c.text, is_final_chunk=(i == len(self._chunks) - 1)),
            )
            for i, c in enumerate(self._chunks)
        ]
        return cleaned_chunks


def _normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def _print_final_stitched(stream: TextIO, text: str) -> None:
    """Print final stitched transcription with appropriate formatting."""
    if not text:
        return

    use_color = getattr(stream, "isatty", lambda: False)()
    if use_color:
        green = "\x1b[32m"
        reset = "\x1b[0m"
        bold = "\x1b[1m"
        print(f"\n{bold}{green}[FINAL STITCHED]{reset} {text}\n", file=stream)
    else:
        print(f"\n[FINAL STITCHED] {text}\n", file=stream)


def compute_transcription_diff(stitched_text: str, complete_text: str) -> tuple[float, list[dict[str, str]]]:
    """
    Compute diff between stitched and complete transcriptions.

    Returns:
        Tuple of (similarity_ratio, diff_snippets)
    """
    if not (stitched_text.strip() and complete_text.strip()):
        return (0.0, [])

    stitched_tokens_norm = [norm for _, norm in _tokenize_with_original(stitched_text)]
    complete_tokens_norm = [norm for _, norm in _tokenize_with_original(complete_text)]
    similarity = difflib.SequenceMatcher(None, stitched_tokens_norm, complete_tokens_norm).ratio()
    diff_snippets = _generate_diff_snippets(stitched_text, complete_text, use_color=False)

    return (similarity, diff_snippets)


def print_transcription_summary(
    stream: TextIO,
    final_text: str,
    complete_audio_text: str,
) -> None:
    """Print stitched transcript, complete-audio transcript, and comparison details."""
    use_color = bool(getattr(stream, "isatty", lambda: False)())
    final_clean = final_text.strip()
    complete_audio_clean = complete_audio_text.strip()

    bold = ""
    green = ""
    reset = ""
    if use_color:
        green = "\x1b[32m"
        reset = "\x1b[0m"
        bold = "\x1b[1m"

    if final_clean:
        if use_color:
            print(f"\n{bold}{green}[FINAL STITCHED]{reset} {final_clean}\n", file=stream)
        else:
            print(f"\n[FINAL STITCHED] {final_clean}\n", file=stream)

    if complete_audio_clean:
        if use_color:
            print(
                f"{bold}{green}[COMPLETE AUDIO]{reset} {complete_audio_clean}\n",
                file=stream,
            )
        else:
            print(f"[COMPLETE AUDIO] {complete_audio_clean}\n", file=stream)

    if not (final_clean and complete_audio_clean):
        return

    stitched_tokens_norm = [norm for _, norm in _tokenize_with_original(final_clean)]
    complete_tokens_norm = [norm for _, norm in _tokenize_with_original(complete_audio_clean)]
    stitched_normalized = " ".join(stitched_tokens_norm)
    complete_normalized = " ".join(complete_tokens_norm)
    comparison_label = f"{bold}{green}[COMPARISON]{reset}" if use_color else "[COMPARISON]"

    if stitched_normalized == complete_normalized:
        print(
            f"{comparison_label} Stitched transcription matches complete audio transcription.\n",
            file=stream,
        )
        return

    similarity = difflib.SequenceMatcher(None, stitched_tokens_norm, complete_tokens_norm).ratio()
    print(
        f"{comparison_label} Difference detected (similarity {similarity:.2%}).",
        file=stream,
    )
    diff_label = "\x1b[2;36m[DIFF]\x1b[0m" if use_color else "[DIFF]"
    diff_snippets = _generate_diff_snippets(final_clean, complete_audio_clean, use_color)
    for snippet in diff_snippets:
        print(
            f"{diff_label} {snippet['tag']}:\n    stitched: {snippet['stitched']}\n    complete: {snippet['complete']}",
            file=stream,
        )


def _tokenize_with_original(text: str) -> list[tuple[str, str]]:
    """Return (raw, normalized) tokens where normalized strips punctuation and lowercases."""
    tokens: list[tuple[str, str]] = []
    for raw in text.split():
        normalized = re.sub(r"[^\w']+", "", raw).lower()
        if not normalized:
            continue
        tokens.append((raw, normalized))
    return tokens


def _colorize_token(token: str, use_color: bool, color_code: str) -> str:
    if use_color:
        return f"\x1b[2;{color_code}m{token}\x1b[0m"
    return f"[[{token}]]"


def _format_diff_snippet(
    tokens: list[tuple[str, str]],
    diff_start: int,
    diff_end: int,
    use_color: bool,
    color_code: str,
) -> str:
    if not tokens:
        return _colorize_token("∅", use_color, color_code)

    context = 3
    window_end = max(diff_end, diff_start)
    start = max(diff_start - context, 0)
    end = min(window_end + context, len(tokens))
    parts: list[str] = []

    for idx in range(start, end):
        raw = tokens[idx][0]
        if diff_start <= idx < diff_end:
            parts.append(_colorize_token(raw, use_color, color_code))
        else:
            parts.append(raw)

    if diff_start == diff_end:
        placeholder = _colorize_token("∅", use_color, color_code)
        insert_at = diff_start - start
        if insert_at < 0:
            parts.insert(0, placeholder)
        elif insert_at >= len(parts):
            parts.append(placeholder)
        else:
            parts.insert(insert_at, placeholder)

    snippet = " ".join(parts).strip()
    if start > 0:
        snippet = "... " + snippet
    if end < len(tokens):
        snippet = snippet + " ..."
    return snippet or _colorize_token("∅", use_color, color_code)


def _generate_diff_snippets(
    stitched_text: str,
    complete_text: str,
    use_color: bool,
) -> list[dict[str, str]]:
    stitched_tokens = _tokenize_with_original(stitched_text)
    complete_tokens = _tokenize_with_original(complete_text)
    stitched_norm = [norm for _, norm in stitched_tokens]
    complete_norm = [norm for _, norm in complete_tokens]

    matcher = difflib.SequenceMatcher(None, stitched_norm, complete_norm)
    snippets: list[dict[str, str]] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        snippets.append(
            {
                "tag": tag,
                "stitched": _format_diff_snippet(stitched_tokens, i1, i2, use_color, "33"),
                "complete": _format_diff_snippet(complete_tokens, j1, j2, use_color, "36"),
            }
        )

    return snippets


def _finalize_transcription_session(
    collector: ChunkCollectorWithStitching,
    result: WhisperTranscriptionResult | RealtimeTranscriptionResult | None,
    session_logger: SessionLogger,
    compare_transcripts: bool,
    min_log_duration: float,
    full_audio_transcription: str | None = None,
) -> None:
    """
    Finalize transcription session with common cleanup logic.

    This function consolidates the finalization logic shared between Whisper
    and Realtime backends, eliminating code duplication.

    Args:
        collector: Chunk collector with stitched results
        result: Transcription result from backend
        session_logger: Session logger for persistence
        compare_transcripts: Whether to compare and show diffs
        min_log_duration: Minimum duration for logging
        full_audio_transcription: Optional full audio transcription (for Realtime)
    """
    # Get final stitched result
    final = collector.get_final_stitched()

    # Update session logger with cleaned chunk text
    for chunk_index, cleaned_text in collector.get_cleaned_chunks():
        session_logger.update_chunk_cleaned_text(chunk_index, cleaned_text)

    # Compute diff if comparison is enabled
    similarity = None
    diff_snippets = None
    comparison_text = full_audio_transcription

    if result is not None:
        # For Whisper, use result.full_audio_transcription
        # For Realtime, use the passed full_audio_transcription
        if isinstance(result, WhisperTranscriptionResult):
            comparison_text = result.full_audio_transcription
        elif hasattr(result, "full_audio_transcription") and result.full_audio_transcription:
            # Support duck-typed results (e.g., test mocks)
            comparison_text = result.full_audio_transcription

        if comparison_text:
            similarity, diff_snippets = compute_transcription_diff(final, comparison_text)

    # Finalize session logging
    if result is not None:
        session_logger.finalize(
            capture_duration=result.capture_duration,
            full_audio_transcription=comparison_text,
            stitched_transcription=final,
            extra_metadata=result.metadata,
            min_duration=min_log_duration,
            transcription_similarity=similarity,
            transcription_diffs=diff_snippets,
        )

    # Print results
    if compare_transcripts:
        complete_audio_text = comparison_text or ""
        print_transcription_summary(sys.stdout, final, complete_audio_text)
    else:
        _print_final_stitched(sys.stdout, final)

    # Print captured duration
    if result is not None:
        print(f"Total captured audio duration: {result.capture_duration:.2f} seconds", file=sys.stderr)


def transcribe(
    backend: WhisperBackend | RealtimeBackend,
    audio_source: AudioSource,
) -> Generator[TranscriptionChunk, None, WhisperTranscriptionResult | RealtimeTranscriptionResult]:
    """
    Generator that yields transcription chunks from the specified backend.

    This function orchestrates the transcription process by:
    1. Running the backend in a background thread
    2. Yielding chunks as they are produced
    3. Returning the final transcription result

    Args:
        backend: Configured backend instance (WhisperBackend or RealtimeBackend)
        audio_source: Configured audio source (FileAudioSource or AudioCaptureManager)

    Yields:
        TranscriptionChunk: Individual transcription chunks with metadata

    Returns:
        Final transcription result (WhisperTranscriptionResult or RealtimeTranscriptionResult)

    Example:
        backend = WhisperBackend(model_name="turbo", language="en")
        audio_source = FileAudioSource("audio.mp3", sample_rate=16000)

        for chunk in transcribe(backend, audio_source):
            print(chunk.text)
    """
    chunk_queue: queue.Queue[TranscriptionChunk | None] = queue.Queue()
    result_container: list[WhisperTranscriptionResult | RealtimeTranscriptionResult] = []
    error_container: list[BaseException] = []

    def backend_worker() -> None:
        """Worker thread that runs the backend and puts chunks in the queue."""
        try:
            result = backend.run(audio_source, chunk_queue)
            result_container.append(result)
        except BaseException as e:
            error_container.append(e)
        finally:
            # Always send sentinel to unblock the generator
            chunk_queue.put(None)

    # Start backend worker thread
    worker_thread = threading.Thread(target=backend_worker, daemon=True)
    worker_thread.start()

    # Yield chunks as they arrive
    try:
        while True:
            chunk = chunk_queue.get()
            if chunk is None:  # Sentinel value
                break
            yield chunk
    finally:
        # Wait for worker to complete
        worker_thread.join()

    # Check for errors
    if error_container:
        raise error_container[0]

    # Return final result
    if not result_container:
        raise RuntimeError("Backend worker completed without producing a result")
    return result_container[0]


def main(argv: list[str]) -> None:
    # Check for unimplemented features
    if FLAGS.refine_with_context:
        print(
            "ERROR: --refine-with-context is not yet implemented.\n"
            "This feature will use a 3-chunk sliding window to refine transcriptions with more context.\n"
            "See TODO comments in main.py for implementation details.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Warn about memory usage with unlimited duration and comparison enabled
    if FLAGS.compare_transcripts and FLAGS.max_capture_duration == 0:
        print(
            "WARNING: Running with unlimited duration and comparison enabled will "
            "continuously accumulate audio in memory.\n"
            "Consider setting --max_capture_duration or use --nocompare_transcripts to reduce memory usage.\n",
            file=sys.stderr,
        )

    # Confirm long capture durations with comparison enabled
    if FLAGS.compare_transcripts and FLAGS.max_capture_duration > 300:  # > 5 minutes
        duration_minutes = FLAGS.max_capture_duration / 60.0
        print(
            f"You have set a capture duration of {duration_minutes:.1f} minutes with comparison enabled.\n"
            f"This will keep audio in memory for the entire session.",
            file=sys.stderr,
        )
        if FLAGS.backend == "realtime":
            print(
                "Note: For Realtime API, this will also double your API usage cost.\n",
                file=sys.stderr,
            )

        # Only prompt if stdin is available
        try:
            response = input("Continue? [y/N]: ").strip().lower()
            if response not in ("y", "yes"):
                print("Cancelled.", file=sys.stderr)
                sys.exit(0)
        except (EOFError, OSError):
            # stdin not available (e.g., running in background), proceed without confirmation
            print(
                "(Proceeding without confirmation - stdin not available)",
                file=sys.stderr,
            )

    language_pref = (FLAGS.language or "").strip()

    # Create session logger (always enabled)
    log_dir = Path(FLAGS.session_log_dir)
    session_logger = SessionLogger(
        output_dir=log_dir,
        sample_rate=FLAGS.samplerate,
        channels=FLAGS.channels,
        backend=FLAGS.backend,
        save_chunk_audio=True,  # Always save everything
        audio_format=FLAGS.audio_format,
    )

    # Create chunk collector
    collector = ChunkCollectorWithStitching(sys.stdout)

    # Create audio source
    if FLAGS.audio_file:
        audio_source: AudioSource = FileAudioSource(
            audio_file=FLAGS.audio_file,
            sample_rate=FLAGS.samplerate,
            channels=FLAGS.channels,
            max_capture_duration=FLAGS.max_capture_duration,
            collect_full_audio=FLAGS.compare_transcripts or (session_logger is not None),
            playback_speed=FLAGS.playback_speed,
        )
    else:
        audio_source = AudioCaptureManager(
            sample_rate=FLAGS.samplerate,
            channels=FLAGS.channels,
            max_capture_duration=FLAGS.max_capture_duration,
            collect_full_audio=FLAGS.compare_transcripts or (session_logger is not None),
        )

    # Create backend
    if FLAGS.backend == "whisper":
        backend: WhisperBackend | RealtimeBackend = WhisperBackend(
            model_name=FLAGS.model or "turbo",
            device_preference=FLAGS.device or "auto",
            require_gpu=FLAGS.require_gpu,
            vad_aggressiveness=FLAGS.vad_aggressiveness,
            vad_min_silence_duration=FLAGS.vad_min_silence_duration,
            vad_min_speech_duration=FLAGS.vad_min_speech_duration,
            vad_speech_pad_duration=FLAGS.vad_speech_pad_duration,
            max_chunk_duration=FLAGS.max_chunk_duration,
            enable_partial_transcription=FLAGS.enable_partial_transcription,
            partial_model=FLAGS.partial_model,
            partial_interval=FLAGS.partial_interval,
            max_partial_buffer_seconds=FLAGS.max_partial_buffer_seconds,
            language=language_pref,
            compare_transcripts=FLAGS.compare_transcripts,
            session_logger=session_logger,
            min_log_duration=FLAGS.min_log_duration,
            ca_cert=Path(FLAGS.ca_cert) if FLAGS.ca_cert else None,
            disable_ssl_verify=FLAGS.disable_ssl_verify,
            temp_file=Path(FLAGS.temp_file) if FLAGS.temp_file else None,
        )
    else:  # realtime
        api_key = FLAGS.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OpenAI API key required for realtime transcription. Provide --api-key or set OPENAI_API_KEY."
            )
        backend = RealtimeBackend(
            api_key=api_key,
            endpoint=FLAGS.realtime_endpoint,
            model=FLAGS.realtime_model,
            instructions=FLAGS.realtime_instructions,
            vad_threshold=FLAGS.realtime_vad_threshold,
            vad_silence_duration_ms=FLAGS.realtime_vad_silence_duration_ms,
            debug=FLAGS.realtime_debug,
            language=language_pref,
            compare_transcripts=FLAGS.compare_transcripts,
            session_logger=session_logger,
            min_log_duration=FLAGS.min_log_duration,
            disable_ssl_verify=FLAGS.disable_ssl_verify,
        )

    # Run transcription using the generator
    result: WhisperTranscriptionResult | RealtimeTranscriptionResult | None = None
    full_audio_transcription: str | None = None
    try:
        # Create transcription generator
        transcription_gen = transcribe(backend, audio_source)

        # Consume chunks and collect result
        for chunk in transcription_gen:
            collector(chunk)

        # Get the final result from the generator's return value
        try:
            result = transcription_gen.send(None)
        except StopIteration as e:
            result = e.value

    except KeyboardInterrupt:
        pass
    finally:
        # Get full audio transcription for comparison if enabled (Realtime-specific)
        if (
            FLAGS.backend == "realtime"
            and result is not None
            and isinstance(result, RealtimeTranscriptionResult)
            and FLAGS.compare_transcripts
            and result.full_audio.size > 0
        ):
            try:
                full_audio_transcription = transcribe_full_audio_realtime(
                    result.full_audio,
                    sample_rate=result.sample_rate,
                    chunk_duration=REALTIME_CHUNK_DURATION,
                    api_key=api_key or "",
                    endpoint=FLAGS.realtime_endpoint,
                    model=FLAGS.realtime_model,
                    instructions=FLAGS.realtime_instructions,
                    disable_ssl_verify=FLAGS.disable_ssl_verify,
                    language=language_pref,
                )
            except Exception as exc:
                print(
                    f"WARNING: Unable to transcribe full audio for comparison: {exc}",
                    file=sys.stderr,
                )

        # Use common finalization logic
        _finalize_transcription_session(
            collector=collector,
            result=result,
            session_logger=session_logger,
            compare_transcripts=FLAGS.compare_transcripts,
            min_log_duration=FLAGS.min_log_duration,
            full_audio_transcription=full_audio_transcription,
        )


def cli_main() -> None:
    """Entry point for the CLI (called by pyproject.toml console_scripts)."""
    app.run(main)


if __name__ == "__main__":
    app.run(main)
