"""Chunk collector for displaying and stitching transcription chunks."""

from __future__ import annotations

from typing import TextIO

from transcribe_demo.backend_protocol import TranscriptionChunk


class ChunkCollector:
    """
    Collects transcription chunks and displays them with intelligent stitching.

    For VAD-based chunking (Whisper), chunks are stitched together with
    punctuation cleanup to create natural-looking transcripts despite
    mid-sentence splits.
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
            print(
                f"[chunk {chunk.index:03d} | t={chunk.end_time:.2f}s | "
                f"audio: {chunk_audio_duration:.2f}s | inference: {chunk.inference_seconds:.2f}s] {chunk.text}",
                file=self._stream,
                flush=True,
            )
        else:
            # Realtime mode: just show timestamp
            print(
                f"[chunk {chunk.index:03d} | {chunk.end_time:.2f}s] {chunk.text}",
                file=self._stream,
                flush=True,
            )

        self._last_time = chunk.end_time

    def get_final_stitched_text(self) -> str:
        """
        Get the final stitched transcription from all chunks.

        Returns cleaned and joined text with smart punctuation handling.
        """
        if not self._chunks:
            return ""

        # Clean and stitch chunks
        cleaned_chunks = []
        for i, chunk in enumerate(self._chunks):
            is_final = i == len(self._chunks) - 1
            cleaned = self._clean_chunk_text(chunk.text, is_final_chunk=is_final)
            if cleaned:
                cleaned_chunks.append(cleaned)

        return " ".join(cleaned_chunks)

    def get_chunks(self) -> list[TranscriptionChunk]:
        """Get all collected chunks."""
        return self._chunks.copy()
