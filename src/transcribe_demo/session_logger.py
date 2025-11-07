"""Session logging module for persisting audio and transcriptions to disk."""

from __future__ import annotations

import json
import shutil
import sys
import wave
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class ChunkMetadata:
    """Metadata for a single transcription chunk."""

    index: int
    text: str
    start_time: float
    end_time: float
    duration: float
    inference_seconds: float | None = None
    audio_filename: str | None = None
    cleaned_text: str | None = None  # Text after stitching punctuation cleanup


@dataclass
class SessionMetadata:
    """Metadata for an entire transcription session."""

    session_id: str
    timestamp: str
    backend: str
    sample_rate: int
    channels: int
    capture_duration: float
    total_chunks: int

    # Backend-specific parameters
    model: str | None = None
    device: str | None = None
    language: str | None = None
    vad_aggressiveness: int | None = None
    vad_min_silence_duration: float | None = None
    vad_min_speech_duration: float | None = None
    vad_speech_pad_duration: float | None = None
    max_chunk_duration: float | None = None

    # Realtime-specific parameters
    realtime_endpoint: str | None = None
    realtime_instructions: str | None = None

    # Complete audio transcription
    full_audio_transcription: str | None = None
    stitched_transcription: str | None = None

    # Diff information
    transcription_similarity: float | None = None  # Similarity ratio between stitched and complete
    transcription_diffs: list[dict[str, str]] | None = None  # Detailed diff snippets


class SessionLogger:
    """
    Logs audio and transcriptions to disk with metadata.

    Directory structure:
        {output_dir}/YYYY-MM-DD/session_HHMMSS_backend/
            ├── session.json           # Session metadata and chunk info
            ├── full_audio.wav         # Complete raw audio (mono, 16kHz)
            ├── chunks/                # Individual chunk audio files (optional)
            │   ├── chunk_000.wav
            │   ├── chunk_001.wav
            │   └── ...
            └── README.txt             # Human-readable description
    """

    def __init__(
        self,
        output_dir: Path,
        sample_rate: int,
        channels: int,
        backend: str,
        save_chunk_audio: bool = False,
        session_id: str | None = None,
    ):
        """
        Initialize the session logger.

        Args:
            output_dir: Base directory for saving sessions
            sample_rate: Audio sample rate in Hz
            channels: Number of audio channels
            backend: Backend name ("whisper" or "realtime")
            save_chunk_audio: Whether to save individual chunk audio files
            session_id: Optional session ID (auto-generated if None)
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.backend = backend
        self.save_chunk_audio = save_chunk_audio

        # Generate session ID and create directory
        now = datetime.now()
        if session_id is None:
            date_str = now.strftime("%Y-%m-%d")
            time_str = now.strftime("%H%M%S")
            session_id = f"session_{time_str}_{backend}"
        else:
            # If custom session_id provided, still organize by current date
            date_str = now.strftime("%Y-%m-%d")

        self.session_id = session_id
        # Organize sessions by date: output_dir/YYYY-MM-DD/session_HHMMSS_backend/
        date_dir = output_dir / date_str
        self.session_dir = date_dir / session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)

        if save_chunk_audio:
            self.chunks_dir = self.session_dir / "chunks"
            self.chunks_dir.mkdir(exist_ok=True)
        else:
            self.chunks_dir = None

        # Storage for chunks and metadata
        self.chunks: list[ChunkMetadata] = []
        self.session_metadata: SessionMetadata | None = None

        print(f"Session logging to: {self.session_dir}", file=sys.stderr)

    def log_chunk(
        self,
        index: int,
        text: str,
        start_time: float,
        end_time: float,
        inference_seconds: float | None = None,
        audio: np.ndarray | None = None,
        cleaned_text: str | None = None,
    ) -> None:
        """
        Log a transcription chunk.

        Args:
            index: Chunk index
            text: Transcribed text (original, before stitching cleanup)
            start_time: Chunk start time (seconds)
            end_time: Chunk end time (seconds)
            inference_seconds: Inference duration (None for realtime)
            audio: Optional audio samples for this chunk
            cleaned_text: Optional cleaned text after stitching punctuation cleanup
        """
        duration = end_time - start_time
        audio_filename = None

        # Save chunk audio if requested and provided
        if self.save_chunk_audio and audio is not None and self.chunks_dir is not None:
            audio_filename = f"chunk_{index:03d}.wav"
            audio_path = self.chunks_dir / audio_filename
            self._save_audio_wav(audio, audio_path)

        # Store chunk metadata
        chunk_meta = ChunkMetadata(
            index=index,
            text=text,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            inference_seconds=inference_seconds,
            audio_filename=audio_filename,
            cleaned_text=cleaned_text,
        )
        self.chunks.append(chunk_meta)

    def update_chunk_cleaned_text(self, index: int, cleaned_text: str) -> None:
        """
        Update the cleaned text for a specific chunk.

        Args:
            index: Chunk index
            cleaned_text: Cleaned text after stitching punctuation cleanup
        """
        for chunk in self.chunks:
            if chunk.index == index:
                chunk.cleaned_text = cleaned_text
                return

    def save_full_audio(self, audio: np.ndarray, capture_duration: float) -> None:
        """
        Save the complete raw audio to disk.

        Args:
            audio: Complete mono audio array (float32)
            capture_duration: Total capture duration in seconds
        """
        if audio.size == 0:
            print("WARNING: No audio to save", file=sys.stderr)
            return

        full_audio_path = self.session_dir / "full_audio.wav"
        self._save_audio_wav(audio, full_audio_path)
        print(f"Saved full audio: {full_audio_path}", file=sys.stderr)

    def finalize(
        self,
        capture_duration: float,
        full_audio_transcription: str | None = None,
        stitched_transcription: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
        min_duration: float = 0.0,
        transcription_similarity: float | None = None,
        transcription_diffs: list[dict[str, str]] | None = None,
    ) -> None:
        """
        Finalize the session and write metadata to disk.

        Args:
            capture_duration: Total capture duration in seconds
            full_audio_transcription: Optional full audio transcription
            stitched_transcription: Optional stitched chunk transcription
            extra_metadata: Optional additional metadata (model, device, VAD params, etc.)
            min_duration: Minimum duration (seconds) required to save logs (default: 0.0 = always save)
            transcription_similarity: Optional similarity ratio between stitched and complete
            transcription_diffs: Optional detailed diff snippets
        """
        # Check if session meets minimum duration requirement
        if capture_duration < min_duration:
            print(
                f"Session duration ({capture_duration:.1f}s) is below minimum ({min_duration:.1f}s). "
                f"Discarding session logs.",
                file=sys.stderr,
            )
            # Clean up the session directory
            if self.session_dir.exists():
                shutil.rmtree(self.session_dir)
            return

        # Build session metadata
        timestamp = datetime.now().isoformat()
        metadata = SessionMetadata(
            session_id=self.session_id,
            timestamp=timestamp,
            backend=self.backend,
            sample_rate=self.sample_rate,
            channels=self.channels,
            capture_duration=capture_duration,
            total_chunks=len(self.chunks),
            full_audio_transcription=full_audio_transcription,
            stitched_transcription=stitched_transcription,
            transcription_similarity=transcription_similarity,
            transcription_diffs=transcription_diffs,
        )

        # Merge extra metadata
        if extra_metadata:
            for key, value in extra_metadata.items():
                if hasattr(metadata, key):
                    setattr(metadata, key, value)

        self.session_metadata = metadata

        # Save session.json
        session_data = {
            "metadata": asdict(metadata),
            "chunks": [asdict(chunk) for chunk in self.chunks],
        }
        session_json_path = self.session_dir / "session.json"
        with open(session_json_path, "w", encoding="utf-8") as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)
        print(f"Saved session metadata: {session_json_path}", file=sys.stderr)

        # Create human-readable README
        self._create_readme()

    def _save_audio_wav(self, audio: np.ndarray, path: Path) -> None:
        """
        Save audio as WAV file (mono, int16 PCM).

        Args:
            audio: Mono audio array (float32, [-1.0, 1.0])
            path: Output WAV file path
        """
        # Ensure audio is mono and float32
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # Clip to valid range and convert to int16
        audio = np.clip(audio, -1.0, 1.0)
        audio_int16 = (audio * 32767.0).astype(np.int16)

        # Write WAV file
        with wave.open(str(path), "wb") as wav_file:
            wav_file.setnchannels(1)  # mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio_int16.tobytes())

    def _create_readme(self) -> None:
        """Create a human-readable README.txt describing the session."""
        if self.session_metadata is None:
            return

        meta = self.session_metadata
        readme_path = self.session_dir / "README.txt"

        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(f"Transcription Session: {meta.session_id}\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Timestamp: {meta.timestamp}\n")
            f.write(f"Backend: {meta.backend}\n")
            f.write(f"Sample Rate: {meta.sample_rate} Hz\n")
            f.write(f"Channels: {meta.channels}\n")
            f.write(f"Capture Duration: {meta.capture_duration:.2f}s\n")
            f.write(f"Total Chunks: {meta.total_chunks}\n\n")

            if meta.model:
                f.write(f"Model: {meta.model}\n")
            if meta.device:
                f.write(f"Device: {meta.device}\n")
            if meta.language:
                f.write(f"Language: {meta.language}\n")

            if meta.vad_aggressiveness is not None:
                f.write("\nVAD Parameters:\n")
                f.write(f"  Aggressiveness: {meta.vad_aggressiveness}\n")
                if meta.vad_min_silence_duration is not None:
                    f.write(f"  Min Silence Duration: {meta.vad_min_silence_duration}s\n")
                if meta.vad_min_speech_duration is not None:
                    f.write(f"  Min Speech Duration: {meta.vad_min_speech_duration}s\n")
                if meta.vad_speech_pad_duration is not None:
                    f.write(f"  Speech Pad Duration: {meta.vad_speech_pad_duration}s\n")
                if meta.max_chunk_duration is not None:
                    f.write(f"  Max Chunk Duration: {meta.max_chunk_duration}s\n")

            if meta.stitched_transcription:
                f.write("\n" + "=" * 60 + "\n")
                f.write("STITCHED TRANSCRIPTION (from chunks):\n")
                f.write("=" * 60 + "\n")
                f.write(meta.stitched_transcription + "\n")

            if meta.full_audio_transcription:
                f.write("\n" + "=" * 60 + "\n")
                f.write("FULL AUDIO TRANSCRIPTION (complete audio):\n")
                f.write("=" * 60 + "\n")
                f.write(meta.full_audio_transcription + "\n")

            # Add diff information if available
            if meta.stitched_transcription and meta.full_audio_transcription:
                f.write("\n" + "=" * 60 + "\n")
                f.write("TRANSCRIPTION COMPARISON:\n")
                f.write("=" * 60 + "\n")
                if meta.transcription_similarity is not None:
                    f.write(f"Similarity: {meta.transcription_similarity:.2%}\n")
                if meta.transcription_diffs and len(meta.transcription_diffs) > 0:
                    f.write(f"Differences found: {len(meta.transcription_diffs)}\n\n")
                    for i, diff in enumerate(meta.transcription_diffs, 1):
                        f.write(f"Diff {i} ({diff['tag']}):\n")
                        f.write(f"  Stitched: {diff['stitched']}\n")
                        f.write(f"  Complete: {diff['complete']}\n\n")
                elif meta.transcription_similarity == 1.0:
                    f.write("Transcriptions match exactly.\n")

            f.write("\n" + "=" * 60 + "\n")
            f.write("CHUNKS:\n")
            f.write("=" * 60 + "\n")
            for chunk in self.chunks:
                f.write(f"\nChunk {chunk.index:03d} [{chunk.start_time:.2f}s - {chunk.end_time:.2f}s]:\n")
                if chunk.inference_seconds is not None:
                    f.write(f"  Inference: {chunk.inference_seconds:.2f}s\n")
                f.write(f"  Text (original): {chunk.text}\n")
                if chunk.cleaned_text is not None:
                    f.write(f"  Text (cleaned): {chunk.cleaned_text}\n")
                if chunk.audio_filename:
                    f.write(f"  Audio: chunks/{chunk.audio_filename}\n")

        print(f"Created README: {readme_path}", file=sys.stderr)
