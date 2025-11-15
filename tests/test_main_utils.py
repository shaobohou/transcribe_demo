import io
import sys

import numpy as np
import pytest
from absl.testing import flagsaver

from transcribe_demo.backend_protocol import TranscriptionChunk
from transcribe_demo.chunk_collector import ChunkCollector
from transcribe_demo.cli import FLAGS, main as main_entry
from transcribe_demo.transcript_diff import (
    _colorize_token,
    _format_diff_snippet,
    _generate_diff_snippets,
    normalize_whitespace,
    _tokenize_with_original,
    print_transcription_summary,
)

if not FLAGS.is_parsed():
    FLAGS(["pytest"], known_only=True)


class ColorStream(io.StringIO):
    """StringIO stream that reports itself as a TTY for colorized output."""

    def isatty(self) -> bool:  # pragma: no cover - trivial wrapper
        return True


def test_chunk_collector_writes_and_stitches_chunks():
    stream = io.StringIO()
    collector = ChunkCollector(stream)

    collector(TranscriptionChunk(index=0, text="Hello, ", start_time=0.0, end_time=1.5, inference_seconds=0.25))
    collector(TranscriptionChunk(index=1, text="world.", start_time=1.5, end_time=3.0, inference_seconds=0.30))
    collector(TranscriptionChunk(index=2, text="How are you?", start_time=3.0, end_time=5.0, inference_seconds=0.45))

    output = stream.getvalue()
    assert "[chunk 000 | t=1.50s" in output
    assert "[chunk 001 | t=3.00s" in output
    assert "[chunk 002 | t=5.00s" in output
    assert "[STITCHED] Hello world How are you?" in output
    assert collector.get_final_stitched_text() == "Hello world How are you?"


def test_chunk_collector_colorized_output_in_realtime_mode():
    stream = ColorStream()
    collector = ChunkCollector(stream)

    collector(TranscriptionChunk(index=0, text="Realtime chunk", start_time=0.0, end_time=2.0, inference_seconds=None))

    output = stream.getvalue()
    assert "\x1b[36m" in output  # cyan label
    assert "\x1b[32m" not in output  # no stitched summary emitted for single chunk


@pytest.mark.parametrize(
    ("text", "is_final", "expected"),
    [
        ("Hello,", False, "Hello"),
        ("Hello.", False, "Hello"),
        ("Hello?", False, "Hello?"),
        ("Hello.", True, "Hello."),
        ("  spaced out  ,", False, "spaced out"),
    ],
)
def test_clean_chunk_text(text, is_final, expected):
    assert ChunkCollector._clean_chunk_text(text, is_final_chunk=is_final) == expected


def test_print_transcription_summary_reports_differences():
    stream = io.StringIO()
    print_transcription_summary(stream, "hello world", "hello brave world")

    written = stream.getvalue()
    assert "[FINAL STITCHED] hello world" in written
    assert "[COMPLETE AUDIO] hello brave world" in written
    assert "[COMPARISON] Difference detected" in written
    # Diff snippets should include stitched/complete lines
    assert "stitched:" in written
    assert "complete:" in written


def test_print_transcription_summary_identical_texts(capsys):
    stream = io.StringIO()
    print_transcription_summary(stream, " same ", "same")

    written = stream.getvalue()
    assert "matches complete audio transcription" in written


def test_tokenize_with_original_strips_punctuation():
    tokens = _tokenize_with_original("Hello, WORLD! it's me...")
    assert tokens == [
        ("Hello,", "hello"),
        ("WORLD!", "world"),
        ("it's", "it's"),
        ("me...", "me"),
    ]


def test_generate_diff_snippets_describes_changes():
    snippets = _generate_diff_snippets("hello world", "hello brave world", use_color=False)
    assert snippets  # at least one snippet
    tags = {entry["tag"] for entry in snippets}
    assert tags <= {"insert", "replace"}


def test_colorize_token_respects_color_flag():
    colored = _colorize_token("token", True, "33")
    assert colored.startswith("\x1b[2;33m") and colored.endswith("\x1b[0m")
    plain = _colorize_token("token", False, "33")
    assert plain == "[[token]]"


def test_format_diff_snippet_insertion_placeholder():
    tokens = _tokenize_with_original("one two three")
    snippet = _format_diff_snippet(tokens, diff_start=1, diff_end=1, use_color=False, color_code="36")
    assert "[[∅]]" in snippet


def test_normalize_whitespace_collapses_gaps():
    assert normalize_whitespace("alpha   beta\tgamma\n") == "alpha beta gamma"


def test_main_exits_when_refine_flag_enabled(monkeypatch):
    stdout = io.StringIO()
    stderr = io.StringIO()
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)

    with flagsaver.flagsaver():
        FLAGS.refine_with_context = True
        with pytest.raises(SystemExit) as exc:
            main_entry(["prog"])

    assert exc.value.code == 1
    assert "--refine-with-context" in stderr.getvalue()


def test_main_whisper_flow_prints_summary(monkeypatch, temp_session_dir):
    import queue
    from test_helpers import create_fake_audio_capture_factory, generate_synthetic_audio

    class FakeCollector:
        def __init__(self, stream):
            self.stream = stream
            self.calls = []

        def __call__(self, chunk):
            self.calls.append((chunk.index, chunk.text, chunk.start_time, chunk.end_time, chunk.inference_seconds))
            self.stream.write(f"[fake {chunk.index}] {chunk.text}\n")

        def get_final_stitched_text(self):
            return "stitched text"

        def get_cleaned_chunks(self):
            return [(i, text) for i, (_, text, _, _, _) in enumerate(self.calls)]

    class DummyResult:
        def __init__(self):
            self.full_audio_transcription = "complete audio text"
            self.capture_duration = 15.0
            self.metadata = {"model": "test", "device": "cpu"}

    # Create synthetic audio for mocking
    audio, sample_rate = generate_synthetic_audio(duration_seconds=2.0)

    stdout = io.StringIO()
    stderr = io.StringIO()
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)
    # Monkeypatch where ChunkCollector is actually imported and used (in cli.py)
    monkeypatch.setattr(
        "transcribe_demo.cli.ChunkCollector",
        FakeCollector,
    )

    # Mock AudioCaptureManager
    monkeypatch.setattr(
        "transcribe_demo.audio_capture.AudioCaptureManager",
        create_fake_audio_capture_factory(audio, sample_rate, frame_size=480),
    )

    # Mock backend to put chunks in queue
    def fake_run_whisper(**kwargs):
        chunk_queue = kwargs.get("chunk_queue")
        if chunk_queue:
            # Put one test chunk
            from transcribe_demo.backend_protocol import TranscriptionChunk
            chunk_queue.put(TranscriptionChunk(
                index=0, text="test", start_time=0.0, end_time=1.0, inference_seconds=0.1
            ))
            chunk_queue.put(None)  # Sentinel
        return DummyResult()

    monkeypatch.setattr(
        "transcribe_demo.whisper_backend.run_whisper_transcriber",
        fake_run_whisper,
    )

    with flagsaver.flagsaver(
        backend="whisper",
        compare_transcripts=True,
        refine_with_context=False,
        temp_file=None,
        session_log_dir=str(temp_session_dir),
    ):
        main_entry(["prog"])

    output = stdout.getvalue()
    assert "[FINAL STITCHED] stitched text" in output
    assert "[COMPLETE AUDIO] complete audio text" in output


def test_main_realtime_flow_without_comparison(monkeypatch, temp_session_dir):
    from test_helpers import create_fake_audio_capture_factory, generate_synthetic_audio

    class FakeCollector:
        def __init__(self, stream):
            self.stream = stream

        def __call__(self, *args, **kwargs):
            self.stream.write("[fake realtime]\n")

        def get_final_stitched_text(self):
            return "realtime stitched"

        def get_cleaned_chunks(self):
            return []

    class DummyRealtimeResult:
        def __init__(self):
            self.full_audio = np.zeros(0, dtype=np.float32)
            self.sample_rate = 24000
            self.capture_duration = 15.0
            self.metadata = {"model": "test-realtime"}
            self.full_audio_transcription = None  # Required by TranscriptionResult protocol

    # Create synthetic audio for mocking
    audio, sample_rate = generate_synthetic_audio(duration_seconds=2.0)

    stdout = io.StringIO()
    stderr = io.StringIO()
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)
    monkeypatch.setenv("OPENAI_API_KEY", "")
    # Monkeypatch where ChunkCollector is actually imported and used (in cli.py)
    monkeypatch.setattr("transcribe_demo.cli.ChunkCollector", FakeCollector)

    # Mock AudioCaptureManager
    monkeypatch.setattr(
        "transcribe_demo.audio_capture.AudioCaptureManager",
        create_fake_audio_capture_factory(audio, sample_rate, frame_size=320),
    )

    # Mock backend to put chunks in queue
    def fake_run_realtime(**kwargs):
        chunk_queue = kwargs.get("chunk_queue")
        if chunk_queue:
            # Put one test chunk
            from transcribe_demo.backend_protocol import TranscriptionChunk
            chunk_queue.put(TranscriptionChunk(
                index=0, text="realtime", start_time=0.0, end_time=1.0, inference_seconds=None
            ))
            chunk_queue.put(None)  # Sentinel
        return DummyRealtimeResult()

    monkeypatch.setattr(
        "transcribe_demo.realtime_backend.run_realtime_transcriber",
        fake_run_realtime,
    )
    monkeypatch.setattr(
        "transcribe_demo.realtime_backend.transcribe_full_audio_realtime",
        lambda *args, **kwargs: "full realtime transcription",
    )

    with flagsaver.flagsaver(
        backend="realtime",
        api_key="dummy",
        compare_transcripts=False,
        refine_with_context=False,
        session_log_dir=str(temp_session_dir),
    ):
        main_entry(["prog"])

    assert "[FINAL STITCHED] realtime stitched" in stdout.getvalue()
