import io
import sys

import numpy as np
import pytest
from absl.testing import flagsaver

from transcribe_demo.main import (
    ChunkCollectorWithStitching,
    FLAGS,
    _colorize_token,
    _format_diff_snippet,
    _generate_diff_snippets,
    _normalize_whitespace,
    _tokenize_with_original,
    main as main_entry,
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
    collector = ChunkCollectorWithStitching(stream)

    collector(
        chunk_index=0,
        text="Hello, ",
        absolute_start=0.0,
        absolute_end=1.5,
        inference_seconds=0.25,
    )
    collector(
        chunk_index=1,
        text="world.",
        absolute_start=1.5,
        absolute_end=3.0,
        inference_seconds=0.30,
    )
    collector(
        chunk_index=2,
        text="How are you?",
        absolute_start=3.0,
        absolute_end=5.0,
        inference_seconds=0.45,
    )

    output = stream.getvalue()
    assert "[chunk 000 | t=1.50s" in output
    assert "[chunk 001 | t=3.00s" in output
    assert "[chunk 002 | t=5.00s" in output
    assert "[STITCHED] Hello world How are you?" in output
    assert collector.get_final_stitched() == "Hello world How are you?"


def test_chunk_collector_colorized_output_in_realtime_mode():
    stream = ColorStream()
    collector = ChunkCollectorWithStitching(stream)

    collector(
        chunk_index=0,
        text="Realtime chunk",
        absolute_start=0.0,
        absolute_end=2.0,
        inference_seconds=None,
    )

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
    assert ChunkCollectorWithStitching._clean_chunk_text(text, is_final_chunk=is_final) == expected


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
    assert _normalize_whitespace("alpha   beta\tgamma\n") == "alpha beta gamma"


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


def test_main_whisper_flow_prints_summary(monkeypatch):
    class FakeCollector:
        def __init__(self, stream):
            self.stream = stream
            self.calls = []

        def __call__(self, chunk_index, text, absolute_start, absolute_end, inference_seconds):
            self.calls.append(
                (chunk_index, text, absolute_start, absolute_end, inference_seconds)
            )
            self.stream.write(f"[fake {chunk_index}] {text}\n")

        def get_final_stitched(self):
            return "stitched text"

    class DummyResult:
        def __init__(self):
            self.full_audio_transcription = "complete audio text"

    stdout = io.StringIO()
    stderr = io.StringIO()
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)
    monkeypatch.setattr(
        "transcribe_demo.main.ChunkCollectorWithStitching",
        FakeCollector,
    )
    monkeypatch.setattr(
        "transcribe_demo.main.run_whisper_transcriber",
        lambda **kwargs: DummyResult(),
    )

    with flagsaver.flagsaver(
        backend="whisper",
        compare_transcripts=True,
        refine_with_context=False,
        temp_file=None,
    ):
        main_entry(["prog"])

    output = stdout.getvalue()
    assert "[FINAL STITCHED] stitched text" in output
    assert "[COMPLETE AUDIO] complete audio text" in output


def test_main_realtime_flow_without_comparison(monkeypatch):
    class FakeCollector:
        def __init__(self, stream):
            self.stream = stream

        def __call__(self, *args, **kwargs):
            self.stream.write("[fake realtime]\n")

        def get_final_stitched(self):
            return "realtime stitched"

    class DummyRealtimeResult:
        def __init__(self):
            self.full_audio = np.zeros(0, dtype=np.float32)
            self.sample_rate = 24000

    stdout = io.StringIO()
    stderr = io.StringIO()
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setattr("transcribe_demo.main.ChunkCollectorWithStitching", FakeCollector)
    monkeypatch.setattr(
        "transcribe_demo.main.run_realtime_transcriber",
        lambda **kwargs: DummyRealtimeResult(),
    )
    monkeypatch.setattr(
        "transcribe_demo.main.transcribe_full_audio_realtime",
        lambda *args, **kwargs: "full realtime transcription",
    )

    with flagsaver.flagsaver(
        backend="realtime",
        api_key="dummy",
        compare_transcripts=False,
        refine_with_context=False,
    ):
        main_entry(["prog"])

    assert "[FINAL STITCHED] realtime stitched" in stdout.getvalue()
