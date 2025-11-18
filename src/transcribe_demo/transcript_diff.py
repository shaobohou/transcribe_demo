"""Transcription comparison and output formatting utilities.

This module provides functions for comparing transcriptions, computing diffs,
and formatting output with colored highlights.
"""

from __future__ import annotations

import difflib
import re
from typing import TextIO


def _normalize_whitespace(*, text: str) -> str:
    """Normalize whitespace in text by collapsing multiple spaces into one."""
    return " ".join(text.split())


def print_final_stitched(*, stream: TextIO, text: str) -> None:
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


def compute_transcription_diff(*, stitched_text: str, complete_text: str) -> tuple[float, list[dict[str, str]]]:
    """
    Compute diff between stitched and complete transcriptions.

    Args:
        stitched_text: Text from stitched chunks
        complete_text: Text from complete audio transcription

    Returns:
        Tuple of (similarity_ratio, diff_snippets)
        - similarity_ratio: Float between 0.0 and 1.0
        - diff_snippets: List of dicts with 'tag', 'stitched', 'complete' keys
    """
    if not (stitched_text.strip() and complete_text.strip()):
        return (0.0, [])

    stitched_tokens_norm = [norm for _, norm in _tokenize_with_original(text=stitched_text)]
    complete_tokens_norm = [norm for _, norm in _tokenize_with_original(text=complete_text)]
    similarity = difflib.SequenceMatcher(None, stitched_tokens_norm, complete_tokens_norm).ratio()
    diff_snippets = _generate_diff_snippets(stitched_text=stitched_text, complete_text=complete_text, use_color=False)

    return (similarity, diff_snippets)


def print_transcription_summary(
    *,
    stream: TextIO,
    final_text: str,
    complete_audio_text: str,
) -> None:
    """
    Print stitched transcript, complete-audio transcript, and comparison details.

    Args:
        stream: Output stream (e.g., sys.stdout)
        final_text: Final stitched transcription text
        complete_audio_text: Complete audio transcription text
    """
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

    stitched_tokens_norm = [norm for _, norm in _tokenize_with_original(text=final_clean)]
    complete_tokens_norm = [norm for _, norm in _tokenize_with_original(text=complete_audio_clean)]
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
    diff_snippets = _generate_diff_snippets(
        stitched_text=final_clean, complete_text=complete_audio_clean, use_color=use_color
    )
    for snippet in diff_snippets:
        print(
            f"{diff_label} {snippet['tag']}:\n    stitched: {snippet['stitched']}\n    complete: {snippet['complete']}",
            file=stream,
        )


def _tokenize_with_original(*, text: str) -> list[tuple[str, str]]:
    """
    Return (raw, normalized) tokens where normalized strips punctuation and lowercases.

    Args:
        text: Input text to tokenize

    Returns:
        List of (raw_token, normalized_token) tuples
    """
    tokens: list[tuple[str, str]] = []
    for raw in text.split():
        normalized = re.sub(r"[^\w']+", "", raw).lower()
        if not normalized:
            continue
        tokens.append((raw, normalized))
    return tokens


def _colorize_token(*, token: str, use_color: bool, color_code: str) -> str:
    """
    Colorize a token with ANSI color codes or bracket notation.

    Args:
        token: Token to colorize
        use_color: Whether to use ANSI color codes
        color_code: ANSI color code (e.g., "33" for yellow)

    Returns:
        Colorized token string
    """
    if use_color:
        return f"\x1b[2;{color_code}m{token}\x1b[0m"
    return f"[[{token}]]"


def _format_diff_snippet(
    *,
    tokens: list[tuple[str, str]],
    diff_start: int,
    diff_end: int,
    use_color: bool,
    color_code: str,
) -> str:
    """
    Format a diff snippet with context and highlighting.

    Args:
        tokens: List of (raw, normalized) token tuples
        diff_start: Start index of diff region
        diff_end: End index of diff region
        use_color: Whether to use ANSI color codes
        color_code: ANSI color code for highlighting

    Returns:
        Formatted diff snippet string
    """
    if not tokens:
        return _colorize_token(token="∅", use_color=use_color, color_code=color_code)

    context = 3
    window_end = max(diff_end, diff_start)
    start = max(diff_start - context, 0)
    end = min(window_end + context, len(tokens))
    parts: list[str] = []

    for idx in range(start, end):
        raw = tokens[idx][0]
        if diff_start <= idx < diff_end:
            parts.append(_colorize_token(token=raw, use_color=use_color, color_code=color_code))
        else:
            parts.append(raw)

    if diff_start == diff_end:
        placeholder = _colorize_token(token="∅", use_color=use_color, color_code=color_code)
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
    return snippet or _colorize_token(token="∅", use_color=use_color, color_code=color_code)


def _generate_diff_snippets(
    *,
    stitched_text: str,
    complete_text: str,
    use_color: bool,
) -> list[dict[str, str]]:
    """
    Generate diff snippets from stitched and complete text.

    Args:
        stitched_text: Stitched transcription text
        complete_text: Complete transcription text
        use_color: Whether to use ANSI color codes

    Returns:
        List of diff snippets, each with 'tag', 'stitched', 'complete' keys
    """
    stitched_tokens = _tokenize_with_original(text=stitched_text)
    complete_tokens = _tokenize_with_original(text=complete_text)
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
                "stitched": _format_diff_snippet(
                    tokens=stitched_tokens, diff_start=i1, diff_end=i2, use_color=use_color, color_code="33"
                ),
                "complete": _format_diff_snippet(
                    tokens=complete_tokens, diff_start=j1, diff_end=j2, use_color=use_color, color_code="36"
                ),
            }
        )

    return snippets
