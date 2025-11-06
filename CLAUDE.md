# CLAUDE.md

**⚠️ Update this file when changing: defaults, architecture, CLI args, or test strategy**

User-facing docs are in README.md - this is for development only.

## Common Commands

```bash
uv sync                           # Install dependencies
uv run transcribe-demo            # Run with default settings (turbo + VAD)
uv run pytest                     # Run all tests (pre-commit hook enforces)
uv run pytest tests/test_vad.py   # Run VAD tests only
```

## Key Files

- **main.py**: `parse_args()` for CLI changes, `ChunkCollectorWithStitching` handles stitching logic
- **whisper_backend.py**: `WebRTCVAD` class, `run_whisper_transcriber()` main loop
- **realtime_backend.py**: `run_realtime_transcriber()` WebSocket streaming

## Critical Implementation Rules

### VAD Chunking (Whisper Backend)
**YOU MUST understand:** Chunks split at natural speech pauses = NO audio overlap between chunks

**IMPORTANT - Stitching logic:**
- Strip trailing `,` and `.` from intermediate chunks (preserve `?` and `!`)
- Whisper punctuates assuming chunk completeness, but VAD splits mid-sentence
- Breaking this creates unnatural transcripts

### Language Parameter
**Default:** `language="en"` to prevent hallucinations on silence/noise
**WARNING:** Changing this affects transcription quality on non-speech audio

### Device Selection
- Auto-detection order: CUDA → MPS → CPU
- `--require-gpu` flag aborts if no GPU detected

## When to Change Defaults

**Model (`turbo`)**: Change if speed/accuracy tradeoff needs adjustment
**VAD aggressiveness (`2`)**: Increase if missing speech, decrease if capturing noise
**Min silence duration (`0.2s`)**: Increase for slower chunking, decrease for faster response
**Max chunk duration (`60s`)**: Increase if seeing duration warnings during long speech

Realtime backend has fixed 2.0s chunks - NOT configurable.

## Do Not

- **DO NOT** change VAD default values without testing on real audio samples
- **DO NOT** modify punctuation stripping logic without understanding stitching implications
- **DO NOT** skip running tests before commits (pre-commit hook will block anyway)
- **DO NOT** remove or change `language="en"` without considering hallucination risks
- **DO NOT** assume Realtime backend uses VAD chunking (it doesn't)

## Testing

**YOU MUST** run tests before commits - pre-commit hook enforces this.
When modifying VAD logic, add test cases to `tests/test_vad.py`.
Always run `uv run pyright` to confirm the type checker is clean before submitting changes.
Use PEP 585 generics (`list[str]`, `dict[str, int]`) instead of `typing.List`/`typing.Dict`, and prefer union syntax (`str | None`) over `typing.Optional`.
Treat linting and formatting as mandatory test steps; keep `uv run ruff check` (and any repo-specific formatters) passing before submitting changes.

## Related Files

- **REFACTORING.md**: Known refactoring opportunities
