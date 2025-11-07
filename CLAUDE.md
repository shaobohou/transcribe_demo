# CLAUDE.md

**⚠️ Update this file when changing: defaults, architecture, CLI args, or test strategy**

User-facing docs are in README.md - this is for development only.

## Common Commands

```bash
uv sync                                 # Install dependencies
uv run transcribe-demo                  # Run with default settings (turbo + VAD)
uv run python -m pytest                 # Run all tests (pre-commit hook enforces)
uv run python -m pytest tests/test_vad.py  # Run VAD tests only
```

## Development Workflow

### Branch Strategy
**YOU MUST always develop on a feature branch** - NEVER commit directly to `main`.

**Creating a feature branch:**
```bash
git checkout -b feature/descriptive-name
# or
git checkout -b fix/bug-description
```

**Branch naming conventions:**
- Feature: `feature/descriptive-name`
- Bug fix: `fix/bug-description`
- Refactor: `refactor/what-you-refactor`

**Before creating a PR:**
1. Ensure all tests pass: `uv run python -m pytest`
2. Type checking passes: `uv run pyright`
3. Linting passes: `uv run ruff check`
4. Commit and push your branch: `git push -u origin your-branch-name`

**CI/CD:** GitHub Actions runs all checks automatically on PRs. All checks must pass before merge.

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

### Test Types
- **Unit tests** (~0.1s): `test_vad.py`, `test_main_utils.py`, backend utils
- **Integration tests** (2-3s): `test_backend_time_limits.py`, backend integration tests

### Critical Testing Gotchas

**VAD + Time Limits = Flaky Tests**
- VAD chunking is unpredictable - avoid combining with strict time limits
- Use synthetic audio: `_generate_synthetic_audio(duration_seconds=2.0)`
- Keep `max_chunk_duration` high (≥10s) to prevent premature splits
- Test VAD behavior separately from timing

**Time Limits vs User Stop**
- These use the SAME mechanism (stop_event) - don't test user stop separately

**Compare Transcripts**
- Test with realtime backend only (stable fixed chunks)
- Whisper + VAD is flaky due to unpredictable chunk timing

### Writing Fast Tests
- Use synthetic audio, not real files
- Disable extras: `save_chunk_audio=False`
- Never add `time.sleep()` in test infrastructure
- Keep audio duration minimal (2-4s)
- Run 3-5 times to check for flakiness

### Before Committing
```bash
uv run python -m pytest  # All tests must pass
uv run pyright           # Must be clean (0 errors)
uv run ruff check        # Must pass
```

**Code style:** Use `list[str]` not `List[str]`, use `str | None` not `Optional[str]`

## Related Files

- **REFACTORING.md**: Known refactoring opportunities
