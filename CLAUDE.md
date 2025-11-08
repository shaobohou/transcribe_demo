# CLAUDE.md

**⚠️ Update this file when changing: defaults, architecture, CLI args, or test strategy**

Development workflow and critical implementation rules. See **DESIGN.md** for architecture, **TODO.md** for improvements, **README.md** for user docs.

## Common Commands

```bash
uv sync                                 # Install dependencies
uv run transcribe-demo                  # Run with default settings (turbo + VAD)
uv run python -m pytest                 # Run all tests (pre-commit hook enforces)
uv run python -m pytest tests/test_vad.py  # Run VAD tests only

# CPU-only torch (avoids CUDA downloads, matches CI behavior)
python3 -m venv .venv
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple -e ".[dev]"
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

### CI Testing (No Audio Hardware)

**Mocking strategy (tests/conftest.py):**
- **sounddevice**: Module-level mock (not fixture) - pytest imports during collection before fixtures run
- **stdin.isatty()**: Autouse fixture returns False - prevents blocking listener threads

**Monkeypatching AudioCaptureManager:**
- Backends: `from transcribe_demo import audio_capture as audio_capture_lib`
- Tests: `monkeypatch.setattr("transcribe_demo.audio_capture.AudioCaptureManager", FakeCls)`
- Use STRING-based paths (not attribute-based) for reliable cross-module mocking

**FakeAudioCaptureManager critical requirements:**
- Set `stop_event` after feeding audio (prevents hanging in `wait_until_stopped()`)
- Respect `max_capture_duration` with `capture_limit_reached` flag
- Add 1ms delay after limit to give backend time to process queued frames

**CPU-only PyTorch for CI (Avoid CUDA Dependencies):**
- CI uses regular pip (not uv) with PyTorch's CPU index to avoid downloading CUDA dependencies entirely
- Command: `.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple -e ".[dev]"`
- Why pip not uv? uv doesn't properly support PyTorch's custom package index
- `--index-url` makes PyTorch CPU index primary, `--extra-index-url` adds PyPI for other packages
- This completely avoids ~3GB of CUDA downloads, installing only ~200MB CPU-only torch
- Tests run on CPU with mocked audio hardware (see mocking strategy above)
- For local development with GPU, use regular `uv sync` which installs GPU-enabled torch from PyPI

## Related Documentation

See [SITEMAP.md](SITEMAP.md) for a complete guide to all documentation.
