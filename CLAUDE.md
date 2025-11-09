# CLAUDE.md

**Purpose:** This file provides Claude Code with essential context about the transcribe_demo project. It contains development workflows, critical implementation rules, and testing strategies. Claude automatically reads this file at the start of every session.

**⚠️ Update this file when changing: defaults, architecture, CLI args, or test strategy**

**See also:** [DESIGN.md](DESIGN.md) for architecture, [TODO.md](TODO.md) for improvements, [README.md](README.md) for user docs, [SITEMAP.md](SITEMAP.md) for complete documentation guide.

---

## Tech Stack

**Core Technologies:**
- Python 3.12+ with modern type hints (`list[str]`, `str | None`)
- Package manager: `uv` (NOT pip/poetry)
- Testing: pytest + pyright + ruff
- Audio processing: sounddevice, webrtcvad, openai-whisper
- ML frameworks: torch (CUDA/MPS/CPU), numpy
- APIs: OpenAI Realtime API (WebSocket)

**Project Structure:**
- `src/transcribe_demo/` - Main source code
- `tests/` - Test files (must match `test_*.py` pattern)
- `.claude/` - Claude Code configuration
- `vendor/` - Vendored dependencies (triton-cpu-stub)

---

## Critical Constraints

**MANDATORY - Always perform these validation checks:**

### Pre-Action Validation
1. Read existing code before modifying
2. Check test files for existing patterns
3. Verify imports and dependencies
4. Review related documentation (DESIGN.md, TODO.md)

### Post-Action Validation
1. Run all tests: `uv --project ci run python -m pytest`
2. Type check: `uv run pyright`
3. Lint check: `uv run ruff check`
4. Verify code follows project style (see Code Standards below)

**Breaking any of these will fail CI/CD checks.**

---

## Common Commands

```bash
# Development (with CUDA/MPS support)
uv sync                                 # Install dependencies
uv run transcribe-demo                  # Run with default settings (turbo + VAD)
uv run python -m pytest                 # Run all tests
uv run python -m pytest tests/test_vad.py  # Run specific tests
uv run pyright                          # Type checking
uv run ruff check                       # Linting

# CI/Sandbox (CPU-only, no CUDA downloads)
uv sync --project ci --refresh              # CPU-only sync
uv --project ci run transcribe-demo --audio_file audio.mp3
uv --project ci run python -m pytest        # Tests in CPU-only env
```

**⚠️ IMPORTANT:** In sandboxes/CI, `uv run transcribe-demo` will download gigabytes of CUDA packages. **Always use `uv --project ci run` instead.**

**Why two workspaces?**
- Default `uv sync` resolves official `torch` and `triton` wheels (CUDA/MPS support)
- CI workspace: `uv sync --project ci` uses `torch==2.9.0+cpu` and vendored `triton-cpu-stub`
- **Never use CPU workspace for daily development**

---

## Code Standards

**Type Annotations:**
- ✅ Use modern syntax: `list[str]`, `dict[str, int]`, `str | None`
- ❌ Avoid old syntax: `List[str]`, `Dict[str, int]`, `Optional[str]`
- ✅ Always include return type annotations on functions
- ✅ Use `from __future__ import annotations` for forward references

**Naming Conventions:**
- Functions/variables: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`
- Private members: `_leading_underscore`

**Imports:**
- Group by: stdlib, third-party, local (separated by blank lines)
- Use absolute imports: `from transcribe_demo.whisper_backend import ...`
- ❌ Avoid relative imports in source code

**Formatting:**
- Follow existing code style (ruff enforces)
- No trailing whitespace
- Max line length: 120 characters (enforced by ruff)

**Documentation:**
- Docstrings: Use Google style
- Inline comments: Explain "why", not "what"
- Update CLAUDE.md when changing defaults or architecture

---

## File Structure

**Canonical Locations:**
```
src/transcribe_demo/
├── main.py                 # CLI entrypoint, argument parsing, ChunkCollectorWithStitching
├── whisper_backend.py      # WebRTCVAD class, run_whisper_transcriber() main loop
├── realtime_backend.py     # run_realtime_transcriber() WebSocket streaming
├── audio_capture.py        # AudioCaptureManager for microphone capture
├── file_audio_source.py    # FileAudioSource for file/URL simulation
├── session_logger.py       # Session logging and persistence
├── session_replay.py       # Session replay and retranscription
└── vad.py                  # VAD utilities

tests/
├── conftest.py            # Shared fixtures and mocks
├── test_helpers.py        # Test utilities (FakeAudioCaptureManager, etc.)
├── test_*.py              # Test files (match source file names)
└── fixtures/              # Test audio files
```

**When adding new functionality:**
- Backend logic → `*_backend.py`
- Audio sources → `*_audio_source.py`
- Tests → `tests/test_*.py` (must start with `test_`)
- Utilities → Match closest existing pattern

---

## Development Workflow

### Branch Strategy
**YOU MUST always develop on a feature branch** - NEVER commit directly to `main`.

**Creating a branch:**
```bash
git checkout -b feature/descriptive-name   # For new features
git checkout -b fix/bug-description        # For bug fixes
git checkout -b refactor/what-changed      # For refactoring
```

**Before creating a PR:**
1. ✅ All tests pass: `uv run python -m pytest`
2. ✅ Type checking clean: `uv run pyright` (0 errors)
3. ✅ Linting passes: `uv run ruff check`
4. ✅ Commit and push: `git push -u origin your-branch-name`

**CI/CD:** GitHub Actions runs all checks automatically. All must pass before merge.

### Commit Format
Follow conventional commits:
```
type(scope): subject

body (optional)
```

**Types:** `feat`, `fix`, `refactor`, `test`, `docs`, `chore`
**Example:** `feat(realtime): add partial chunk flushing on timeout`

---

## Critical Implementation Rules

### VAD Chunking (Whisper Backend)
**Core principle:** Chunks split at natural speech pauses = NO audio overlap between chunks

**Stitching Logic (DO NOT break this):**
- Strip trailing `,` and `.` from intermediate chunks
- Preserve `?` and `!` (they signal intentional sentence ends)
- Whisper punctuates assuming chunk completeness, but VAD splits mid-sentence
- Breaking this creates unnatural transcripts like "Hello, world,. How are you."

**Implementation:** See `ChunkCollectorWithStitching._clean_chunk_text()` in main.py:219-232

### Language Parameter
- **Default:** `language="en"` prevents hallucinations on silence/noise
- **WARNING:** Changing this affects transcription quality on non-speech audio
- Auto-detection (`language="auto"`) can hallucinate on silence

### Device Selection
- Auto-detection order: CUDA → MPS → CPU
- `--require_gpu` flag aborts if no GPU detected
- Never assume device availability in tests

### Audio Source Selection
**Two implementations with identical interface:**
- **AudioCaptureManager**: Microphone capture (default)
- **FileAudioSource**: File/URL simulation (`--audio_file` flag)

**Common interface:**
- `audio_queue`: Queue for audio frames
- `stop_event`: Event to signal stop
- `get_full_audio()`: Returns complete audio buffer
- Both support `max_capture_duration` time limits

**File source specifics:**
- Simulates real-time playback
- Configurable speed: `--playback_speed` (default 1.0 = real-time)
- Supports local files and HTTP/HTTPS URLs

### SSL/Certificate Configuration
- `--ca_cert`: Custom certificate bundle for corporate proxies
- `--disable_ssl_verify`: Bypass SSL verification (⚠️ **insecure**)
  - Affects: Model downloads (Whisper) + Realtime API connections
  - Only use in restricted networks with certificate issues
  - Never use in production

---

## Testing

**YOU MUST run tests before commits** - pre-commit hook enforces this.

### Test Requirements
- All tests pass: `uv --project ci run python -m pytest`
- Type checking clean: `uv run pyright` (0 errors)
- Linting passes: `uv run ruff check`
- Test coverage maintained (no decrease without justification)

### Test Types
- **Unit tests** (~0.1s): `test_vad.py`, `test_main_utils.py`, `test_file_audio_source.py`, backend utils
- **Integration tests** (2-3s): `test_backend_time_limits.py`, backend integration tests

### Critical Testing Gotchas

**1. VAD + Time Limits = Flaky Tests**
- VAD chunking is unpredictable - avoid combining with strict time limits
- Use synthetic audio: `generate_synthetic_audio(duration_seconds=2.0)`
- Keep `max_chunk_duration` high (≥10s) to prevent premature splits
- Test VAD behavior separately from timing constraints

**2. Time Limits vs User Stop**
- Both use the SAME mechanism (`stop_event`)
- Don't test user stop separately - it's redundant
- Time limit tests cover both scenarios

**3. Compare Transcripts**
- Test with Realtime backend only (stable fixed chunks)
- Whisper + VAD is flaky due to unpredictable chunk timing
- Full audio comparison depends on deterministic chunking

### Writing Fast Tests
- ✅ Use synthetic audio (not real files)
- ✅ Disable extras: `save_chunk_audio=False`
- ✅ Keep audio duration minimal (2-4s)
- ✅ Use high `playback_speed` (10.0x) for FileAudioSource tests
- ❌ Never add `time.sleep()` in test infrastructure
- ❌ Don't use real network requests (mock instead)

**Run tests 3-5 times to check for flakiness before committing.**

### Mocking Strategy (CI/No Audio Hardware)

**conftest.py patterns:**
- `sounddevice`: Module-level mock (NOT fixture)
  - Why: pytest imports during collection before fixtures run
- `stdin.isatty()`: Autouse fixture returns False
  - Why: prevents blocking listener threads

**Monkeypatching AudioCaptureManager:**
```python
# In backends (import pattern):
from transcribe_demo import audio_capture as audio_capture_lib

# In tests (monkeypatch pattern):
monkeypatch.setattr("transcribe_demo.audio_capture.AudioCaptureManager", FakeCls)
```
- ✅ Use STRING-based paths (not attribute-based)
- Why: reliable cross-module mocking

**FakeAudioCaptureManager requirements:**
- Set `stop_event` after feeding audio (prevents hanging in `wait_until_stopped()`)
- Respect `max_capture_duration` with `capture_limit_reached` flag
- Add 1ms delay after limit to give backend time to process queued frames

**Testing FileAudioSource with URLs:**
```python
# Mock urllib to avoid actual network requests
unittest.mock.patch("transcribe_demo.file_audio_source.urlopen")
```
- Create mock response with audio file content from temporary test file
- See `test_file_audio_source.py::test_file_audio_source_url_detection`

---

## Performance & Quality Standards

**Test Performance:**
- Unit tests: < 0.5s each
- Integration tests: < 5s each
- Full test suite: < 30s

**Type Safety:**
- Zero `pyright` errors (enforced by CI)
- Zero `type: ignore` comments without explanation

**Code Quality:**
- Zero `ruff` violations (enforced by CI)
- No unused imports or variables
- All docstrings present for public APIs

**Breaking these standards will fail CI checks.**

---

## Configuration Defaults

**When to change defaults:**

| Parameter | Default | Change when... |
|-----------|---------|----------------|
| `model` | `"turbo"` | Speed/accuracy tradeoff needs adjustment |
| `vad_aggressiveness` | `2` | Missing speech (increase) or capturing noise (decrease) |
| `vad_min_silence_duration` | `0.2s` | Want slower chunking (increase) or faster response (decrease) |
| `max_chunk_duration` | `60s` | Seeing duration warnings during long speech |
| `language` | `"en"` | Transcribing non-English audio (but test carefully) |

**Realtime backend:**
- Fixed 2.0s chunks - **NOT configurable**
- No VAD (server-side VAD only)

---

## Common Pitfalls

**DO NOT:**
- ❌ Change VAD default values without testing on real audio samples
- ❌ Modify punctuation stripping logic without understanding stitching implications
- ❌ Skip running tests before commits (pre-commit hook will block anyway)
- ❌ Remove or change `language="en"` without considering hallucination risks
- ❌ Assume Realtime backend uses VAD chunking (it doesn't - server-side only)
- ❌ Use `time.sleep()` in test infrastructure (causes flaky tests)
- ❌ Combine VAD with strict time limits in tests (unpredictable)
- ❌ Use `uv run` commands in CI/sandboxes (use `uv --project ci run` instead)

---

## Key Files Reference

| File | Purpose | Key Components |
|------|---------|----------------|
| `main.py` | CLI entrypoint | `parse_args()`, `ChunkCollectorWithStitching`, stitching logic |
| `whisper_backend.py` | Whisper transcription | `WebRTCVAD` class, `run_whisper_transcriber()` main loop |
| `realtime_backend.py` | Realtime API | `run_realtime_transcriber()` WebSocket streaming |
| `audio_capture.py` | Microphone input | `AudioCaptureManager` for live capture |
| `file_audio_source.py` | File/URL input | `FileAudioSource` for simulated live transcription |
| `session_logger.py` | Persistence | Session logging, audio saving, metadata |
| `session_replay.py` | Replay | Load and retranscribe saved sessions |

---

## Related Documentation

See [SITEMAP.md](SITEMAP.md) for a complete guide to all documentation.
