# CLAUDE.md

**AI Assistant Instructions for transcribe-demo**

⚠️ **Update this file when changing:** defaults, architecture, CLI args, or test strategy

Development workflow and critical implementation rules. See **DESIGN.md** for architecture, **README.md** for user docs, **SITEMAP.md** for all documentation.

## Common Commands

```bash
# Development
uv sync                                   # Install with GPU support (local dev)
uv run transcribe-demo                    # Run with defaults (turbo + VAD)
./run-checks.sh                           # Run all checks (pytest w/ timeout, pyright, ruff)
uv run python -m pytest                   # Run all tests
uv run pyright                            # Type checking
uv run ruff check                         # Linting
uv run ruff format                        # Format code

# CPU-only (CI/sandboxes - REQUIRED!)
uv sync --project ci --refresh
./run-checks.sh --ci                      # Run all checks (CPU-only, no CUDA downloads)
uv --project ci run transcribe-demo --audio_file audio.mp3 --model base.en
uv --project ci run python -m pytest
```

**⚠️ CRITICAL:** In sandboxes/CI, `uv run` downloads gigabytes of CUDA packages. Always use `uv --project ci run` instead.

## Development Workflow

### Branch Strategy

**Note for AI assistants:** You may receive specific branch instructions from the system that override these guidelines. Follow system instructions when provided.

**YOU MUST develop on feature branches** - NEVER commit directly to `main`.

```bash
git checkout -b feature/your-feature   # or fix/, refactor/
# ... make changes ...
./run-checks.sh                        # Verify all checks pass (use --ci in sandboxes)
git add . && git commit -m "message"   # Pre-commit hook runs automatically
git push -u origin your-branch-name
```

**Pre-commit hook** auto-runs: `ruff format`, `ruff check --fix`, `pyright`, `pytest` (blocks commits if any fail).

**Code style:**
- Modern Python typing: `list[str]` not `List[str]`, `str | None` not `Optional[str]`
- All modules start with `from __future__ import annotations`
- Use keyword-only parameters (`*,`) for functions/methods (except external callbacks - see gotcha #6)
- Frozen dataclasses: `@dataclasses.dataclass(frozen=True, kw_only=True)`
- Minimize public API: prefix internal helpers with `_`

### Key Files

- **cli.py** - CLI args (`parse_args()`), stitching (`ChunkCollectorWithStitching`)
- **whisper_backend.py** - VAD (`WebRTCVAD`), transcription loop (`run_whisper_transcriber()`)
- **realtime_backend.py** - WebSocket streaming (`run_realtime_transcriber()`)
- **audio_capture.py** - Microphone (`AudioCaptureManager`)
- **file_audio_source.py** - File/URL simulation (`FileAudioSource`)
- **tests/conftest.py** - CI mocking strategy

## Critical Implementation Rules

### VAD Chunking & Stitching (Whisper Backend)

**YOU MUST understand:** VAD splits audio at natural speech pauses. Chunks are sequential with NO overlap.

**Stitching logic (cli.py:ChunkCollectorWithStitching):**
- Strip trailing `,` and `.` from intermediate chunks (preserve `?` and `!`)
- **Why:** Whisper adds punctuation assuming completeness, but VAD splits mid-sentence
- **Breaking this:** Creates unnatural transcripts like "Hello world. How are you." → "Hello world, how are you."

### Backend Differences

| Feature | Whisper | Realtime |
|---------|---------|----------|
| Chunking | VAD-based (variable) | Fixed 2.0s |
| Punctuation cleanup | Required | Not needed |
| Device | Local GPU/CPU | Cloud API |
| Testing | Avoid strict timing | Safe for timing |

**Consequence:** Never test transcript comparison with Whisper backend (flaky). Use Realtime for transcript tests.

### Critical Defaults (DO NOT change without testing)

- **`language="en"`** - Prevents hallucinations on silence/noise (auto-detection causes issues)
- **`vad_aggressiveness=2`** - Balance speech detection
- **`min_silence_duration=0.2s`** - Controls chunking speed
- **`max_chunk_duration=60s`** - Prevents buffer overflow
- **`model="turbo"`** - Default (requires GPU); **use `base.en` for CPU-only**

**For model selection and VAD tuning guidance**, see **[README.md](README.md)** (user-facing documentation).

### Common Gotchas

1. **CPU vs GPU:** `uv sync` downloads 2.5GB+ CUDA in CI. Always use `uv sync --project ci --refresh` in sandboxes.
2. **Language parameter:** Don't set to `None` or `"auto"` - causes hallucinations on silence.
3. **Punctuation stripping:** Don't modify without understanding VAD chunking implications.
4. **Whisper testing:** VAD makes transcript tests flaky. Use Realtime for transcript comparison.
5. **Realtime chunking:** Fixed 2.0s chunks, NOT configurable (no VAD).
6. **External callback signatures:** `_audio_callback` MUST accept positional args (no `*`). sounddevice.InputStream calls callbacks positionally. See `tests/test_audio_capture.py::test_audio_callback_accepts_positional_args`.
7. **SSL/Certificate issues:** For development/testing with corporate proxies or self-signed certs:
   - Use `--ca_cert /path/to/cert.pem` for custom certificate bundles
   - Use `--disable_ssl_verify` as last resort (insecure, not for production)
   - Affects both model downloads and Realtime API WebSocket connections

## Testing

Pre-commit hook auto-runs all tests - they MUST pass before commits.

### Test Types
- **Unit** (~0.1s): `test_vad.py`, `test_main_utils.py`, `test_file_audio_source.py`
- **Integration** (2-3s): `test_backend_time_limits.py`

### Critical Testing Rules

**VAD + Time Limits = Flaky Tests**
- VAD chunking is unpredictable - never combine with strict time assertions
- Test VAD behavior separately from timing
- Use Realtime backend for transcript comparison tests

**Writing Fast Tests:**
- Use synthetic audio: `_generate_synthetic_audio(duration_seconds=2.0)`
- Keep `max_chunk_duration` high (≥10s) to prevent VAD splits
- Use high `playback_speed` (10.0x) for FileAudioSource
- Disable extras: `save_chunk_audio=False`
- Never add `time.sleep()` in test infrastructure
- Run 3-5 times to verify non-flakiness

**URL testing:** Always mock `urllib.request.urlopen` (see `test_file_audio_source.py::test_file_audio_source_url_detection`)

### CI Testing (No Audio Hardware)

**Mocking strategy (tests/conftest.py):**
1. **sounddevice:** Module-level mock (pytest imports during collection before fixtures)
2. **stdin.isatty():** Autouse fixture returns `False` (prevents blocking listener threads)
3. **AudioCaptureManager:** Monkeypatch with STRING paths: `monkeypatch.setattr("transcribe_demo.audio_capture.AudioCaptureManager", FakeCls)`

**FakeAudioCaptureManager requirements:**
- Set `stop_event` after feeding audio
- Respect `max_capture_duration` with `capture_limit_reached` flag
- Add 1ms delay after limit for backend processing time

## Related Documentation

See **[SITEMAP.md](SITEMAP.md)** for complete documentation guide.

**Quick links:** [DESIGN.md](DESIGN.md) (architecture), [README.md](README.md) (user docs), [TODO.md](TODO.md) (improvements), [SESSIONS.md](SESSIONS.md) (sessions)

---

*Last Updated: 2025-11-15*
