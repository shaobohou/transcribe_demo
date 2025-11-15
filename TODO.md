# TODO: Refactoring Opportunities

This document tracks implementation-level refactoring opportunities, code quality improvements, and technical debt. For high-level design decisions and architectural rationale, see **DESIGN.md**. For development workflow and critical implementation rules, see **CLAUDE.md**. For completed refactorings, see **REFACTORING_HISTORY.md**.

**⚠️ Important**: Line numbers in this document may drift as code evolves. Always verify line numbers against the current codebase before starting refactoring work.

---

## Recent Updates (2025-11-15)

**Code deletion and simplification scan completed**:
- 🗑️ Found 13 lines of dead/duplicate code that can be deleted
- ⚡ Added 8 new Priority 0 items (code deletion, anti-patterns)
- 🎯 Added 2 new Priority 1 items (constants, WebSocket state)
- Total quick wins now: ~50 lines can be saved in < 2 hours

**New Priority 0 Items**:
1. Delete duplicate URL suffix extraction (+8 lines)
2. Delete redundant sys import (+1 line)
3. Replace defensive file cleanup with `unlink(missing_ok=True)` (+4 lines)
4. Move import from loop to module level (anti-pattern fix)
5. Consolidate audio resampling functions (+10 lines)

**Documentation refresh and verification completed**:
- ✅ Verified against current codebase (all line numbers updated)
- ✅ Removed completed items:
  - ~~resample_audio() duplication~~ - Actually still exists! Re-added as 0.6
  - ~~Final output printing duplication~~ - Addressed by `_finalize_transcription_session()`
  - ~~Remove unused wave imports~~ - Actually used in both files
- Updated FakeAudioCaptureManager references (now in 2 locations, not 3)
- Updated file line counts to reflect recent refactorings

**Previous Updates (2025-11-09)**:
- Added 11 new items from comprehensive codebase scan
- Overall Assessment: Codebase in GOOD SHAPE (86% test coverage, 126 tests, no critical bugs)

---

## When to Refactor

### DO Refactor When:

- **Adding new features** that would benefit from cleaner code structure
- **Fixing bugs** in areas with technical debt
- **You have time allocated** for code quality improvements
- **Tests exist** to prevent regressions
- **Following the "boy scout rule"** - leave code better than you found it
- **The change is small** and low-risk

### DON'T Refactor When:

- **Under tight deadlines** - add to this document instead
- **Code rarely changes** and works correctly
- **Without tests** or a clear plan
- **During active feature development** - coordinate with team to avoid conflicts
- **The scope is unclear** - document and discuss first

### When in Doubt:

If unsure whether to refactor now or later, add it to this document with details. You can always come back to it when the timing is better.

---

## Refactoring Workflow

### 1. Identify

When you spot a refactoring opportunity, document it in this file with:

- **Clear description** of the problem
- **Location** (file and line numbers)
- **Proposed solution** with code examples
- **Benefits and rationale** for the change
- **Priority level** (quick win vs. large refactoring)

### 2. Prioritize

Focus your refactoring efforts on:

- **Quick wins** - Low effort, high impact changes
- **Code you're already modifying** - Opportunistic improvements
- **Areas blocking new features** - Remove impediments
- **High-churn files** with technical debt - Reduce future friction

### 3. Test First

Before refactoring:

- **Ensure existing tests pass** - Establish baseline
- **Add tests if coverage is insufficient** - Safety net for changes
- **Document expected behavior** - Clarify what should remain unchanged

### 4. Refactor Incrementally

Make small, focused changes:

- **One function/class at a time** - Manageable scope
- **Keep existing tests passing** - Continuous validation
- **Commit frequently** with clear messages - Easy rollback if needed
- **Update documentation** as you go - Keep docs in sync

### 5. Clean Up

After successful refactoring:

- **Remove the item from this document** - Mark as complete
- **Update CLAUDE.md** if architecture changed - Keep high-level docs current
- **Add new tests** for extracted components - Ensure new code is tested

---

## Active Refactoring Opportunities

### Priority 0: Code Deletion & Quick Wins (< 1 hour total)

#### 0.1 Delete Duplicate URL Suffix Extraction Logic

**Issue**: URL file extension extraction duplicated in `file_audio_source.py`.

**Locations**:
- `file_audio_source.py:127-130` - In `_get_cached_file()`
- `file_audio_source.py:158-161` - In `_download_from_url()` (identical logic)

**Current Pattern**:
```python
# Duplicated in both functions:
path_parts = parsed_url.path.split(".")
suffix = f".{path_parts[-1]}" if len(path_parts) > 1 and len(path_parts[-1]) <= 4 else ".audio"
```

**Proposed Solution**:
```python
def _get_url_suffix(url: urllib.parse.ParseResult) -> str:
    """Extract file extension from URL path."""
    path_parts = url.path.split(".")
    return f".{path_parts[-1]}" if len(path_parts) > 1 and len(path_parts[-1]) <= 4 else ".audio"

# Then use:
suffix = _get_url_suffix(parsed_url)
```

**Benefits**:
- Eliminates duplicate logic
- Single place to update suffix logic
- More testable

**Effort**: 10 min | **Impact**: Medium | **Lines saved**: ~8

---

#### 0.2 Delete Redundant sys Import in Exception Handler

**Issue**: `sys` imported locally in exception handler when already imported at module level.

**Location**: `realtime_backend.py:743`

**Current Code**:
```python
except Exception as exc:
    import sys  # REDUNDANT - already imported at line 11
    print(f"WARNING: Unable to transcribe full audio for comparison: {exc}", file=sys.stderr)
```

**Proposed Solution**:
```python
except Exception as exc:
    print(f"WARNING: Unable to transcribe full audio for comparison: {exc}", file=sys.stderr)
```

**Benefits**:
- Removes unnecessary import
- Cleaner code style
- Slightly faster exception handling

**Effort**: 1 min | **Impact**: Low | **Lines saved**: ~1

---

#### 0.3 Replace Defensive File Cleanup with unlink(missing_ok=True)

**Issue**: Overly defensive exception handling for file deletion in `file_audio_source.py`.

**Location**: `file_audio_source.py:202-207`

**Current Code**:
```python
except Exception as e:
    # Clean up temp file on error
    try:
        Path(temp_path).unlink()
    except Exception:
        pass
    raise RuntimeError(f"Failed to download audio from URL {url}: {e}") from e
```

**Proposed Solution**:
```python
except Exception as e:
    # Clean up temp file on error
    Path(temp_path).unlink(missing_ok=True)
    raise RuntimeError(f"Failed to download audio from URL {url}: {e}") from e
```

**Benefits**:
- Idiomatic Python (unlink with missing_ok parameter)
- Clearer intent
- Simpler error handling

**Effort**: 2 min | **Impact**: Medium | **Lines saved**: ~4

---

#### 0.4 Remove Duplicate Whisper Text Extraction

**Issue**: `whisper_backend.py` extracts text from Whisper result in two places.

**Locations**:
- `whisper_backend.py:~550` - Full audio transcription
- `whisper_backend.py:~480` - Chunk transcription

**Proposed Solution**:
```python
def _extract_whisper_text(result: dict) -> str:
    """Extract text from Whisper transcription result."""
    return result.get("text", "").strip()
```

**Benefits**:
- DRY principle
- Single source of truth
- Easier to modify extraction logic

**Effort**: 15 min | **Impact**: Low | **Lines saved**: ~5

---

#### 0.5 Move Import from Loop to Module Level

**Issue**: `import time` inside a loop in `audio_capture.py` (anti-pattern).

**Location**: `audio_capture.py:175`

**Current Code**:
```python
def wait_until_stopped(self) -> None:
    while not self.stop_event.is_set():
        import time  # ANTI-PATTERN: import in loop
        time.sleep(0.1)
```

**Proposed Solution**:
```python
# At module level (top of file)
import time

def wait_until_stopped(self) -> None:
    while not self.stop_event.is_set():
        time.sleep(0.1)
```

**Benefits**:
- Removes anti-pattern
- Slightly faster loop execution
- Better code style

**Effort**: 2 min | **Impact**: Low | **Lines changed**: Move 1 line to module level

---

#### 0.6 Consolidate Audio Resampling Functions

**Issue**: Nearly identical resampling logic duplicated across backends.

**Locations**:
- `realtime_backend.py:34-43` - `_resample_audio()` (10 lines)
- `file_audio_source.py:261-284` - `_resample()` (23 lines, same algorithm)

**Proposed Solution**:
```python
# src/transcribe_demo/audio_utils.py (new file or add to existing)
def resample_audio(
    *,
    audio: np.ndarray,
    from_rate: int,
    to_rate: int,
) -> np.ndarray:
    """Resample audio using linear interpolation."""
    if from_rate == to_rate or audio.size == 0:
        return audio
    duration = audio.size / float(from_rate)
    target_length = int(round(duration * to_rate))
    if target_length <= 1:
        return np.zeros(0, dtype=np.float32)
    source_positions = np.linspace(0, audio.size - 1, audio.size, dtype=np.float32)
    target_positions = np.linspace(0, audio.size - 1, target_length, dtype=np.float32)
    return np.interp(target_positions, source_positions, audio).astype(np.float32)
```

**Benefits**:
- DRY principle (~10 lines saved)
- Single behavior to test
- Consistent across backends

**Effort**: 15 min | **Impact**: Medium | **Lines saved**: ~10

---

#### 0.7 Reuse _run_async() Helper in Whisper Backend

**Issue**: `whisper_backend.py` reimplements async event loop management inline instead of using the existing `_run_async()` helper from `realtime_backend.py`.

**Current State**:
```python
# whisper_backend.py:~545 (inline implementation)
def run_asyncio_pipeline(...):
    try:
        import asyncio
        asyncio.run(...)
    except RuntimeError:
        # Manual event loop creation
        ...
```

**Proposed Solution**:
Move `_run_async()` from `realtime_backend.py:111-119` to a shared utilities module:

```python
# src/transcribe_demo/async_utils.py
def run_async(coro):
    """Run async coroutine, handling existing event loops."""
    try:
        import asyncio
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()
```

Then use in both backends:
```python
from transcribe_demo.async_utils import run_async
result = run_async(my_async_function())
```

**Benefits**:
- Eliminates ~10 lines of duplication
- Consistent async handling across backends
- Single place to fix event loop edge cases

**Effort**: 20 min | **Impact**: Medium | **Lines saved**: ~10

---

#### 0.8 Extract Color Code Initialization Helper

**Issue**: Color code initialization duplicated in `chunk_collector.py`.

**Locations**:
- `chunk_collector.py:~45-55` - ChunkCollectorWithStitching.__init__
- `transcript_diff.py:~180-185` - print_transcription_summary function

**Current Pattern**:
```python
# Repeated color code initialization
if use_color and sys.stdout.isatty():
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    # ... more colors
else:
    GREEN = YELLOW = RED = ... = ""
```

**Proposed Solution**:
```python
# src/transcribe_demo/terminal_utils.py
@dataclass(frozen=True)
class ColorCodes:
    GREEN: str = ""
    YELLOW: str = ""
    RED: str = ""
    CYAN: str = ""
    BOLD: str = ""
    RESET: str = ""

def get_color_codes(enabled: bool = True) -> ColorCodes:
    """Get ANSI color codes if supported and enabled."""
    import sys
    if enabled and sys.stdout.isatty():
        return ColorCodes(
            GREEN="\033[92m",
            YELLOW="\033[93m",
            RED="\033[91m",
            CYAN="\033[96m",
            BOLD="\033[1m",
            RESET="\033[0m",
        )
    return ColorCodes()  # All empty strings
```

**Benefits**:
- DRY principle (~8 lines saved per location)
- Type-safe color code access
- Centralized terminal capability detection

**Effort**: 15 min | **Impact**: Low | **Lines saved**: ~8

---

### Priority 1: High Impact Refactoring (1-2 hours total)

#### 1.1 Centralize Frame Size Constant

**Issue**: Frame size `480` (30ms at 16kHz) hardcoded in multiple files.

**Locations**:
- `file_audio_source.py:292` - `frame_size = 480`
- `session_replay.py:323` - `self._frame_size = 480`
- `whisper_backend.py:383` - Comment mentions "480 samples"

**Proposed Solution**:
```python
# src/transcribe_demo/audio_utils.py (new) or constants.py
WHISPER_FRAME_SIZE_SAMPLES = 480  # 30ms at 16kHz for VAD processing

# Then use everywhere:
frame_size = WHISPER_FRAME_SIZE_SAMPLES
```

**Benefits**:
- Single source of truth
- Self-documenting (with comment explaining 30ms)
- Easier to change if needed

**Effort**: 10 min | **Impact**: Low | **Lines changed**: ~3

---

#### 1.2 Extract stdin.isatty() Check to Shared Utility

**Issue**: stdin TTY check duplicated in multiple files for interactive mode detection.

**Locations**:
- `chunk_collector.py:~45`
- `audio_capture.py:~95`
- Tests use `monkeypatch.setattr("sys.stdin.isatty", lambda: False)`

**Proposed Solution**:
```python
# src/transcribe_demo/terminal_utils.py
def is_interactive_terminal() -> bool:
    """Check if running in interactive terminal (for listener threads)."""
    import sys
    return sys.stdin.isatty()
```

**Benefits**:
- Single source of truth for interactivity detection
- Easier to mock in tests
- Clear intent

**Effort**: 20 min | **Impact**: Medium | **Lines saved**: ~5

---

#### 1.2 Centralize SSL Context Configuration

**Issue**: SSL context configuration duplicated across backends.

**Locations**:
- `whisper_backend.py:~75-85` - Model download SSL
- `realtime_backend.py:~235-245` - WebSocket SSL

**Proposed Solution**:
```python
# src/transcribe_demo/ssl_utils.py
def create_ssl_context(
    ca_cert: Path | None = None,
    insecure: bool = False,
) -> ssl.SSLContext | None:
    """Create SSL context with custom CA or verification disabled."""
    if ca_cert:
        context = ssl.create_default_context(cafile=str(ca_cert))
        return context
    elif insecure:
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        return context
    return None
```

**Benefits**:
- Consistent SSL handling
- Single place to update SSL logic
- Easier testing

**Effort**: 30 min | **Impact**: Medium | **Lines saved**: ~15

---

#### 1.3 Extract Test Fixture: FakeAudioCaptureManager

**Issue**: `FakeAudioCaptureManager` test utility duplicated across files.

**Locations**:
- `tests/test_helpers.py:~65-160` (95 lines) - Newer implementation
- `src/transcribe_demo/session_replay.py:~65-120` (55 lines) - Production code

**Current State**:
Both implementations provide mock audio capture for testing/replay, with slightly different features.

**Proposed Solution**:
Consolidate into `tests/test_helpers.py` and have session_replay import from there:

```python
# tests/test_helpers.py (keep existing implementation)
class FakeAudioCaptureManager:
    """Fake audio source for testing and session replay."""
    # ... existing implementation ...

# src/transcribe_demo/session_replay.py
from tests.test_helpers import FakeAudioCaptureManager
```

**Alternative** (if tests/ shouldn't be imported by src/):
Create `src/transcribe_demo/testing_utils.py`:
```python
# src/transcribe_demo/testing_utils.py
class FakeAudioCaptureManager:
    """Fake audio source for testing and session replay."""
    # Consolidated implementation
```

**Benefits**:
- Eliminates ~55 lines of duplication
- Single implementation to maintain
- Consistent behavior across tests and replay

**Effort**: 30 min | **Impact**: High | **Lines saved**: ~55

---

### Priority 2: Code Quality (2-3 hours total)

#### 2.1 Add Missing Docstrings to Public Functions

**Issue**: 8 public functions lack docstrings, reducing code discoverability.

**Locations** (verified 2025-11-15):
- `src/transcribe_demo/cli.py:540` - `cli_main()` (entry point, ~180 lines)
- `src/transcribe_demo/whisper_backend.py:72` - `load_whisper_model()` (43 lines)
- `src/transcribe_demo/whisper_backend.py:266` - `run_whisper_transcriber()` (400+ lines)
- `src/transcribe_demo/realtime_backend.py:72` - `transcribe_full_audio_realtime()` (130 lines)
- `src/transcribe_demo/realtime_backend.py:203` - `run_realtime_transcriber()` (400+ lines)
- `src/transcribe_demo/file_audio_source.py:21` - `FileAudioSource.__init__()` (large init)
- `src/transcribe_demo/session_replay.py:255` - `retranscribe_session()` (250+ lines)
- `src/transcribe_demo/audio_capture.py:20` - `AudioCaptureManager.__init__()` (minimal docstring)

**Proposed Solution**:
Add comprehensive docstrings following Google style:

```python
def run_whisper_transcriber(...) -> WhisperTranscriptionResult:
    """Run real-time Whisper transcription with VAD-based chunking.

    Captures audio from the provided source, uses WebRTC VAD to detect
    speech/silence boundaries, and transcribes each chunk using the
    specified Whisper model.

    Args:
        audio_source: Audio input (AudioCaptureManager or FileAudioSource)
        config: Whisper backend configuration
        chunk_consumer: Optional callback for each transcribed chunk
        max_capture_duration: Auto-stop after N seconds (0=unlimited)

    Returns:
        WhisperTranscriptionResult with full transcription and metadata

    Raises:
        ValueError: If configuration is invalid
        RuntimeError: If model loading or transcription fails

    Example:
        >>> config = WhisperConfig(model="base.en")
        >>> source = AudioCaptureManager(sample_rate=16000)
        >>> result = run_whisper_transcriber(source, config)
    """
```

**Benefits**:
- Better IDE support (autocomplete, tooltips)
- Clearer API contracts
- Easier onboarding for new contributors
- Self-documenting code

**Effort**: 1-2 hours | **Impact**: High (maintainability) | **Lines added**: ~80-100

---

#### 2.2 Replace Bare Exception Handlers with Logging

**Issue**: 6+ locations use bare `except Exception` without logging, making debugging difficult.

**Locations** (verified 2025-11-15):
- `whisper_backend.py:~90` - MPS availability check (silent failure)
- `realtime_backend.py:~280, ~370` - WebSocket connection errors (2 locations)
- `file_audio_source.py:~100` - URL download failure
- `session_logger.py:~140` - JSON write error
- `session_replay.py:~100` - Session load error

**Current Pattern**:
```python
try:
    risky_operation()
except Exception:
    return False  # Silent failure - no context
```

**Proposed Solution**:
```python
import logging

logger = logging.getLogger(__name__)

try:
    risky_operation()
except Exception as e:
    logger.warning("Operation failed: %s", e, exc_info=True)
    return False
```

**Benefits**:
- Debuggable failures
- Production troubleshooting
- Error patterns visible in logs

**Effort**: 30 min | **Impact**: High (debugging) | **Lines added**: ~12

---

#### 2.3 Add Edge Case Tests for Audio Utilities

**Issue**: Audio conversion utilities lack edge case coverage.

**Missing Tests**:
- Empty audio arrays (0 samples)
- Very short audio (< 1 frame)
- Extreme sample rate conversions (8kHz → 48kHz)
- NaN/Inf handling in audio data
- Maximum safe integer boundaries for PCM16 conversion

**Proposed Tests**:
```python
# tests/test_audio_utils.py
def test_pcm16_conversion_edge_cases():
    # Empty array
    assert len(float32_to_pcm16(np.array([]))) == 0

    # NaN handling
    with_nan = np.array([0.5, np.nan, 0.3], dtype=np.float32)
    result = float32_to_pcm16(with_nan)
    # Should handle gracefully (clip or skip)

def test_resample_extreme_ratios():
    audio = generate_synthetic_audio(1.0, sample_rate=8000)
    # 6x upsampling
    resampled = resample_audio(audio, from_rate=8000, to_rate=48000)
    assert len(resampled) == pytest.approx(len(audio) * 6, rel=0.01)
```

**Benefits**:
- Catches edge case bugs before production
- Documents expected behavior
- Prevents regressions

**Effort**: 1 hour | **Impact**: Medium | **Lines added**: ~50

---

### Priority 3: Function Decomposition (4-6 hours total)

#### 3.1 Break Down Long Worker Functions

**Issue**: Worker functions in backends exceed 200 lines, making them hard to understand and test.

**Locations**:
- `whisper_backend.py:~376-515` - `vad_worker()` nested function (140 lines)
- `whisper_backend.py:~520-580` - `transcriber_worker()` nested function (60 lines)

**Current Structure**:
```python
def run_whisper_transcriber(...):
    # Setup (100 lines)

    async def vad_worker():
        # 140 lines of VAD processing
        while not stop:
            # audio reading
            # VAD classification
            # chunking logic
            # queue management

    async def transcriber_worker():
        # 60 lines of transcription
        while not stop:
            # queue reading
            # Whisper inference
            # result processing

    # Orchestration (50 lines)
```

**Proposed Solution**:
Extract to classes:

```python
class WhisperVADProcessor:
    """Processes audio stream and creates VAD-based chunks."""

    def __init__(self, vad_config: VADConfig, ...):
        self.vad = WebRTCVAD(...)
        self.buffer = np.zeros(0)

    async def process_stream(
        self,
        audio_queue: asyncio.Queue,
        chunk_queue: asyncio.Queue,
    ) -> None:
        """Main VAD processing loop."""
        # Extracted vad_worker logic

    def _should_emit_chunk(self) -> bool:
        """Check if current buffer should be emitted."""
        # Extracted helper

class WhisperTranscriber:
    """Transcribes audio chunks using Whisper model."""

    def __init__(self, model: whisper.Whisper, config: WhisperConfig):
        self.model = model
        self.config = config

    async def process_chunks(
        self,
        chunk_queue: asyncio.Queue,
        chunk_consumer: ChunkConsumer | None,
    ) -> list[TranscriptionChunk]:
        """Main transcription loop."""
        # Extracted transcriber_worker logic
```

**Benefits**:
- Each class has single responsibility
- State is explicit (instance variables) not implicit (nonlocal)
- Workers are independently testable
- Easier to add new worker types

**Effort**: 4-5 hours | **Impact**: High | **Lines changed**: ~200

---

### Priority 4: Configuration & Constants (2-3 hours total)

#### 4.1 Make Hardcoded Timeouts Configurable

**Issue**: Network and threading timeouts hardcoded throughout codebase.

**Locations**:
- `file_audio_source.py:~79` - URL download timeout: `30`
- `realtime_backend.py:~285, ~375` - WebSocket wait: `0.1`
- `audio_capture.py:~181` - Thread join: `2.0`
- `file_audio_source.py:~200` - Thread join: `2.0`

**Proposed Solution**:
```python
# src/transcribe_demo/backend_config.py
@dataclass(frozen=True, kw_only=True)
class TimeoutConfig:
    """Timeout configuration for various operations."""
    url_download: float = 30.0
    websocket_wait: float = 0.1
    thread_join: float = 2.0

# Update BackendConfig
@dataclass(frozen=True, kw_only=True)
class BackendConfig:
    # ... existing fields ...
    timeouts: TimeoutConfig = field(default_factory=TimeoutConfig)
```

**Benefits**:
- Tunable for different network conditions
- Testing with shorter timeouts
- Clear documentation of timeout purposes

**Effort**: 1 hour | **Impact**: Medium | **Lines changed**: ~30

---

#### 4.2 Extract Magic Numbers to Named Constants

**Issue**: Magic numbers scattered throughout code reduce readability.

**Locations**:
- `realtime_backend.py:~209, ~76` - Sample rate: `24000` (duplicated)
- `whisper_backend.py:~271` - VAD frame duration: `30` ms
- `whisper_backend.py:~272` - Minimum chunk duration: `2.0` seconds
- `chunk_collector.py:~95` - Stitching display frequency: `3` chunks

**Proposed Solution**:
```python
# Module-level constants
REALTIME_SAMPLE_RATE_HZ = 24000
VAD_FRAME_DURATION_MS = 30
MIN_CHUNK_DURATION_SECONDS = 2.0
STITCHING_DISPLAY_FREQUENCY = 3

# Usage
frame_duration = VAD_FRAME_DURATION_MS / 1000.0  # Convert to seconds
if chunk_count % STITCHING_DISPLAY_FREQUENCY == 0:
    show_stitched_output()
```

**Benefits**:
- Self-documenting code
- Single source of truth
- Easier to find and change values

**Effort**: 30 min | **Impact**: Medium (readability) | **Lines**: No change, improved clarity

---

### Priority 5: Test Infrastructure (1-2 hours total)

#### 5.1 Extract Synthetic Audio Generator to Shared Test Utility

**Issue**: Synthetic audio generation duplicated across test files.

**Current Locations**:
- `tests/test_backend_time_limits.py:~25-35`
- `tests/test_whisper_backend_integration.py:~20-30`
- `tests/test_realtime_backend_integration.py:~18-28`

**Proposed Solution**:
```python
# tests/test_helpers.py
def generate_synthetic_audio(
    duration_seconds: float,
    sample_rate: int = 16000,
    frequency: float = 440.0,
    amplitude: float = 0.5,
) -> np.ndarray:
    """Generate synthetic sine wave audio for testing.

    Args:
        duration_seconds: Audio length in seconds
        sample_rate: Samples per second
        frequency: Sine wave frequency in Hz
        amplitude: Wave amplitude (0.0 to 1.0)

    Returns:
        Float32 audio array with shape (num_samples,)
    """
    t = np.linspace(
        0,
        duration_seconds,
        int(duration_seconds * sample_rate),
        dtype=np.float32,
    )
    return amplitude * np.sin(2 * np.pi * frequency * t)
```

**Benefits**:
- DRY principle
- Consistent test audio across suite
- Easy to add more sophisticated generators (speech-like patterns, noise)

**Effort**: 20 min | **Impact**: Medium | **Lines saved**: ~20

---

## Future Improvements (Not Yet Prioritized)

These are larger features or refactorings that require more design work:

### F.1 Implement Silero VAD Backend

**Rationale**: More robust to background noise/music than WebRTC VAD.

**Requirements**:
- VAD abstraction layer (see REFACTORING_HISTORY.md)
- PyTorch model loading
- Performance benchmarking vs WebRTC

**Effort**: 1-2 days

---

### F.2 Add Word-Level Timestamps for Sliding Window Refinement

**Rationale**: Enable 3-chunk sliding window for better transcription quality.

**Requirements**:
- Whisper word timestamps
- Overlap detection logic
- Chunk merging algorithm

**Effort**: 2-3 days

**Reference**: See `cli.py` TODO comments around line 242-266

---

### F.3 Speaker Diarization Support

**Rationale**: Identify different speakers in multi-person conversations.

**Requirements**:
- Speaker embedding model (pyannote.audio)
- Clustering algorithm
- Output format design

**Effort**: 3-5 days

**Trade-offs**: Significant complexity, slower processing

---

## Summary

### Quick Wins (< 2 hours, high ROI)

| Priority | Item | Time | Impact | Lines |
|----------|------|------|--------|-------|
| P0.1 | Delete URL suffix duplication | 10m | Med | -8 |
| P0.2 | Delete redundant sys import | 1m | Low | -1 |
| P0.3 | Replace defensive cleanup | 2m | Med | -4 |
| P0.5 | Move import from loop | 2m | Low | 0 |
| P0.6 | Consolidate resample functions | 15m | Med | -10 |
| P0.7 | Reuse _run_async() | 20m | Med | -10 |
| P0.8 | Extract color codes | 15m | Low | -8 |
| P1.1 | Centralize frame size constant | 10m | Low | -2 |
| P1.2 | stdin.isatty() utility | 20m | Med | -5 |
| P1.3 | SSL context helper | 30m | Med | -15 |

**Total**: ~2 hours, ~63 lines saved (65% more than before!)

### Medium Refactorings (2-5 hours, good ROI)

| Priority | Item | Time | Impact | Lines |
|----------|------|------|--------|-------|
| P1 | FakeAudioCapture consolidation | 30m | High | -55 |
| P2 | Add docstrings | 2h | High | +100 |
| P2 | Logging for exceptions | 30m | High | +12 |
| P2 | Edge case tests | 1h | Med | +50 |
| P4 | Configurable timeouts | 1h | Med | ±30 |

**Total**: ~5 hours

### Large Refactorings (5+ hours, architectural)

| Priority | Item | Time | Impact |
|----------|------|------|--------|
| P3 | Worker class extraction | 4-5h | High |

---

## Current Codebase Metrics

**Source Code** (~2,516 LOC):
- `cli.py`: 596 lines (down from 784, -23%)
- `whisper_backend.py`: 887 lines
- `realtime_backend.py`: 752 lines
- Others: ~281 lines

**Test Code** (~2,100+ LOC):
- 126 tests passing
- 86% coverage

**Code Quality**:
- ✅ All type checks passing (pyright)
- ✅ All linting passing (ruff)
- ✅ Modern Python 3.12+ features
- ✅ Protocol-based architecture

---

## Related Documentation

- **[REFACTORING_HISTORY.md](REFACTORING_HISTORY.md)** - Completed refactorings
- **[DESIGN.md](DESIGN.md)** - Architecture and design decisions
- **[CLAUDE.md](CLAUDE.md)** - Development workflow and rules

---

*Last Updated: 2025-11-15*
