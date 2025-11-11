# TODO: Refactoring Opportunities

This document tracks implementation-level refactoring opportunities, code quality improvements, and technical debt. For high-level design decisions and architectural rationale, see **DESIGN.md**. For development workflow and critical implementation rules, see **CLAUDE.md**.

**⚠️ Important**: Line numbers in this document may drift as code evolves. Always verify line numbers against the current codebase before starting refactoring work.

---

## Recent Updates (2025-11-09)

**Comprehensive codebase scan completed** - Added 11 new items:

**Priority 0 - Code Simplification**:
1. **§0.2**: Reuse `_run_async()` helper in Whisper backend (~10 lines, 20 min)
2. **§0.3**: Extract color code initialization helper (~8 lines, 15 min)

**Priority 1-5 - Refactoring & Quality**:
3. **§1.1**: Deduplicate `resample_audio()` function (duplicate in 2 files)
4. **§1.1b**: Extract `stdin.isatty()` check to shared utility
5. **§3.0**: Make hardcoded timeouts configurable (network, async, thread joins)
6. **§4.3**: Add edge case tests for audio utilities
7. **§5.0**: Add missing docstrings to 8 public functions ⚠️ HIGH IMPACT
8. **§5.6**: Replace 11 bare exception handlers with logging ⚠️ HIGH IMPACT
9. **§5.7**: Remove unused imports (wave module in 2 files)
10. **§5.8**: Move time import to module level (PEP 8 compliance)

**Quick Wins Summary**: ~15 lines can be removed via remaining Priority 0 items (total time: 35 min)

**Overall Assessment**: Codebase is in GOOD SHAPE (86% test coverage, 97 tests, no critical bugs detected). Main opportunities are code deduplication, missing documentation, and silent failures.

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

## Common Cleanup Patterns

### Code Duplication

**Signs:**
- Repeated logic across multiple files or functions
- Similar classes with overlapping functionality
- Duplicated constants or configuration values

**Solutions:**
- Extract to shared function/class
- Create base class or use composition
- Move to constants.py or config module

### Long Functions

**Signs:**
- Functions >100 lines should be evaluated
- Functions >200 lines definitely need decomposition
- Multiple responsibilities in one function
- Deeply nested functions (>3 levels)

**Solutions:**
- Apply Single Responsibility Principle
- Extract to methods or standalone functions
- Create helper classes to encapsulate related logic

### Configuration Issues

**Signs:**
- Magic numbers scattered throughout code
- Many function parameters (>5)
- Hardcoded values that vary by context

**Solutions:**
- Extract to named constants
- Create configuration dataclasses
- Move to config files or CLI arguments

### Testing Gaps

**Signs:**
- Repeated test setup across test files
- Missing test coverage for critical paths
- Hard-to-test code (global state, tight coupling)

**Solutions:**
- Extract to fixtures (conftest.py)
- Add tests to TODO.md testing section
- Refactor for testability (dependency injection, smaller functions)

### Error Handling

**Signs:**
- Generic error messages without context
- Silent failures (caught exceptions not logged)
- Using print() instead of logging

**Solutions:**
- Add context and suggestions to error messages
- Add logging or explicit error handling
- Use proper logging framework with levels

### 5.6 Replace Bare Exception Handlers with Logging

**Issue**: 11 locations use bare `except Exception` without logging, making debugging difficult.

**Locations**:
- `whisper_backend.py:86-87` - MPS availability check (silent failure)
- `realtime_backend.py:271, 358, 362` - WebSocket connection errors (3 locations)
- `file_audio_source.py:96-98` - URL download failure (silent)
- `file_audio_source.py:175-177` - Wave file read error (silent)
- `session_logger.py:134-136` - JSON write error (silent)
- `session_replay.py:95-97` - Session load error (silent)
- Additional locations in async exception handling

**Current Pattern**:
```python
# whisper_backend.py:86-87
try:
    return mps_backend.is_available()
except Exception:
    return False  # Silent failure - no logging
```

**Proposed Solution**:
```python
import logging

logger = logging.getLogger(__name__)

# Add logging to all exception handlers:
try:
    return mps_backend.is_available()
except Exception as e:
    logger.debug("MPS backend check failed: %s", e)
    return False

# For critical errors:
try:
    with open(session_path) as f:
        return json.load(f)
except Exception as e:
    logger.error("Failed to load session from %s: %s", session_path, e)
    raise  # Re-raise if critical
```

**Benefits**:
- Visible error messages help debugging
- Can adjust verbosity with log levels
- Structured error tracking
- Easier to diagnose production issues

**Priority**: High - 1 hour, significantly improves debuggability

---

## Priority 0: Code Simplification & Deletion

These are small, targeted changes that delete or consolidate duplicate code.

### 0.2 Reuse _run_async() Helper in Whisper Backend

**Issue**: Whisper backend duplicates the asyncio error handling logic that's already in `realtime_backend.py:_run_async()`.

**Locations**:
- `whisper_backend.py:541-551` (inline `run_asyncio_pipeline()` function)
- `realtime_backend.py:111-119` (`_run_async()` reusable helper)

**Current Code**:
```python
# whisper_backend.py:541-551 (DUPLICATE):
def run_asyncio_pipeline() -> None:
    try:
        asyncio.run(orchestrate())
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(orchestrate())
        finally:
            loop.close()

run_asyncio_pipeline()

# realtime_backend.py:111-119 (REUSABLE):
def _run_async(coro_factory: Callable[[], Coroutine[Any, Any, str]]) -> str:
    try:
        return asyncio.run(coro_factory())
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro_factory())
        finally:
            loop.close()
```

**Proposed Solution**:
```python
# Option 1: Move _run_async to shared utilities module
# Create: src/transcribe_demo/async_utils.py
from typing import Callable, Coroutine, TypeVar, Any
import asyncio

T = TypeVar('T')

def run_async(coro_factory: Callable[[], Coroutine[Any, Any, T]]) -> T:
    """
    Run async coroutine, handling event loop conflicts.

    Args:
        coro_factory: Function that creates the coroutine

    Returns:
        Result from coroutine

    Note:
        Handles RuntimeError when asyncio.run() fails (e.g., nested event loops).
        Creates new event loop as fallback.
    """
    try:
        return asyncio.run(coro_factory())
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro_factory())
        finally:
            loop.close()

# Usage in whisper_backend.py:
from transcribe_demo.async_utils import run_async

run_async(orchestrate)  # Replace run_asyncio_pipeline()
```

**Benefits**:
- DRY: Single implementation for asyncio error handling
- Consistent error handling across backends
- Removes 10 lines of duplicate code
- Type-safe with generic T return type

**Priority**: Medium - 20 minutes, removes code duplication

---

### 0.3 Extract Color Code Initialization Helper

**Issue**: ANSI color code initialization is duplicated in two locations in main.py.

**Locations**:
- `main.py:321-333` (in `ChunkCollectorWithStitching.__init__`)
- `main.py:428-434` (in `print_transcription_summary`)

**Current Code**:
```python
# Both locations do:
use_color = stream.isatty()
cyan = ""
green = ""
reset = ""
bold = ""
if use_color:
    cyan = "\x1b[36m"
    green = "\x1b[32m"
    reset = "\x1b[0m"
    bold = "\x1b[1m"
```

**Proposed Solution**:
```python
from dataclasses import dataclass
from typing import TextIO

@dataclass
class ColorCodes:
    """ANSI color codes for terminal output."""
    cyan: str
    green: str
    reset: str
    bold: str

    @classmethod
    def for_stream(cls, stream: TextIO) -> ColorCodes:
        """
        Get color codes appropriate for the stream.

        Args:
            stream: Output stream

        Returns:
            ColorCodes with ANSI codes if stream is TTY, empty strings otherwise
        """
        if getattr(stream, "isatty", lambda: False)():
            return cls(
                cyan="\x1b[36m",
                green="\x1b[32m",
                reset="\x1b[0m",
                bold="\x1b[1m",
            )
        return cls(cyan="", green="", reset="", bold="")

# Usage:
colors = ColorCodes.for_stream(sys.stdout)
print(f"{colors.green}Success{colors.reset}")
```

**Benefits**:
- DRY: Single color initialization logic
- Type-safe access to color codes
- Easy to add more colors (red, yellow, etc.)
- Clear intent with dataclass

**Priority**: Low - 15 minutes, removes ~8 lines of duplication

---

## Priority 1: High Impact Refactoring

### 1.1 Deduplicate resample_audio() Function

**Issue**: The exact same `resample_audio()` function is implemented twice with slight variations.

**Locations**:
- `realtime_backend.py:48-73` (25 lines) - Returns np.ndarray
- `file_audio_source.py:261-287` (27 lines) - Returns np.ndarray

**Current Code**:
```python
# Both implementations are nearly identical:
def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio using scipy."""
    if orig_sr == target_sr:
        return audio
    num_samples = int(len(audio) * target_sr / orig_sr)
    return scipy.signal.resample(audio, num_samples).astype(np.float32)
```

**Differences**:
- `file_audio_source.py` version has more detailed docstring
- Otherwise functionally identical

**Proposed Solution**:
```python
# Create new file: src/transcribe_demo/audio_utils.py
"""Audio processing utilities."""

from __future__ import annotations

import numpy as np
import scipy.signal


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    Resample audio to a different sample rate using scipy.

    Args:
        audio: Input audio samples (mono, float32)
        orig_sr: Original sample rate in Hz
        target_sr: Target sample rate in Hz

    Returns:
        Resampled audio at target_sr (mono, float32)

    Note:
        If orig_sr == target_sr, returns the original audio unchanged.
        Uses scipy.signal.resample for high-quality resampling.
    """
    if orig_sr == target_sr:
        return audio

    num_samples = int(len(audio) * target_sr / orig_sr)
    resampled = scipy.signal.resample(audio, num_samples)
    return resampled.astype(np.float32)


def float_to_pcm16(audio: np.ndarray) -> bytes:
    """
    Convert float32 audio to PCM16 bytes.

    Args:
        audio: Float32 audio in range [-1.0, 1.0]

    Returns:
        PCM16-encoded bytes

    Note:
        Clips values to [-1.0, 1.0] before conversion.
    """
    clipped = np.clip(audio, -1.0, 1.0)
    pcm16 = (clipped * 32767).astype(np.int16)
    return pcm16.tobytes()
```

**Usage**:
```python
# In both realtime_backend.py and file_audio_source.py:
from transcribe_demo.audio_utils import resample_audio

# Remove local implementations
```

**Benefits**:
- DRY: Single source of truth for audio resampling
- Easier to test and improve (add caching, different algorithms, etc.)
- Can add other audio utilities (normalization, filtering) to same module
- Consistent behavior across all use cases
- Future-proof: Can switch to faster resampling (e.g., librosa) in one place

**Priority**: High - 30 minutes, eliminates critical code duplication

**Testing**: Add unit tests to new `tests/test_audio_utils.py`:
- Test resampling accuracy
- Test edge cases (empty audio, single sample, very short audio)
- Test no-op case (orig_sr == target_sr)
- Test value range preservation

---

### 1.2 Centralize Terminal Output Formatting (main.py)

**Issue**: ANSI color handling and final-result printing are hand-written in several places, leading to duplicated code and inconsistent styling.

**Locations**:
- `main.py`: chunk labels, stitched banners, final summary for both backends
- `ChunkCollectorWithStitching`: label formatting and `[STITCHED]` announcements

**Proposed Solution**:
1. Introduce a `TerminalFormatter` class that encapsulates color usage (`format_chunk_label`, `format_stitched_label`, `format_warning`, etc.).
2. Provide a helper `_print_final_result()` that reuses the formatter and avoids duplicating the final banner logic in the whisper and realtime branches.

**Benefits**:
- Consistent output styling across the CLI.
- One place to adjust color codes or fall back to plain text.
- Simplifies unit testing of presentation logic.
- Eliminates two large blocks of duplicate `print(..., file=sys.stdout)` code.

---

### 1.1b Extract stdin.isatty() Check to Shared Utility

**Issue**: The `stdin.isatty()` check is duplicated in multiple locations for detecting if the session is interactive.

**Locations**:
- `audio_capture.py:52-54` (capture duration confirmation)
- `audio_capture.py:168-169` (inside __enter__ method, import time statement)
- Similar patterns in multiple places

**Current Code**:
```python
# Repeated pattern:
if sys.stdin.isatty():
    # Interactive session logic
```

**Proposed Solution**:
```python
# src/transcribe_demo/terminal_utils.py
"""Terminal interaction utilities."""

import sys
from typing import TextIO


def is_interactive_session() -> bool:
    """
    Check if running in an interactive terminal session.

    Returns:
        True if stdin is connected to a TTY (interactive), False otherwise
    """
    return sys.stdin.isatty()


def confirm_long_capture(duration_seconds: float, threshold: float = 300.0) -> bool:
    """
    Prompt user to confirm if capture duration exceeds threshold.

    Args:
        duration_seconds: Requested capture duration
        threshold: Duration in seconds that triggers confirmation (default: 5 minutes)

    Returns:
        True if user confirmed or duration below threshold, False if user declined

    Note:
        In non-interactive sessions, always returns True (no confirmation possible)
    """
    if not is_interactive_session() or duration_seconds < threshold:
        return True

    print(
        f"\nWarning: You requested a {duration_seconds}s ({duration_seconds/60:.1f} minute) capture.",
        file=sys.stderr,
    )
    response = input("Continue? [y/N]: ").strip().lower()
    return response in ("y", "yes")
```

**Benefits**:
- DRY: Single implementation for interactive checks
- Easier to test (can mock `sys.stdin.isatty()` in one place)
- Clear naming makes intent obvious
- Can add more terminal utilities (color detection, width, etc.)

**Priority**: Medium - 20 minutes

---

### 1.2b Extract Final Output Printing (main.py)

**Issue**: Final stitched result printing is duplicated in 4 near-identical code blocks.

**Locations**:
- `main.py:642-653` (whisper backend without comparison)
- `main.py:738-751` (realtime backend without comparison)

**Current Code**:
```python
# Whisper path (lines 642-653):
if final:
    use_color = sys.stdout.isatty()
    if use_color:
        green = "\x1b[32m"
        reset = "\x1b[0m"
        bold = "\x1b[1m"
        print(
            f"\n{bold}{green}[FINAL STITCHED]{reset} {final}\n",
            file=sys.stdout,
        )
    else:
        print(f"\n[FINAL STITCHED] {final}\n", file=sys.stdout)

# Realtime path (lines 738-751):
if final:
    use_color = sys.stdout.isatty()
    green = ""
    reset = ""
    bold = ""
    if use_color:
        green = "\x1b[32m"
        reset = "\x1b[0m"
        bold = "\x1b[1m"
        print(
            f"\n{bold}{green}[FINAL STITCHED]{reset} {final}\n",
            file=sys.stdout,
        )
    else:
        print(f"\n[FINAL STITCHED] {final}\n", file=sys.stdout)
```

**Proposed Solution**:
```python
def _print_final_stitched(stream: TextIO, text: str) -> None:
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

# Usage:
_print_final_stitched(sys.stdout, final)
```

**Benefits**:
- DRY: Eliminates 4 code blocks (~40 lines)
- Consistent formatting across both backends
- Easier to modify output format in one place
- More testable

**Priority**: Quick win - 5 minutes, immediate cleanup

---

### 1.3 Extract Device Selection Logic (whisper_backend.py)

**Status**: ⚠️ **PARTIALLY COMPLETED** - Function extracted but not using dataclass pattern

**Issue**: Device detection and selection logic was embedded in `run_whisper_transcriber()`.

**Current State**: Extracted to `load_whisper_model()` function (`whisper_backend.py:54-96`) which returns `tuple[whisper.Whisper, str, bool]`.

**Original Location**: `whisper_backend.py:163-194` (outdated line numbers)

**Proposed Solution**:
```python
@dataclass
class DeviceConfig:
    """Device configuration for Whisper inference."""
    device: str  # 'cuda', 'mps', or 'cpu'
    requires_fp16: bool

    @staticmethod
    def detect_device(
        preference: str = "auto",
        require_gpu: bool = False
    ) -> DeviceConfig:
        """
        Detect and select appropriate device for inference.

        Args:
            preference: Device preference ('auto', 'cuda', 'mps', 'cpu')
            require_gpu: If True, raises error if no GPU available

        Returns:
            DeviceConfig with selected device and fp16 flag

        Raises:
            RuntimeError: If requested device unavailable or GPU required but missing
        """
        cuda_available = torch.cuda.is_available()
        mps_available = _check_mps_available()

        if preference == "auto":
            if cuda_available:
                device = "cuda"
            elif mps_available:
                device = "mps"
            else:
                device = "cpu"
        elif preference == "cuda":
            if not cuda_available:
                raise RuntimeError("CUDA GPU requested but unavailable.")
            device = "cuda"
        elif preference == "mps":
            if not mps_available:
                raise RuntimeError("Apple Metal GPU requested but unavailable.")
            device = "mps"
        else:
            device = "cpu"

        if require_gpu and device not in {"cuda", "mps"}:
            raise RuntimeError("GPU required but none available.")

        return DeviceConfig(
            device=device,
            requires_fp16=(device == "cuda")
        )

def _check_mps_available() -> bool:
    """Check if Apple Metal (MPS) is available."""
    mps_backend = getattr(torch.backends, "mps", None)
    if not mps_backend or not hasattr(mps_backend, "is_available"):
        return False
    try:
        return bool(mps_backend.is_available())
    except Exception:
        return False
```

**Benefits**:
- Separates concerns (device detection vs transcription)
- Easier to test device selection logic
- Reusable in other contexts
- More readable main function

### 1.4 Unify Transcription Session Harness

**Status**: ⚠️ **PARTIALLY COMPLETED** - AudioCaptureManager extracted, but backend-specific logic still duplicated

**Progress**: The `AudioCaptureManager` class (in `audio_capture.py`) now handles:
- Queue + event wiring for `sounddevice.InputStream`
- Capture-duration enforcement and warning logs
- Full-audio accumulation when `--compare_transcripts` is set
- Thread-safe stop coordination

**Remaining Issue**: `whisper_backend.py` and `realtime_backend.py` still each reimplement:

- Final stitched output and optional comparison transcription (handled in `main.py` but duplicated)

The backends only diverge when actually producing text (Whisper VAD loop vs. realtime websocket).

**Opportunity**: Introduce a `TranscriberSession` abstraction that encapsulates the shared harness and lets each backend plug in strategy-specific behavior.

```python
class Backend(Protocol):
    def prepare(self, settings: SessionSettings) -> None: ...
    def start_capture(self, handler: CaptureHandler) -> ContextManager: ...
    def process_frame(self, frame: np.ndarray) -> None: ...
    def finalize(self) -> BackendResult: ...

class TranscriberSession:
    def __init__(self, backend: Backend, settings: SessionSettings):
        self._backend = backend
        self._settings = settings
        # queue, stop events, full_audio_chunks, etc.

    def run(self) -> SessionResult:
        self._backend.prepare(self._settings)
        with self._backend.start_capture(self._handle_frame):
            self._loop_until_stop()
        stitched = self._collector.get_final_stitched()
        comparison = self._maybe_compare_full_audio()
        return SessionResult(stitched=stitched, comparison=comparison, chunks=self._collector.chunks)
```

**Benefits**:
- DRY: shared capture-limit warnings, queue plumbing, and comparison logic live in one place.
- Easier to add new backends (Silero, third-party APIs) without copy/paste.
- Simplifies testing: session harness can be unit-tested once with fake backends; integration tests focus on backend-specific code paths.

**Implementation Sketch**:
1. Extract shared data structures (`SessionSettings`, `SessionMetrics`, `ChunkInfo`) into a new module (e.g., `transcribe_demo/session.py`).
2. Refactor whisper and realtime backends to implement the `Backend` protocol, delegating chunk emission to the shared session.
3. Update `main.py` to construct `TranscriberSession` with the selected backend and reuse the existing `ChunkCollectorWithStitching`.
4. Move post-run printing (`print_transcription_summary`) into session or a helper so both backends use identical output paths.

**Risks**:
- Need to ensure backend-specific threading/async concerns are handled cleanly (session should accommodate both patterns).
- Must preserve existing stderr warnings and log messages; consider injecting hooks so the session can surface backend-specific notices.

### 1.5 Extract SSL Context Configuration (Both Backends)

**Issue**: SSL context setup is duplicated in both backends and within realtime_backend itself.

**Locations**:
- `whisper_backend.py:95-118` (environment variables + SSL context)
- `realtime_backend.py:106-110` (SSL context only, in `transcribe_full_audio_realtime`)
- `realtime_backend.py:253-257` (SSL context only, in `run_realtime_transcriber`, DUPLICATE)

**Proposed Solution**:
```python
# Create new file: ssl_utils.py
from __future__ import annotations

import os
import ssl
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional


@contextmanager
def ssl_context_manager(
    ca_cert: Optional[Path] = None,
    insecure: bool = False
) -> Iterator[None]:
    """
    Context manager for SSL configuration.

    Args:
        ca_cert: Path to custom CA certificate bundle
        insecure: If True, disables SSL verification (not recommended)

    Yields:
        None - modifies global SSL context during execution

    Example:
        with ssl_context_manager(ca_cert=Path("/etc/certs/bundle.pem")):
            # SSL configuration active here
            model = whisper.load_model("turbo")
    """
    # Store original state
    original_ssl_cert = os.environ.get("SSL_CERT_FILE")
    original_requests_ca = os.environ.get("REQUESTS_CA_BUNDLE")
    original_https_context = None

    try:
        # Apply custom CA cert
        if ca_cert is not None:
            if not ca_cert.exists():
                raise FileNotFoundError(f"CA bundle not found: {ca_cert}")
            os.environ["SSL_CERT_FILE"] = str(ca_cert)
            os.environ["REQUESTS_CA_BUNDLE"] = str(ca_cert)

        # Apply insecure mode
        if insecure:
            original_https_context = ssl._create_default_https_context
            ssl._create_default_https_context = ssl._create_unverified_context

        yield

    finally:
        # Restore original state
        if original_ssl_cert is not None:
            os.environ["SSL_CERT_FILE"] = original_ssl_cert
        elif "SSL_CERT_FILE" in os.environ:
            del os.environ["SSL_CERT_FILE"]

        if original_requests_ca is not None:
            os.environ["REQUESTS_CA_BUNDLE"] = original_requests_ca
        elif "REQUESTS_CA_BUNDLE" in os.environ:
            del os.environ["REQUESTS_CA_BUNDLE"]

        if original_https_context is not None:
            ssl._create_default_https_context = original_https_context


def create_websocket_ssl_context(insecure: bool = False) -> Optional[ssl.SSLContext]:
    """
    Create SSL context for WebSocket connections.

    Args:
        insecure: If True, disables certificate verification

    Returns:
        SSLContext if insecure mode, None otherwise (use defaults)
    """
    if not insecure:
        return None

    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    return context
```

**Benefits**:
- Eliminates duplication
- Proper cleanup with context manager
- Safer state management
- Reusable across backends

---

## Priority 2: Function Decomposition

### 2.1 Break Down 200-Line `worker()` Function (whisper_backend.py)

**Issue**: The worker function is ~200 lines long with multiple responsibilities, making it the most complex function in the codebase.

**Location**: `whisper_backend.py:300-500` (nested inside `run_whisper_transcriber`)

**Responsibilities**:
- Audio queue processing
- VAD frame processing
- Buffer management
- Speech/silence detection logic
- Chunk transcription triggering
- Punctuation cleanup
- Output formatting

**Proposed Solution**:
```python
class WhisperTranscriptionWorker:
    """Manages VAD-based audio chunking and transcription."""

    def __init__(self, model, sample_rate, vad_config, ...):
        self.model = model
        self.sample_rate = sample_rate
        self.vad = WebRTCVAD(sample_rate, ...)
        self.buffer = np.zeros(0, dtype=np.float32)
        self.chunk_index = 0
        # ... initialize all state

    def process_audio_chunk(self, mono: np.ndarray) -> None:
        """Process incoming audio and update buffers."""
        # Check capture limit
        # Update buffers
        # Process VAD frames
        # ~30 lines

    def should_transcribe(self) -> bool:
        """Determine if buffer is ready for transcription."""
        # Check min/max chunk size
        # Check VAD silence threshold
        # ~15 lines

    def transcribe_buffer(self) -> TranscriptionChunk:
        """Transcribe the current audio buffer."""
        # Run Whisper inference
        # Calculate timestamps
        # ~30 lines

    def run(self, audio_queue, stop_event) -> None:
        """Main worker loop."""
        while not stop_event.is_set():
            mono = self._get_audio(audio_queue)
            if mono is None:
                continue

            self.process_audio_chunk(mono)

            if self.should_transcribe():
                chunk = self.transcribe_buffer()
                self.output_chunk(chunk)

            if self.capture_limit_reached():
                break
```

**Benefits**:
- Each method has single responsibility
- Testable components (can test VAD logic independently)
- Clearer control flow
- Easier to add features (e.g., different chunking strategies)
- Reduced cyclomatic complexity

**Priority**: High impact - 4-6 hours, improves maintainability significantly

**Status**: Most impactful refactoring opportunity

---

### 2.2 Decompose `run_whisper_transcriber()` (whisper_backend.py)

**Issue**: Function is ~339 lines with multiple responsibilities.

**Location**: `whisper_backend.py:244-582`

**Responsibilities**:
1. Device detection and setup
2. Model loading with SSL configuration
3. Audio stream setup
4. VAD configuration
5. Worker thread implementation
6. Main loop coordination

**Proposed Solution**:
```python
@dataclass
class WhisperConfig:
    """Configuration for Whisper transcription."""
    model_name: str
    sample_rate: int
    channels: int
    device_config: DeviceConfig
    vad_config: VADConfig
    temp_file: Optional[Path] = None


@dataclass
class VADConfig:
    """Voice Activity Detection configuration."""
    aggressiveness: int = 2
    min_silence_duration: float = 0.2
    min_speech_duration: float = 0.25
    speech_pad_duration: float = 0.2
    max_chunk_duration: float = 60.0

    @property
    def min_chunk_samples(self) -> int:
        return int(self.sample_rate * 2.0)  # 2 seconds minimum

    @property
    def max_chunk_samples(self) -> int:
        return int(self.sample_rate * self.max_chunk_duration)


class WhisperTranscriber:
    """Handles Whisper model transcription with VAD-based chunking."""

    def __init__(self, config: WhisperConfig):
        self.config = config
        self.model = None
        self.session_start_time = 0.0

    def load_model(self, ca_cert: Optional[Path], insecure: bool):
        """Load Whisper model with SSL configuration."""
        with ssl_context_manager(ca_cert, insecure):
            preferred_models = {"tiny": "tiny.en", "base": "base.en"}
            effective_name = preferred_models.get(
                self.config.model_name,
                self.config.model_name
            )
            self.model = whisper.load_model(
                effective_name,
                device=self.config.device_config.device
            )

    def transcribe_chunk(self, audio: np.ndarray) -> str:
        """Transcribe a single audio chunk."""
        result = self.model.transcribe(
            audio,
            fp16=self.config.device_config.requires_fp16,
            temperature=0.0,
            beam_size=1,
            best_of=1,
            language="en",
        )
        return result["text"]

    def run(
        self,
        ca_cert: Optional[Path],
        insecure: bool,
        chunk_consumer: Optional[Callable],
    ):
        """Run the transcription loop."""
        self.load_model(ca_cert, insecure)
        self.session_start_time = time.perf_counter()

        audio_processor = AudioProcessor(
            sample_rate=self.config.sample_rate,
            vad_config=self.config.vad_config,
        )

        transcription_worker = TranscriptionWorker(
            transcriber=self,
            processor=audio_processor,
            chunk_consumer=chunk_consumer,
        )

        with audio_capture(
            sample_rate=self.config.sample_rate,
            channels=self.config.channels,
            callback=audio_processor.on_audio,
        ):
            transcription_worker.run()


class AudioProcessor:
    """Processes audio with VAD-based chunking."""

    def __init__(self, sample_rate: int, vad_config: VADConfig):
        self.sample_rate = sample_rate
        self.vad_config = vad_config
        self.vad = WebRTCVAD(
            sample_rate=sample_rate,
            frame_duration_ms=30,
            aggressiveness=vad_config.aggressiveness
        )
        self.audio_queue: queue.Queue[np.ndarray] = queue.Queue()

    def on_audio(self, indata: np.ndarray, frames: int, time, status):
        """Callback for audio stream."""
        if status:
            print(f"InputStream status: {status}", file=sys.stderr)
        self.audio_queue.put(indata.copy())


class TranscriptionWorker:
    """Worker that processes audio queue and performs transcription."""

    def __init__(
        self,
        transcriber: WhisperTranscriber,
        processor: AudioProcessor,
        chunk_consumer: Optional[Callable],
    ):
        self.transcriber = transcriber
        self.processor = processor
        self.chunk_consumer = chunk_consumer
        self.buffer = np.zeros(0, dtype=np.float32)
        self.chunk_index = 0
        self.next_chunk_start = 0.0
        # ... other state

    def run(self):
        """Main processing loop."""
        # Move worker logic here
        pass
```

**Benefits**:
- Each class has a single responsibility
- Easier to test individual components
- Configuration is explicit and type-safe
- Worker can be tested without audio stream

---

### 2.2 Decompose ChunkCollectorWithStitching.__call__() (main.py)

**Issue**: The `__call__` method is 93 lines with multiple responsibilities.

**Location**: `main.py:46-138`

**Proposed Solution**:
```python
class ChunkCollectorWithStitching:
    """Collects and displays transcription chunks."""

    def __init__(self, stream: TextIO) -> None:
        self._stream = stream
        self._last_time = float("-inf")
        self._chunks: list[TranscriptionChunk] = []
        self._formatter = TerminalFormatter(
            use_color=getattr(stream, "isatty", lambda: False)()
        )
        self._stitch_frequency = 3  # Show stitched every N chunks

    def __call__(
        self,
        chunk_index: int,
        text: str,
        absolute_start: float,
        absolute_end: float,
        inference_seconds: Optional[float] = None,
    ) -> None:
        if not text:
            return

        chunk = self._create_chunk(
            chunk_index, text, absolute_start, absolute_end, inference_seconds
        )
        self._chunks.append(chunk)

        self._display_chunk(chunk)

        if self._should_show_stitched(chunk_index):
            self._display_stitched()

    def _create_chunk(
        self, index: int, text: str, start: float, end: float, inference: Optional[float]
    ) -> TranscriptionChunk:
        """Create a transcription chunk."""
        return TranscriptionChunk(
            index=index,
            text=text,
            start_time=start,
            end_time=end,
            overlap_start=max(0.0, self._last_time),
            inference_seconds=inference,
        )

    def _format_label(
        self, chunk_index: int, absolute_end: float, inference_seconds: Optional[float]
    ) -> str:
        """Format the chunk label based on mode (whisper or realtime)."""
        if inference_seconds is not None:
            # Whisper mode: show audio duration and inference time
            audio_duration = absolute_end - self._chunks[-1].start_time
            timing = (
                f" | t={absolute_end:.2f}s"
                f" | audio: {audio_duration:.2f}s"
                f" | inference: {inference_seconds:.2f}s"
            )
        else:
            # Realtime mode: show absolute timestamp only
            timing = f" | t={absolute_end:.2f}s"

        return f"[chunk {chunk_index:03d}{timing}]"

    def _display_chunk(self, chunk: TranscriptionChunk) -> None:
        """Display a single chunk."""
        label = self._format_label(
            chunk.index, chunk.end_time, chunk.inference_seconds
        )
        formatted_label = self._formatter.format_chunk_label(label)
        line = f"{formatted_label} {chunk.text.strip()}"

        self._stream.write(line + "\n")
        self._stream.flush()
        self._last_time = max(self._last_time, chunk.end_time)

    def _should_show_stitched(self, chunk_index: int) -> bool:
        """Check if we should display stitched result."""
        return (chunk_index + 1) % self._stitch_frequency == 0

    def _display_stitched(self) -> None:
        """Display stitched result of all chunks so far."""
        cleaned_chunks = [
            self._clean_chunk_text(c.text, is_final_chunk=(i == len(self._chunks) - 1))
            for i, c in enumerate(self._chunks)
        ]
        stitched = " ".join(chunk for chunk in cleaned_chunks if chunk)

        label = self._formatter.format_stitched_label("[STITCHED]")
        self._stream.write(f"\n{label} {stitched}\n\n")
        self._stream.flush()
```

**Benefits**:
- Each method has a single clear purpose
- Easier to test individual formatting logic
- More maintainable
- Reduced cyclomatic complexity

---

### 2.3 Extract Transcript Comparison Logic to Separate Module (main.py)

**Issue**: main.py contains 150+ lines of transcript comparison logic that could be its own module.

**Location**: `main.py:294-448`

**Functions to Extract**:
- `_normalize_whitespace()` - 2 lines
- `print_transcription_summary()` - 65 lines
- `_tokenize_with_original()` - 13 lines
- `_colorize_token()` - 4 lines
- `_format_diff_snippet()` - 40 lines
- `_generate_diff_snippets()` - 15 lines

**Proposed Solution**:
```python
# Create new file: src/transcribe_demo/comparison.py

class TranscriptComparator:
    """Compares and displays differences between transcripts."""

    def __init__(self, use_color: bool = True):
        self.use_color = use_color

    def compare(self, stitched: str, full_audio: str) -> ComparisonResult:
        """
        Compare two transcripts and return structured result.

        Returns:
            ComparisonResult with similarity score and diff details
        """
        stitched_tokens = self._tokenize(stitched)
        full_tokens = self._tokenize(full_audio)

        matcher = difflib.SequenceMatcher(None, stitched_tokens, full_tokens)
        similarity = matcher.ratio()

        return ComparisonResult(
            similarity=similarity,
            diffs=self._generate_diffs(matcher, stitched, full_audio),
        )

    def print_summary(
        self,
        stream: TextIO,
        stitched: str,
        full_audio: str,
    ) -> None:
        """Print formatted comparison to stream."""
        # Format and print transcripts
        # Show similarity and diffs

# Usage in main.py:
comparator = TranscriptComparator(use_color=sys.stdout.isatty())
comparator.print_summary(sys.stdout, final, full_audio_text)
```

**Benefits**:
- Cleaner main.py (150 lines moved)
- Reusable comparison logic
- Easier to test diff algorithms
- Could add other comparison methods (WER, CER, etc.)
- Clear separation of concerns

**Priority**: Medium impact - 2-3 hours

---

### 2.4 Decompose `run_realtime_transcriber()` (realtime_backend.py)

**Issue**: Function is ~200 lines with 3 nested async functions.

**Location**: `realtime_backend.py:193-410`

**Proposed Solution**:
```python
class RealtimeTranscriber:
    """Manages OpenAI Realtime API transcription."""

    def __init__(
        self,
        api_key: str,
        endpoint: str,
        model: str,
        sample_rate: int,
        channels: int,
        chunk_duration: float,
        instructions: str,
        insecure: bool = False,
    ):
        self.api_key = api_key
        self.endpoint = endpoint
        self.model = model
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_duration = chunk_duration
        self.instructions = instructions
        self.insecure = insecure

        self.session_sample_rate = 24000
        self.audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self.stop_event = threading.Event()

    def run(self, chunk_consumer: Optional[Callable]) -> None:
        """Run the realtime transcription."""
        with self._audio_stream():
            asyncio.run(self._async_runtime(chunk_consumer))

    @contextmanager
    def _audio_stream(self):
        """Context manager for audio input stream."""
        with sd.InputStream(
            callback=self._audio_callback,
            channels=self.channels,
            samplerate=self.sample_rate,
            dtype="float32",
        ):
            yield

    def _audio_callback(self, indata: np.ndarray, frames: int, time, status):
        """Callback for audio stream."""
        if status:
            print(f"InputStream status: {status}", file=sys.stderr)
        if not self.stop_event.is_set():
            self.audio_queue.put(indata.copy())

    async def _async_runtime(self, chunk_consumer: Optional[Callable]):
        """Main async runtime."""
        uri = f"{self.endpoint}?model={self.model}"
        headers = [
            ("Authorization", f"Bearer {self.api_key}"),
            ("OpenAI-Beta", "realtime=v1"),
        ]

        ssl_context = create_websocket_ssl_context(self.insecure)

        async with websockets.connect(
            uri, additional_headers=headers, max_size=None, ssl=ssl_context
        ) as ws:
            lock = asyncio.Lock()

            await self._configure_session(ws, lock)

            tasks = [
                asyncio.create_task(AudioSender(self).run(ws, lock)),
                asyncio.create_task(TranscriptReceiver(self).run(ws, chunk_consumer)),
                asyncio.create_task(self._wait_for_stop()),
            ]

            done, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED
            )

            self.stop_event.set()
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

            for task in done:
                if task.exception():
                    raise task.exception()


class AudioSender:
    """Handles sending audio to the Realtime API."""

    def __init__(self, transcriber: RealtimeTranscriber):
        self.transcriber = transcriber

    async def run(
        self,
        ws: websockets.WebSocketClientProtocol,
        lock: asyncio.Lock
    ):
        """Send audio chunks to the API."""
        buffer = np.zeros(0, dtype=np.float32)
        chunk_size = max(
            int(self.transcriber.sample_rate * self.transcriber.chunk_duration),
            1
        )

        while not self.transcriber.stop_event.is_set():
            # ... audio sending logic
            pass


class TranscriptReceiver:
    """Handles receiving transcripts from the Realtime API."""

    def __init__(self, transcriber: RealtimeTranscriber):
        self.transcriber = transcriber
        self.chunk_counter = 0
        self.session_start_time = time.perf_counter()
        self.cumulative_time = 0.0
        self.partials: dict[str, str] = {}

    async def run(
        self,
        ws: websockets.WebSocketClientProtocol,
        chunk_consumer: Optional[Callable],
    ):
        """Receive and process transcripts."""
        try:
            async for message in ws:
                await self._handle_message(message, chunk_consumer)
        except websockets.ConnectionClosed:
            self.transcriber.stop_event.set()

    async def _handle_message(
        self, message: str, chunk_consumer: Optional[Callable]
    ):
        """Handle a single WebSocket message."""
        payload = json.loads(message)
        event_type = payload.get("type")

        if event_type == "conversation.item.input_audio_transcription.delta":
            self._handle_delta(payload)
        elif event_type == "conversation.item.input_audio_transcription.completed":
            self._handle_completed(payload, chunk_consumer)
        elif event_type == "error" or event_type == "error.session":
            self._handle_error(payload, event_type)
```

**Benefits**:
- Separation of concerns (sending vs receiving vs control)
- Each class is testable independently
- Clearer control flow
- Easier to modify individual components

---

### 2.5 Other Large Functions/Classes Requiring Attention

**Scan Results** (2025-11-07): Several additional large functions/classes identified:

1. **`SessionLogger` class** (`session_logger.py:67`): 348 lines
   - Status: ✅ Already well-structured with clear methods
   - No immediate action needed (previously completed refactoring)

2. **`retranscribe_session()` function** (`session_replay.py:238`): 253 lines
   - Handles loading, retranscribing, and saving sessions
   - Could benefit from extraction of save logic to helper function
   - Priority: Low (complex but linear flow)

3. **`main()` function** (`main.py:514`): 240 lines
   - CLI orchestration and backend selection
   - Could extract backend initialization to separate functions
   - Priority: Medium (affects readability but not frequently modified)

4. **`transcribe_full_audio_realtime()` function** (`realtime_backend.py:77`): 132 lines
   - Async transcription of full audio
   - Could extract message handling logic
   - Priority: Low (mostly async boilerplate)

5. **`ChunkCollectorWithStitching` class** (`main.py:193`): 149 lines
   - Chunk collection and display
   - Related to item 2.2 (duplicate section number - needs fixing)
   - Priority: Already documented

6. **`load_whisper_model()` function** (`whisper_backend.py:54`): 121 lines
   - Device detection + model loading
   - Related to item 1.3 (partially completed)
   - Priority: Already documented

**Note**: Functions over 100 lines aren't necessarily problematic if they have linear flow and single responsibility. Priority should focus on functions with high cyclomatic complexity or multiple responsibilities.

---

## Priority 3: Configuration and Constants

### 3.0 Make Hardcoded Timeouts Configurable

**Issue**: Multiple timeout values are hardcoded throughout the codebase, making it difficult to adjust for different network conditions or debugging.

**Locations**:
- `file_audio_source.py:79` - URL download timeout: `30` seconds (hardcoded)
- `realtime_backend.py:275, 363` - WebSocket wait timeout: `0.1` seconds (hardcoded, used in multiple async loops)
- `audio_capture.py:181` - Thread join timeout: `2.0` seconds (hardcoded)
- `file_audio_source.py:200` - Thread join timeout: `2.0` seconds (hardcoded)
- Various other async polling intervals

**Current Code**:
```python
# file_audio_source.py:79
with urllib.request.urlopen(self._audio_file, timeout=30) as response:
    # Hard to adjust for slow connections

# realtime_backend.py:275
await asyncio.wait_for(
    self._audio_queue.get(),
    timeout=0.1  # Hardcoded polling interval
)
```

**Proposed Solution**:
```python
# constants.py (add to existing file or create new)
"""Timeout and performance tuning constants."""

# Network timeouts
URL_DOWNLOAD_TIMEOUT_SECONDS = 30.0
WEBSOCKET_POLL_TIMEOUT_SECONDS = 0.1
WEBSOCKET_CONNECT_TIMEOUT_SECONDS = 10.0

# Thread coordination
THREAD_JOIN_TIMEOUT_SECONDS = 2.0
ASYNC_SHUTDOWN_TIMEOUT_SECONDS = 5.0

# Audio streaming
AUDIO_QUEUE_GET_TIMEOUT_SECONDS = 0.1
```

**Optional CLI Flags** (for advanced debugging):
```python
# main.py
parser.add_argument(
    "--network-timeout",
    type=float,
    default=30.0,
    help="Timeout for network operations (seconds)",
)
parser.add_argument(
    "--async-poll-interval",
    type=float,
    default=0.1,
    help="Polling interval for async operations (seconds)",
)
```

**Benefits**:
- Easy to adjust for debugging (increase timeouts)
- Can optimize for fast networks (decrease timeouts)
- Clear documentation of timing behavior
- Easier testing (mock with shorter timeouts)
- One place to tune performance

**Priority**: Medium - 1 hour, improves configurability

---

### 3.1 Extract Magic Numbers to Named Constants

**Issue**: Magic numbers scattered throughout the code make intent unclear and harder to configure.

**Current Examples**:
- `realtime_backend.py:209,76`: `24000` - Realtime session sample rate (duplicated)
- `realtime_backend.py:110,430`: `0.3` - VAD threshold (duplicated in session config)
- `realtime_backend.py:112,432`: `300` - Silence duration ms (duplicated)
- `realtime_backend.py:111,431`: `200` - Prefix padding ms (duplicated)
- `realtime_backend.py:93`: `0.6` - Temperature (duplicated)
- `realtime_backend.py:94`: `4096` - Max tokens
- `realtime_backend.py:276,275`: `0.1` - Async poll interval (many occurrences)
- `whisper_backend.py:272`: `2.0` - Minimum chunk duration
- `whisper_backend.py:271`: `30` - VAD frame duration ms
- `main.py:183,217`: `300` - 5 minutes in seconds (for confirmation prompt)
- `main.py:183`: `120` - Default capture duration

**Proposed Solution**:
```python
# constants.py
"""Constants for transcribe_demo."""

# Audio processing
DEFAULT_SAMPLE_RATE = 16000
REALTIME_SESSION_SAMPLE_RATE = 24000
DEFAULT_CHANNELS = 1

# VAD configuration
VAD_FRAME_DURATION_MS = 30
VAD_DEFAULT_AGGRESSIVENESS = 2
MIN_CHUNK_DURATION_SECONDS = 2.0
DEFAULT_MAX_CHUNK_DURATION = 60.0
DEFAULT_MIN_SILENCE_DURATION = 0.2
DEFAULT_MIN_SPEECH_DURATION = 0.25
DEFAULT_SPEECH_PAD_DURATION = 0.2

# Display configuration
CONCATENATION_DISPLAY_FREQUENCY = 3  # Show every N chunks

# Realtime API
REALTIME_CHUNK_DURATION = 2.0
REALTIME_SESSION_SAMPLE_RATE = 24000
REALTIME_VAD_THRESHOLD = 0.3
REALTIME_VAD_PREFIX_PADDING_MS = 200
REALTIME_VAD_SILENCE_DURATION_MS = 300
REALTIME_TEMPERATURE = 0.6
REALTIME_MAX_TOKENS = 4096
ASYNC_POLL_INTERVAL = 0.1  # seconds

# User interaction
CAPTURE_DURATION_CONFIRMATION_THRESHOLD = 300  # 5 minutes
DEFAULT_CAPTURE_DURATION = 120  # 2 minutes

# Device preferences
DEVICE_PREFERENCE_ORDER = ["cuda", "mps", "cpu"]

# Model mappings
ENGLISH_ONLY_MODELS = {
    "tiny": "tiny.en",
    "base": "base.en",
}
```

**Benefits**:
- Self-documenting code
- Easy to adjust configuration
- Centralized values
- Type hints possible with Final

---

### 3.2 Create Configuration Dataclasses

**Issue**: Functions have many parameters (10+), making them hard to use and test.

**Example**: `run_whisper_transcriber()` has 12 parameters

**Proposed Solution**: Already covered in sections 2.1 with `WhisperConfig`, `VADConfig`, and `DeviceConfig`.

Additional suggestion:
```python
@dataclass
class TranscribeConfig:
    """Main configuration for the transcribe-demo application."""
    backend: str  # 'whisper' or 'realtime'

    # Audio settings
    sample_rate: int = DEFAULT_SAMPLE_RATE
    channels: int = DEFAULT_CHANNELS

    # Whisper settings
    whisper_model: str = "turbo"
    device_preference: str = "auto"
    require_gpu: bool = False

    # VAD settings
    vad: VADConfig = field(default_factory=VADConfig)

    # Realtime settings
    realtime_model: str = "gpt-realtime-mini"
    realtime_endpoint: str = "wss://api.openai.com/v1/realtime"
    realtime_instructions: str = "..."

    # SSL settings
    ca_cert: Optional[Path] = None
    disable_ssl_verify: bool = False

    # Optional
    api_key: Optional[str] = None
    temp_file: Optional[Path] = None

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> TranscribeConfig:
        """Create config from argparse namespace."""
        return cls(
            backend=args.backend,
            sample_rate=args.samplerate,
            channels=args.channels,
            whisper_model=args.model,
            device_preference=args.device,
            require_gpu=args.require_gpu,
            vad=VADConfig(
                aggressiveness=args.vad_aggressiveness,
                min_silence_duration=args.vad_min_silence_duration,
                min_speech_duration=args.vad_min_speech_duration,
                speech_pad_duration=args.vad_speech_pad_duration,
                max_chunk_duration=args.max_chunk_duration,
            ),
            # ... other fields
        )
```

**Benefits**:
- Single object to pass around
- Easy to serialize/deserialize
- Type-safe with dataclasses
- Clear structure

### 3.3 Decouple Abseil Flags From Runtime Modules (main.py)

**Issue**: Every `flags.DEFINE_*` call runs at import time in `main.py:15-188`, which means simply importing `transcribe_demo.main` mutates global Abseil state. Tests (`tests/test_main_utils.py:12-21`) must manually call `FLAGS(["pytest"], known_only=True)` to avoid `DuplicateFlagError`, and any library code that wants to reuse `ChunkCollectorWithStitching` or the comparison helpers must also drag in the CLI flag globals. The rest of the file (`main.py:320-740`) reads `FLAGS` directly, so there is no way to construct configurations programmatically.

**Proposed Solution**:
1. Move all CLI flag registration into a dedicated module (e.g., `transcribe_demo/cli_flags.py`) that exposes `def register_flags(flag_holder: flags.FlagValues) -> None`. Only the executable path should import Abseil.
2. Add a `parse_cli_config(argv: Sequence[str]) -> TranscribeConfig` helper that converts the parsed flag values into the dataclasses outlined in §3.2. This helper should live in a regular module (no Abseil dependency) so tests can create configs directly.
3. Update `cli_main()` to be a thin wrapper:
   ```python
   # cli.py
   from absl import app, flags
   from .config import parse_cli_config
   from .runner import run_transcribe_session

   register_flags(flags.FLAGS)

   def main(argv):
       config = parse_cli_config(flags.FLAGS)
       run_transcribe_session(config)

   if __name__ == "__main__":
       app.run(main)
   ```
   The new `run_transcribe_session()` function should contain the current backend-selection logic and accept a `TranscribeConfig`, making it importable without Abseil.
4. Update tests to import `run_transcribe_session()` (or the backend-specific helpers) instead of mutating `FLAGS`. This removes the `FLAGS(["pytest"], known_only=True)` workaround entirely.

**Benefits**:
- Cleaner separation between CLI parsing and business logic.
- Importing `transcribe_demo.main` no longer has side effects, which simplifies type checking and tooling.
- Tests and future GUIs/REST endpoints can instantiate `TranscribeConfig` directly without depending on Abseil.
- Eliminates sporadic `DuplicateFlagError` issues when multiple tests import the module.

---

## Priority 4: Testing Improvements

### 4.1 Extract Test Fixtures (test_vad.py)

**Issue**: Frame generation patterns are repeated across tests.

**Proposed Solution**:
```python
# conftest.py or in test_vad.py
import pytest
import numpy as np

@pytest.fixture
def vad_16khz():
    """Standard 16kHz VAD instance."""
    return WebRTCVAD(sample_rate=16000, frame_duration_ms=30, aggressiveness=2)

@pytest.fixture
def frame_size_16khz():
    """Frame size for 16kHz, 30ms."""
    return int(16000 * 30 / 1000)  # 480 samples

@pytest.fixture
def silent_frame(frame_size_16khz):
    """Generate a silent frame."""
    return np.zeros(frame_size_16khz, dtype=np.float32)

@pytest.fixture
def noise_frame(frame_size_16khz):
    """Generate a low-level noise frame."""
    return np.random.randn(frame_size_16khz).astype(np.float32) * 0.01

@pytest.fixture
def speech_like_frame(frame_size_16khz):
    """Generate a speech-like periodic signal."""
    t = np.linspace(0, 0.03, frame_size_16khz)
    return (np.sin(2 * np.pi * 200 * t) * 0.3).astype(np.float32)

# Usage:
def test_silence_detection(vad_16khz, silent_frame):
    """Test that silence is detected as non-speech."""
    assert not vad_16khz.is_speech(silent_frame)

def test_speech_detection(vad_16khz, speech_like_frame):
    """Test with simulated speech-like signal."""
    result = vad_16khz.is_speech(speech_like_frame)
    assert isinstance(result, bool)
```

**Benefits**:
- DRY principle for tests
- Consistent test data
- Easier to maintain
- More readable tests

---

### 4.2 Add Unit Tests for Remaining Gaps

Recent work added coverage for `ChunkCollectorWithStitching` (`tests/test_main_utils.py`) and the PCM conversion helper (`tests/test_realtime_backend_clipping.py`). Outstanding areas that still lack focused tests:

- **Device selection / GPU fallbacks**: cover `_mps_available` failure modes and `--require-gpu` behaviour.
- **SSL context helpers** (once extracted per §1.5).
- **`resample_audio` edge cases**: extremely short input, downsampling, and value range preservation.

Targeted tests for these helpers will make future refactors safer without duplicating existing coverage.

### 4.3 Add Edge Case Tests for Audio Utilities

**Issue**: Critical audio processing functions lack comprehensive edge case testing.

**Functions Missing Tests**:
- `resample_audio()` in `realtime_backend.py:48` and `file_audio_source.py:261`
- `float_to_pcm16()` in `realtime_backend.py:34`
- Audio format conversion edge cases
- URL download error handling

**Proposed Test Cases**:
```python
# tests/test_audio_utils.py (after extracting to audio_utils.py per §1.1)

def test_resample_audio_no_op():
    """Test that resampling with same rate returns unchanged audio."""
    audio = np.random.randn(1000).astype(np.float32)
    resampled = resample_audio(audio, 16000, 16000)
    np.testing.assert_array_equal(resampled, audio)

def test_resample_audio_empty():
    """Test resampling empty audio."""
    audio = np.zeros(0, dtype=np.float32)
    resampled = resample_audio(audio, 16000, 8000)
    assert len(resampled) == 0

def test_resample_audio_single_sample():
    """Test resampling very short audio."""
    audio = np.array([0.5], dtype=np.float32)
    resampled = resample_audio(audio, 16000, 8000)
    assert len(resampled) >= 1

def test_resample_audio_upsampling():
    """Test upsampling preserves value range."""
    audio = np.random.randn(100).astype(np.float32) * 0.5
    resampled = resample_audio(audio, 8000, 16000)
    assert len(resampled) == 200
    assert resampled.dtype == np.float32
    assert np.max(np.abs(resampled)) <= 1.0

def test_resample_audio_downsampling():
    """Test downsampling preserves value range."""
    audio = np.random.randn(1000).astype(np.float32) * 0.5
    resampled = resample_audio(audio, 16000, 8000)
    assert len(resampled) == 500
    assert resampled.dtype == np.float32

def test_float_to_pcm16_clipping():
    """Test PCM16 conversion clips values correctly."""
    audio = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    pcm_bytes = float_to_pcm16(audio)
    # Unpack and verify clipping
    pcm = np.frombuffer(pcm_bytes, dtype=np.int16)
    assert pcm[0] == -32767  # Clipped from -2.0
    assert pcm[1] == -32767  # -1.0
    assert pcm[2] == 0
    assert pcm[3] == 32767   # 1.0
    assert pcm[4] == 32767   # Clipped from 2.0

def test_float_to_pcm16_value_range():
    """Test PCM16 conversion preserves valid range."""
    audio = np.linspace(-1.0, 1.0, 100, dtype=np.float32)
    pcm_bytes = float_to_pcm16(audio)
    pcm = np.frombuffer(pcm_bytes, dtype=np.int16)
    assert np.min(pcm) >= -32767
    assert np.max(pcm) <= 32767
```

**Benefits**:
- Catches edge case bugs before production
- Documents expected behavior
- Makes refactoring safer
- Tests are fast (<1ms each)

**Priority**: Medium - 2 hours

---

## Priority 5: Code Quality Improvements

### 5.0 Add Missing Docstrings to Public Functions

**Issue**: 8 public functions lack docstrings, making the codebase harder to understand and maintain.

**Locations**:
- `main.py:514` - `main()` function (240 lines, no docstring)
- `whisper_backend.py:54` - `load_whisper_model()` (43 lines, no docstring)
- `whisper_backend.py:244` - `run_whisper_transcriber()` (339 lines, no docstring)
- `realtime_backend.py:77` - `transcribe_full_audio_realtime()` (132 lines, no docstring)
- `realtime_backend.py:193` - `run_realtime_transcriber()` (218 lines, no docstring)
- `file_audio_source.py:18` - `FileAudioSource.__init__()` (no docstring)
- `session_replay.py:238` - `retranscribe_session()` (253 lines, no docstring)
- `audio_capture.py:17` - `AudioCaptureManager.__init__()` (minimal docstring)

**Proposed Solution**:
Add comprehensive docstrings following Google/NumPy style with:
- Brief one-line summary
- Detailed description of behavior
- Args section with types and descriptions
- Returns section
- Raises section for exceptions
- Examples for complex functions
- Notes for important behavior/caveats

**Example**:
```python
def run_whisper_transcriber(
    model_name: str,
    sample_rate: int,
    channels: int,
    device: str,
    require_gpu: bool,
    language: str | None,
    vad_aggressiveness: int,
    min_silence_duration: float,
    min_speech_duration: float,
    speech_pad_duration: float,
    max_chunk_duration: float,
    chunk_consumer: object | None = None,
    max_capture_duration: float | None = None,
    audio_source: object | None = None,
    save_chunk_audio: str | None = None,
    ca_cert: str | None = None,
    insecure: bool = False,
) -> None:
    """
    Run Whisper-based transcription with VAD chunking.

    Captures audio from the default microphone (or custom audio source), uses
    WebRTC VAD to detect speech segments, and transcribes each chunk with
    OpenAI's Whisper model. Chunks are transcribed as soon as VAD detects
    sufficient silence or the chunk reaches max_chunk_duration.

    Args:
        model_name: Whisper model name ('tiny', 'base', 'small', 'medium', 'large', 'turbo')
        sample_rate: Audio sample rate in Hz (typically 16000)
        channels: Number of audio channels (1 for mono, 2 for stereo)
        device: Device for inference ('auto', 'cuda', 'mps', 'cpu')
        require_gpu: If True, abort if no GPU available
        language: Language code ('en', 'es', etc.) or None for auto-detection.
                  IMPORTANT: Setting to None can cause hallucinations on silence.
        vad_aggressiveness: WebRTC VAD aggressiveness (0-3, higher = more aggressive)
        min_silence_duration: Minimum silence duration (seconds) to trigger chunk boundary
        min_speech_duration: Minimum speech duration (seconds) to keep a chunk
        speech_pad_duration: Padding (seconds) around speech segments
        max_chunk_duration: Maximum chunk duration (seconds) before forced transcription
        chunk_consumer: Optional callback for each transcribed chunk.
                        Signature: (chunk_index, text, start, end, inference_seconds) -> None
        max_capture_duration: Optional maximum capture duration in seconds
        audio_source: Optional custom audio source (for testing/file input)
        save_chunk_audio: Optional directory to save raw chunk audio files
        ca_cert: Optional path to CA certificate bundle for SSL
        insecure: If True, disable SSL certificate verification (NOT RECOMMENDED)

    Raises:
        RuntimeError: If requested device unavailable or GPU required but missing
        FileNotFoundError: If ca_cert path doesn't exist

    Notes:
        - VAD chunking creates variable-length chunks based on speech pauses
        - Chunk text may have trailing punctuation that needs cleanup for stitching
        - See CLAUDE.md for critical implementation details about VAD and stitching
    """
```

**Benefits**:
- Better IDE autocomplete and help
- Easier onboarding for new developers
- Self-documenting code
- Can generate API documentation automatically
- Clarifies expected behavior and edge cases

**Priority**: High - 30-60 minutes, significantly improves maintainability

---

### 5.1 Add Type Hints for Callables

**Issue**: Callback functions lack proper type hints.

**Examples**:
- `whisper_backend.py:156`: `chunk_consumer` type is too vague
- `realtime_backend.py:56`: `chunk_consumer` has type `object`

**Proposed Solution**:
```python
# types.py
"""Type definitions for transcribe_demo."""

from typing import Protocol, Optional

class ChunkConsumer(Protocol):
    """Protocol for chunk consumer callbacks."""

    def __call__(
        self,
        chunk_index: int,
        text: str,
        absolute_start: float,
        absolute_end: float,
        inference_seconds: Optional[float] = None,
    ) -> None:
        """
        Process a transcription chunk.

        Args:
            chunk_index: Sequential chunk number
            text: Transcribed text
            absolute_start: Start time from session beginning
            absolute_end: End time from session beginning
            inference_seconds: Inference time (None for realtime mode)
        """
        ...

# Usage:
def run_whisper_transcriber(
    # ... other params
    chunk_consumer: Optional[ChunkConsumer] = None,
) -> None:
    ...
```

**Benefits**:
- Better IDE support
- Type checking catches errors
- Self-documenting code
- Protocol is duck-typed (existing code works)

---

### 5.2 Improve Error Messages

**Issue**: Some error messages could be more helpful.

**Examples**:
```python
# Current (whisper_backend.py:184):
raise RuntimeError("CUDA GPU requested (--device=cuda) but none is available.")

# Better:
raise RuntimeError(
    "CUDA GPU requested (--device=cuda) but none is available. "
    "Available options: use --device=cpu or install CUDA drivers."
)

# Current (main.py:318):
raise RuntimeError(
    "OpenAI API key required for realtime transcription. "
    "Provide --api-key or set OPENAI_API_KEY."
)

# Better:
raise RuntimeError(
    "OpenAI API key required for realtime transcription. "
    "Either:\n"
    "  1. Set environment variable: export OPENAI_API_KEY='sk-...'\n"
    "  2. Use command line flag: --api-key sk-...\n"
    "Get your API key from: https://platform.openai.com/api-keys"
)
```

**Benefits**:
- Reduces user frustration
- Faster problem resolution
- Better user experience

---

### 5.3 Add Logging Instead of Print Statements

**Issue**: Using `print(..., file=sys.stderr)` throughout makes it hard to control verbosity.

**Proposed Solution**:
```python
import logging

logger = logging.getLogger(__name__)

# Instead of:
print("Running transcription on CUDA GPU.", file=sys.stderr)

# Use:
logger.info("Running transcription on CUDA GPU.")

# Add --verbose flag:
parser.add_argument(
    "--verbose", "-v",
    action="count",
    default=0,
    help="Increase verbosity (-v, -vv, -vvv)"
)

# Configure logging based on verbosity:
def configure_logging(verbosity: int):
    levels = [logging.WARNING, logging.INFO, logging.DEBUG]
    level = levels[min(verbosity, len(levels) - 1)]
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
```

**Benefits**:
- Configurable verbosity
- Structured logging
- Can redirect to file
- Standard Python practice

### 5.4 Surface Cleaned Chunk Text During Logging (ChunkCollector + SessionLogger)

**Issue**: `session_logger.log_chunk()` is called without `cleaned_text`, so each chunk is stored with the raw transcript only. After the backend returns, `main.py:606-615` iterates over `collector.get_cleaned_chunks()` and calls `session_logger.update_chunk_cleaned_text()` to backfill the cleaned text. This means:
- If the process exits early (Ctrl+C, crash), the session log never receives cleaned text.
- We traverse the entire chunk list twice even though `ChunkCollectorWithStitching` already knows the cleaned text while emitting each chunk.
- The `SessionLogger` API exposes a mutating `update_chunk_cleaned_text()` method solely to compensate for this delayed write.

**Proposed Solution**:
1. Have `ChunkCollectorWithStitching` compute and store the cleaned text alongside each `TranscriptionChunk` when `_clean_chunk_text()` is called. Expose it via the chunk record rather than recomputing later.
2. Pass the cleaned text to `session_logger.log_chunk(..., cleaned_text=cleaned)` immediately inside both backends (`whisper_backend.py:499` and `realtime_backend.py:438`). This allows JSON serialization to include the cleaned form even if the run stops mid-session.
3. Remove `SessionLogger.update_chunk_cleaned_text()` entirely; callers no longer need a second pass over the chunks section in `main.py`.

**Benefits**:
- Session logs always contain both raw and cleaned text per chunk, even if the session is interrupted.
- Eliminates redundant O(n) passes over the chunk list and the extra logger mutation API.
- Simplifies the main control flow (no post-run loop just to backfill data) and makes the logging path easier to test.

**Current Implementation** (2-pass approach):
```python
# main.py:603-604 (whisper) and 688-689 (realtime) - identical code
for chunk_index, cleaned_text in collector.get_cleaned_chunks():
    session_logger.update_chunk_cleaned_text(chunk_index, cleaned_text)
```

### 5.7 Remove Unused Imports

**Issue**: Unused imports clutter the codebase and can confuse readers.

**Locations**:
- `session_logger.py:4` - `import wave` (never used, JSON-only logging)
- `session_replay.py:7` - `import wave` (never used, delegates to session_logger)

**Solution**:
```python
# Remove these lines:
import wave  # Not needed
```

**Benefits**:
- Cleaner imports
- Faster import times (minimal but measurable)
- Clearer dependencies
- Linters won't complain

**Priority**: Low - 2 minutes, cleanup

---

### 5.8 Move Import to Module Level

**Issue**: `import time` is inside a function instead of at module level.

**Location**: `audio_capture.py:168`

**Current Code**:
```python
def __enter__(self) -> AudioCaptureManager:
    import time  # Inside function
    # ...
```

**Solution**:
```python
# At top of file with other imports:
import time

def __enter__(self) -> AudioCaptureManager:
    # Remove import from here
```

**Benefits**:
- Follows Python style guide (PEP 8)
- Clearer dependencies at top of file
- Slightly faster (imports cached, but still best practice)

**Priority**: Low - 1 minute, style fix

---

## Priority 6: Recently Identified Opportunities

This section contains refactoring opportunities discovered during recent codebase review (2025).

### 6.1 Extract Synthetic Audio Test Utility (tests)

**Issue**: The `_generate_synthetic_audio()` function is defined only in `test_backend_time_limits.py:33-56` but would be useful across all test files.

**Location**: `test_backend_time_limits.py:33-56` (24 lines)

**Current Usage**: Used to create fast, predictable test audio without file I/O

**Proposed Solution**:
```python
# tests/conftest.py or tests/test_utils.py
import numpy as np
import pytest

@pytest.fixture
def generate_synthetic_audio():
    """Factory fixture for generating synthetic audio for tests."""
    def _generate(
        duration_seconds: float = 3.0,
        sample_rate: int = 16000,
        frequency: float = 440.0,
    ) -> np.ndarray:
        """
        Generate synthetic audio for testing.

        Args:
            duration_seconds: Duration of audio in seconds
            sample_rate: Sample rate in Hz
            frequency: Frequency of sine wave in Hz

        Returns:
            Mono float32 audio array
        """
        num_samples = int(duration_seconds * sample_rate)
        t = np.linspace(0, duration_seconds, num_samples, endpoint=False)
        audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)
        return audio * 0.3  # Scale to reasonable amplitude

    return _generate

# Usage in tests:
def test_something(generate_synthetic_audio):
    audio = generate_synthetic_audio(duration_seconds=2.0, sample_rate=16000)
    # ... use audio
```

**Benefits**:
- DRY: Reusable across all test files
- Consistent test audio generation
- Can add variations (noise, silence, speech-like patterns)
- Discoverable through pytest fixtures

**Priority**: Low - 15 minutes, improves test maintainability

---

### 6.2 Normalize Language Parameter Utility (both backends)

**Issue**: Language parameter normalization is duplicated in both backends.

**Locations**:
- `whisper_backend.py:272-274`
- `realtime_backend.py:228, 252-254`

**Current Code**:
```python
# whisper_backend.py
normalized_language = None
if language and language.lower() != "auto":
    normalized_language = language

# realtime_backend.py
language_value = (language or "").strip()
if language_value and language_value.lower() != "auto":
    # use it
```

**Proposed Solution**:
```python
# utils.py or config.py
def normalize_language(language: str | None) -> str | None:
    """
    Normalize language parameter.

    Args:
        language: Language code or "auto"

    Returns:
        Normalized language code or None if auto/empty
    """
    if not language:
        return None

    normalized = language.strip().lower()
    return None if normalized == "auto" else language
```

**Benefits**:
- Single source of truth
- Consistent behavior across backends
- Easier to test edge cases
- Clear documentation of "auto" → None conversion

**Priority**: Low - 5 minutes

---

### 6.3 Extract RealtimeMessageHandler Class (realtime_backend.py)

**Issue**: WebSocket message handling logic is embedded in nested `receiver()` function with state management (partials dict, item ID tracking).

**Location**: Inside `realtime_backend.py:run_realtime_transcriber()` nested function

**Current Structure**:
```python
async def receiver() -> None:
    partials: dict[str, str] = {}
    # ... event handling logic mixed with state
```

**Proposed Solution**:
```python
class RealtimeMessageHandler:
    """Handles OpenAI Realtime API WebSocket messages."""

    def __init__(self):
        self.partials: dict[str, str] = {}
        self.completed: list[str] = []
        self.chunk_counter = 0

    def handle_delta(self, payload: dict) -> None:
        """Handle transcript delta event."""
        item_id = payload.get("item_id", "")
        delta_text = payload.get("delta", "")
        self.partials[item_id] = self.partials.get(item_id, "") + delta_text

    def handle_completed(self, payload: dict) -> str | None:
        """
        Handle transcript completed event.

        Returns:
            Final transcript text or None if empty
        """
        item_id = payload.get("item_id", "")
        transcript = payload.get("transcript", "").strip()

        # Prefer completed transcript over partial accumulation
        if transcript:
            self.partials.pop(item_id, None)
            return transcript
        elif item_id in self.partials:
            final_text = self.partials.pop(item_id).strip()
            return final_text if final_text else None

        return None

    def handle_error(self, payload: dict, event_type: str) -> None:
        """Handle error events."""
        error_detail = payload.get("error", {})
        print(
            f"Realtime API {event_type}: {error_detail}",
            file=sys.stderr
        )
```

**Benefits**:
- Testable without WebSocket connection
- Clear state management
- Easier to modify message handling logic
- Could be reused if adding other realtime streaming backends

**Priority**: Medium - 1 hour, improves testability

---

### 6.4 Extract FakeAudioCaptureManager to Test Utils (tests)

**Issue**: `FakeAudioCaptureManager` is duplicated across 3 test files with slight variations.

**Locations**:
- `test_backend_time_limits.py:59-150` (91 lines)
- `test_whisper_backend_integration.py:50-102` (52 lines)
- `test_realtime_backend_integration.py:37-91` (54 lines)

**Impact**:
- Maintenance burden (changes need to be made in 3 places)
- Inconsistent implementations (each has slight differences)
- Tests become brittle if real `AudioCaptureManager` API changes

**Proposed Solution**:
```python
# tests/conftest.py
import pytest
from transcribe_demo.audio_capture import AudioCaptureManager

@pytest.fixture
def fake_audio_capture_manager():
    """Factory for creating FakeAudioCaptureManager instances."""

    class FakeAudioCaptureManager:
        """Test double for AudioCaptureManager with configurable behavior."""

        def __init__(
            self,
            audio_chunks: list[np.ndarray] | None = None,
            simulate_time_limit: bool = False,
            capture_duration_seconds: float | None = None,
        ):
            self.audio_chunks = audio_chunks or []
            self.simulate_time_limit = simulate_time_limit
            self.capture_duration_seconds = capture_duration_seconds

            # State tracking
            self.stop_event = threading.Event()
            self.audio_queue: queue.Queue[np.ndarray] = queue.Queue()
            self.full_audio_chunks: list[np.ndarray] = []
            self.chunk_index = 0

        def __enter__(self):
            # Start feeding audio chunks in background thread
            self._feed_thread = threading.Thread(target=self._feed_audio)
            self._feed_thread.start()
            return self

        def __exit__(self, *args):
            self.stop_event.set()
            self._feed_thread.join(timeout=2.0)

        def _feed_audio(self):
            """Feed audio chunks to queue."""
            for chunk in self.audio_chunks:
                if self.stop_event.is_set():
                    break
                self.audio_queue.put(chunk)
                time.sleep(0.01)  # Small delay to simulate real capture

            if self.simulate_time_limit:
                time.sleep(0.1)
                self.stop_event.set()

    return FakeAudioCaptureManager

# Usage in tests:
def test_something(fake_audio_capture_manager, generate_synthetic_audio):
    audio = generate_synthetic_audio(duration_seconds=2.0)
    fake_mgr = fake_audio_capture_manager(
        audio_chunks=[audio],
        simulate_time_limit=True,
    )
    # ... test with fake_mgr
```

**Benefits**:
- Single source of truth
- Consistent behavior across tests
- Easier to extend with new features
- Reduces test code by ~150 lines

**Priority**: Medium - 1 hour, significant test cleanup

---

### 6.5 Extract Cleaned Text Backfilling Pattern (main.py)

**Issue**: Cleaned text is computed in a second pass after all chunks are collected, duplicated in both backend paths.

**Status**: Related to §5.4, but worth highlighting as a distinct issue

**Locations**:
- `main.py:603-604` (whisper backend)
- `main.py:688-689` (realtime backend)

**Current Flow**:
1. Backend emits chunk with raw text → `collector(chunk_index, text, ...)`
2. ChunkCollector stores chunk in `self._chunks`
3. After transcription completes: `collector.get_cleaned_chunks()` iterates all chunks
4. For each chunk: `session_logger.update_chunk_cleaned_text(chunk_index, cleaned)`

**Problems**:
- Two passes over all chunks (O(n) redundancy)
- Cleaned text not available if session crashes/interrupted
- SessionLogger needs mutation API (`update_chunk_cleaned_text`)
- Logic duplicated in both backend paths

**Proposed Solution**:
1. Compute cleaned text immediately in `ChunkCollectorWithStitching.__call__`:
```python
def __call__(self, chunk_index, text, absolute_start, absolute_end, inference_seconds=None):
    # ... existing code ...

    # Compute cleaned text immediately
    is_final = False  # We don't know yet, so clean conservatively
    cleaned_text = self._clean_chunk_text(text, is_final_chunk=is_final)

    chunk = TranscriptionChunk(
        index=chunk_index,
        text=text,
        cleaned_text=cleaned_text,  # Store both
        start_time=absolute_start,
        end_time=absolute_end,
        # ...
    )
    self._chunks.append(chunk)
```

2. Pass cleaned text to session logger immediately:
```python
# In both whisper_backend.py and realtime_backend.py:
if chunk_consumer:
    chunk_consumer(chunk_index, text, start, end, inference)

    # NEW: Pass cleaned text immediately if logger supports it
    if hasattr(chunk_consumer, 'get_last_cleaned_text'):
        cleaned = chunk_consumer.get_last_cleaned_text()
        # pass to session logger
```

3. Remove the backfilling loop from `main.py`

**Benefits**:
- Single pass over chunks (performance)
- Cleaned text preserved even if session interrupted
- Simpler SessionLogger API (no `update_chunk_cleaned_text` needed)
- No code duplication between backends

**Priority**: Medium - 2 hours, architectural improvement

---

## Future Improvements

### TODO: Use WebRTC VAD for Realtime Backend (Instead of Server-Side VAD)

**Problem**: The OpenAI Realtime API's server-side VAD is designed for conversational AI (detecting when a user stops speaking), not continuous transcription of pre-recorded content. For fast-paced speech like news broadcasts or podcasts, the server VAD requires extensive tuning and still produces fewer, less predictable chunk boundaries compared to local VAD.

**Current Architecture**:
- **Whisper backend**: WebRTC VAD locally → chunks at natural pauses → sends to Whisper
- **Realtime backend**: Streams continuously → server-side VAD detects turns → emits chunks

**Proposed Architecture**:
- **Both backends**: WebRTC VAD locally → chunks at natural pauses → send to transcription service

**Benefits**:
1. **Consistent behavior**: Both backends use identical chunking logic - easier to reason about and predict
2. **Local control**: No dependency on server VAD settings or API changes
3. **Proven for broadcast content**: WebRTC VAD already works great for NPR audio in Whisper backend
4. **Bandwidth efficiency**: Only send audio during speech, skip silence periods
5. **Deterministic**: Server VAD is a black box; WebRTC VAD is fully under our control
6. **Simpler configuration**: One set of VAD settings (`--vad_*` flags) for both backends
7. **Better chunk boundaries**: More frequent, predictable chunks for continuous speech

**Current Issues with Server VAD**:
- Fast-paced content (news, podcasts) requires aggressive tuning (`--realtime_vad_silence_duration_ms 100`)
- NPR newscast produced only 3 chunks in 120 seconds even with tuned settings
- Configuration parameters are realtime-specific (`--realtime_vad_*`) and don't align with Whisper backend settings
- Users must learn two different VAD configuration systems

**Implementation Approach**:
1. Extract WebRTC VAD logic from `whisper_backend.py` into a shared module (e.g., `vad.py`)
2. Make `WebRTCVAD` class reusable for both backends
3. Update `run_realtime_transcriber()` to:
   - Use WebRTC VAD for local chunking (same as Whisper backend)
   - Send each VAD-detected chunk to the API with `include_turn_detection=False`
   - Await transcription response for each chunk before sending the next
4. Remove `--realtime_vad_*` flags (use existing `--vad_*` flags for both backends)
5. Update CLAUDE.md to document unified VAD configuration

**Code Structure**:
```python
# Shared VAD module
vad = WebRTCVAD(sample_rate, vad_aggressiveness, ...)

# In realtime backend
for chunk in vad.process_audio_stream():
    # Send chunk with turn detection disabled
    await send_audio_chunk(chunk, turn_detection=False)
    await commit_and_wait_for_transcription()
```

**Priority**: High - significantly improves realtime backend usability for continuous speech
**Estimated Effort**: 4-6 hours (extract VAD module + update realtime backend + tests + docs)

**Related**:
- Current server VAD implementation: `realtime_backend.py:80-105` (`_create_session_update`)
- WebRTC VAD class: `whisper_backend.py:28-221`
- Server VAD configuration flags: `main.py:166-186`

---

### TODO: Silero VAD Backend for Background Noise/Music Robustness

**Problem**: WebRTC VAD can only distinguish silence vs voice, causing false positives with background music/noise.

**Solution**: Add Silero VAD as alternative backend that can distinguish voice vs music vs noise vs silence.

**Benefits**:
- 10-30x more accurate with background noise/music
- Probabilistic scores (adjustable threshold 0.0-1.0, recommend 0.7-0.9 for noisy environments)
- Deep learning-based vs WebRTC's simple GMM
- Processing: <1ms per 30ms chunk (real-time capable)
- Supports any sample rate (WebRTC limited to 8k/16k/32k/48k)

**Implementation Path**:
1. Add dependency: `uv add silero-vad torch`
2. Create `SileroVAD` class in `whisper_backend.py` (see TODO comment at line 124)
3. Add CLI arguments: `--vad-backend {webrtc,silero}` and `--vad-threshold FLOAT`
4. Update `run_whisper_transcriber()` to instantiate appropriate VAD class based on backend choice

**Additional Noise Robustness Improvements**:
- Add `initial_prompt` to Whisper: "Ignore background music and noise. Transcribe only clear human speech."
- Set `condition_on_previous_text=False` to prevent hallucination loops with noise

**Usage Example**:
```bash
# For noisy environments with background music
uv run transcribe-demo --vad-backend silero --vad-threshold 0.8
```

**Location**: Detailed implementation in `whisper_backend.py:32-80`

**References**:
- https://github.com/snakers4/silero-vad
- Research showing Silero VAD's superior performance with music/noise

### Future Feature: Sliding Window Refinement (NOT YET IMPLEMENTED)

**Concept**: Use a 3-chunk sliding window to refine the middle chunk with more context.

**How it would work:**
1. Store raw audio buffers for the last 3 chunks
2. When chunk N arrives, stitch audio from chunks N-2, N-1, N
3. Re-transcribe the 3-chunk window with Whisper
4. Use word-level timestamps to extract refined text for chunk N-1 (middle chunk)
5. Display refined version of chunk N-1 after chunk N processing

**Benefits:**
- Better context reduces boundary transcription errors
- Improved accuracy for phrases that span chunk boundaries
- More natural linguistic flow across chunks

**Trade-offs:**
- Adds 1-chunk latency (chunk N-1 displayed after N arrives)
- Requires ~3x inference time per chunk
- Needs word timestamps (feature not currently implemented)
- Requires additional memory to store raw audio buffers

**Status:**
- CLI flag added: `--refine-with-context` (currently shows error if used)
- Implementation TODO documented in `main.py` (lines 71-95)
- Requires modifications to `whisper_backend.py` to pass raw audio buffers

---

## Summary and Prioritized Action Plan

### Current State Assessment

**Strengths**:
- Clear module separation between backends
- Strong test coverage (86% test-to-source ratio, 97 tests)
- Well-documented development guidelines (CLAUDE.md)
- Modern Python style (type hints, dataclasses where used)
- No critical bugs detected

**Pain Points**:
- **Long functions**: 4 functions >200 lines (main(), run_whisper_transcriber(), run_realtime_transcriber(), worker())
- **Code duplication**: resample_audio() (2×), session config (15 lines × 2), SSL setup, output formatting (40 lines × 2), test utilities (150+ lines × 3)
- **Missing documentation**: 8 public functions lack docstrings
- **Silent failures**: 11 bare exception handlers without logging
- **Magic numbers**: Scattered constants lack semantic meaning (timeouts, thresholds)
- **Tight coupling**: Abseil flags make library reuse difficult
- **Two-pass patterns**: Cleaned text backfilling after collection

---

### Immediate Wins (~2 hours total)

Start here for quick improvements with high value:

| Priority | Item | Time | Impact | Lines Saved |
|----------|------|------|--------|-------------|
| **0.1** | Extract duplicate Whisper text extraction | 10 min | High | 5 lines |
| **0.2** | Reuse _run_async() in Whisper backend | 20 min | Medium | 10 lines |
| **0.3** | Extract color code initialization | 15 min | Low | 8 lines |
| **5.7** | Remove unused imports | 2 min | Low | 2 lines |
| **5.8** | Move time import to module level | 1 min | Low | 0 (style) |
| **1.1** | Deduplicate resample_audio() | 30 min | High | 25 lines |
| **1.1b** | Extract stdin.isatty() check | 20 min | Medium | 10 lines |
| **1.2b** | Extract final output printing | 5 min | Medium | 40 lines |
| **3.1** | Extract magic numbers to constants | 30 min | High | 0 (readability) |
| **6.2** | Normalize language parameter | 5 min | Low | 5 lines |

**Total**: ~2.3 hours, **saves ~105 lines**, improves maintainability

**Priority 0 wins alone**: 45 minutes, saves 23 lines (pure deletion/consolidation)

**Next Priority** (additional 1 hour):
- **§5.0**: Add docstrings to 8 public functions (30-60 min) - High impact for maintainability

---

### Medium Impact Refactorings (6-10 hours total)

These require more planning but provide substantial benefits:

1. **§5.6: Replace bare exception handlers with logging** (1 hour)
   - Add logging to 11 silent exception handlers
   - Significantly improves debuggability
   - Makes production issues easier to diagnose

2. **§3.0: Make hardcoded timeouts configurable** (1 hour)
   - Extract timeout constants
   - Optional CLI flags for debugging
   - Easier testing and performance tuning

3. **§4.3: Add edge case tests for audio utilities** (2 hours)
   - Test resample_audio() edge cases
   - Test float_to_pcm16() clipping
   - Makes refactoring safer

4. **§1.2: Centralize terminal formatting** (3 hours)
   - Create `TerminalFormatter` class
   - Eliminates scattered ANSI color code handling
   - Consistent styling across all output

5. **§1.3: Extract device selection logic** (2 hours)
   - Create `DeviceConfig.detect_device()` classmethod
   - Testable in isolation, clearer logic

6. **§1.5: SSL context manager** (2 hours)
   - Proper cleanup with context manager
   - Reusable across both backends

7. **§6.3: Extract RealtimeMessageHandler** (1 hour)
   - Testable without WebSocket
   - Clear state management

8. **§6.4: Extract FakeAudioCaptureManager to conftest** (1 hour)
   - Reduces test code by ~150 lines
   - Consistent test infrastructure

---

### Major Architectural Refactorings (1-3 days each)

These are larger efforts that fundamentally improve architecture:

1. **§2.1 + §2.4: Decompose worker functions** (2-3 days)
   - Break down 200-300 line functions into focused classes
   - `WhisperTranscriptionWorker`, `AudioProcessor`, `TranscriptionWorker`
   - `RealtimeTranscriber`, `AudioSender`, `TranscriptReceiver`
   - **Impact**: Each component testable independently, much clearer control flow

2. **§3.3: Decouple from Abseil flags** (1-2 days)
   - Move CLI parsing to separate module
   - Create `parse_cli_config()` that returns dataclasses
   - Make library code importable without side effects
   - **Impact**: Library reuse, cleaner testing, better IDE support

3. **§5.3: Add logging framework** (1 day)
   - Replace all `print(..., file=sys.stderr)` with proper logging
   - Add `--verbose` flag with log levels
   - **Impact**: Configurable verbosity, structured logging, better debugging

4. **§5.4 + §6.5: Inline cleaned text computation** (2 hours)
   - Compute cleaned text during emission, not in second pass
   - Remove SessionLogger mutation API
   - **Impact**: Simpler flow, preserved data on interruption, no duplication

---

### Recommended Execution Order

**Phase 1: Quick Cleanup** (Day 1, ~2.5 hours)
```
Morning:   §5.7, §5.8 (Remove unused imports, fix import location) - 3 min
           §1.1 (Deduplicate resample_audio) - 30 min
           §1.1b (Extract stdin check) - 20 min
           §1.2b (Extract final output printing) - 5 min
           §3.1 (Extract magic numbers) - 30 min
           §6.2 (Normalize language parameter) - 5 min
Afternoon: §5.0 (Add docstrings to 8 functions) - 60 min
           → 82 lines saved, much better documentation
```

**Phase 2: Quality & Debugging** (Day 2-3, ~4 hours)
```
Day 2:     §5.6 (Replace bare exceptions with logging) - 1 hour
           §3.0 (Make timeouts configurable) - 1 hour
           §4.3 (Add edge case tests) - 2 hours
           → Better debuggability and test coverage
```

**Phase 3: Code Organization** (Day 4-5, ~8 hours)
```
Day 4:     §1.2 (Terminal formatting) - 3 hours
           §1.3 (Device selection) - 2 hours
Day 5:     §1.5 (SSL context manager) - 2 hours
           §6.3 (RealtimeMessageHandler) - 1 hour
           → Cleaner abstractions
```

**Phase 4: Test Infrastructure** (Day 6, ~2 hours)
```
Day 6:     §6.1 (Synthetic audio fixture) - 30 min
           §6.4 (FakeAudioCaptureManager to conftest) - 1 hour
           §2.3 (Extract comparison module) - 1 hour
           → 150+ test lines saved
```

**Phase 5: Configuration Refactoring** (Week 2, ~8 hours)
```
Day 7-8:   §3.2 (Configuration dataclasses) - 6 hours
Day 9-10:  §3.3 (Decouple Abseil flags) - 6 hours
           → Library reusability, type-safe configs
```

**Phase 6: Major Decomposition** (Week 3-4, ~2-3 days)
```
Week 3-4:  §2.1, §2.4 (Worker function decomposition)
           §1.4 (Unify transcription session harness)
           → Much clearer code structure, easier to extend
```

**Phase 7: Advanced Features** (Week 5, optional)
```
Week 5:    §5.3 (Logging framework) - 1 day
           §5.4, §6.5 (Inline cleaned text) - 2 hours
           → Professional-grade observability
```

---

### Metrics to Track

**Code Health Indicators**:
- Total lines of code (currently 2,516 source)
- Test-to-source ratio (currently ~82%)
- Average function length (target: <50 lines for 80% of functions)
- Code duplication percentage (current: ~8%, target: <3%)
- Number of functions >100 lines (current: 8, target: <3)

**After Full Refactoring** (estimated):
- Source lines: ~2,800 (slight increase from better organization)
- Duplication: <2%
- Functions >100 lines: 1-2
- Test lines: ~2,200 (better fixtures = more compact tests)
- Test-to-source ratio: ~79% (slightly lower but higher quality)

---

### Questions Before Starting?

Before beginning any refactoring:

1. **Is there test coverage?** If not, add tests first
2. **Are line numbers accurate?** Verify in current codebase
3. **Will this break existing behavior?** Plan for backward compatibility
4. **Can it be done incrementally?** Prefer small PRs over large rewrites
5. **What's the rollback plan?** Keep commits focused and revertible

---

## Related Documentation

See [SITEMAP.md](SITEMAP.md) for a complete guide to all documentation.

**This document** answers "what should we improve?" (implementation opportunities, technical debt)
