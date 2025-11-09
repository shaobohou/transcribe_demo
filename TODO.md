# TODO: Refactoring Opportunities

This document tracks implementation-level refactoring opportunities, code quality improvements, and technical debt. For high-level design decisions and architectural rationale, see **DESIGN.md**. For development workflow and critical implementation rules, see **CLAUDE.md**.

**⚠️ Important**: Line numbers in this document may drift as code evolves. Always verify line numbers against the current codebase before starting refactoring work.

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

---

## Priority 1: High Impact Refactoring

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
    insecure_downloads: bool = False

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

---

## Priority 5: Code Quality Improvements

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
- Strong test coverage (2,061 test lines for 2,516 source lines)
- Well-documented development guidelines (CLAUDE.md)
- Modern Python style (type hints, dataclasses where used)

**Pain Points**:
- **Long functions**: 4 functions >200 lines (main(), run_whisper_transcriber(), run_realtime_transcriber(), ChunkCollectorWithStitching.__call__)
- **Code duplication**: Session config (15 lines × 2), SSL setup, output formatting (40 lines × 2), test utilities (150+ lines × 3)
- **Magic numbers**: Scattered constants lack semantic meaning
- **Tight coupling**: Abseil flags make library reuse difficult
- **Two-pass patterns**: Cleaned text backfilling after collection

---

### Immediate Wins (1-2 hours total)

Start here for quick improvements with high value:

| Priority | Item | Time | Impact | Lines Saved |
|----------|------|------|--------|-------------|
| **1** | §1.1: Extract realtime session config | 10 min | High | 15+ lines |
| **2** | §1.2b: Extract final output printing | 5 min | Medium | 40 lines |
| **3** | §3.1: Extract magic numbers to constants.py | 30 min | High | 0 (readability) |
| **4** | §6.2: Normalize language parameter | 5 min | Low | 5 lines |
| **5** | §2.3: Extract comparison to separate module | 1 hour | Medium | 166 lines |

**Total**: ~2 hours, saves ~226 lines, improves maintainability significantly

---

### Medium Impact Refactorings (4-8 hours each)

These require more planning but provide substantial benefits:

1. **§1.2: Centralize terminal formatting** (4 hours)
   - Create `TerminalFormatter` class
   - Eliminates scattered ANSI color code handling
   - Consistent styling across all output

2. **§1.3: Extract device selection logic** (2 hours)
   - Create `DeviceConfig.detect_device()` classmethod
   - Testable in isolation, clearer logic

3. **§1.5: SSL context manager** (2 hours)
   - Proper cleanup with context manager
   - Reusable across both backends

4. **§3.2: Configuration dataclasses** (6 hours)
   - `WhisperConfig`, `RealtimeConfig`, `VADConfig`, `TranscribeConfig`
   - Type-safe, reduces 12-parameter functions to single config object

5. **§6.3: Extract RealtimeMessageHandler** (1 hour)
   - Testable without WebSocket
   - Clear state management

6. **§6.4: Extract FakeAudioCaptureManager to conftest** (1 hour)
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

**Phase 1: Quick Wins** (Week 1)
```
Day 1-2: Complete all immediate wins (§1.1, §1.2b, §3.1, §6.2, §2.3)
         → 226 lines saved, improved readability
```

**Phase 2: Configuration & Infrastructure** (Week 2-3)
```
Day 3-5:  §1.2, §1.3, §1.5 (Terminal formatting, device selection, SSL)
Day 6-10: §3.2 (Configuration dataclasses) + §3.3 (Decouple Abseil)
         → Type-safe configs, library reusability
```

**Phase 3: Test Infrastructure** (Week 4)
```
Day 11-12: §6.1, §6.4 (Test utilities to conftest.py)
           → Cleaner tests, better maintainability
```

**Phase 4: Major Decomposition** (Week 5-6)
```
Day 13-20: §2.1, §2.4 (Worker function decomposition)
           → Much clearer code structure, easier to extend
```

**Phase 5: Quality & Polish** (Week 7)
```
Day 21-23: §5.3 (Logging framework)
Day 24-25: §5.4, §6.5 (Inline cleaned text)
Day 26:    §6.3 (RealtimeMessageHandler)
           → Professional-grade observability and cleaner flow
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
