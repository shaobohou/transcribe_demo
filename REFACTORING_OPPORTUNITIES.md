# Refactoring Opportunities Summary

**Generated**: 2025-11-14
**Total Lines of Code**: ~8,000 lines (source + tests)

This document summarizes refactoring opportunities identified in the transcribe_demo codebase. Opportunities are categorized by impact and organized from high-priority architectural improvements to lower-priority code quality enhancements.

---

## Table of Contents

1. [High-Priority Architectural Improvements](#high-priority-architectural-improvements)
2. [Medium-Priority Design Improvements](#medium-priority-design-improvements)
3. [Code Quality and Maintainability](#code-quality-and-maintainability)
4. [Performance and Efficiency](#performance-and-efficiency)
5. [Testing and Testability](#testing-and-testability)
6. [Documentation and Type Safety](#documentation-and-type-safety)

---

## High-Priority Architectural Improvements

### 1. Extract Formal AudioSource Protocol

**Current State**: `AudioCaptureManager` (audio_capture.py:20) and `FileAudioSource` (file_audio_source.py:21) implement the same interface informally through duck typing.

**Issue**:
- No explicit contract enforced at type-check time
- Backends check for duck-typed attributes at runtime
- Difficult to add new audio sources without checking all usage sites

**Recommendation**:
```python
# src/transcribe_demo/audio_source.py
from typing import Protocol
import queue
import threading
import numpy as np

class AudioSource(Protocol):
    """Common interface for all audio sources."""
    audio_queue: queue.Queue[np.ndarray | None]
    stop_event: threading.Event
    capture_limit_reached: threading.Event
    sample_rate: int
    channels: int

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def wait_until_stopped(self) -> None: ...
    def close(self) -> None: ...
    def get_full_audio(self) -> np.ndarray: ...
    def get_capture_duration(self) -> float: ...
```

**Benefits**:
- Type-safe audio source polymorphism
- Easier to add new sources (e.g., network streams, test fixtures)
- Better IDE support and documentation
- Pyright will catch interface violations

**Effort**: Low (2-3 hours)
**Impact**: High (foundation for future extensions)

---

### 2. Extract Backend Worker Classes

**Current State**: `run_whisper_transcriber` (whisper_backend.py:266) and `run_realtime_transcriber` (realtime_backend.py:262) use nested async functions for workers (376-515 and 452-604 respectively).

**Issues**:
- Nested functions are 100+ lines each
- Shared state via nonlocal variables is error-prone
- Difficult to unit test individual workers
- High cognitive complexity

**Recommendation**:
```python
# whisper_backend.py refactor
class WhisperVADWorker:
    """Processes audio from queue and splits into VAD-based chunks."""

    def __init__(self, audio_source: AudioSource, vad_config: VADConfig, ...):
        self.audio_source = audio_source
        self.vad = WebRTCVAD(...)
        self.buffer = np.zeros(0, dtype=np.float32)
        # ... other state

    async def run(self, transcription_queue: asyncio.Queue) -> None:
        """Main worker loop."""
        # Current vad_worker logic here (cleaner without nonlocal)

class WhisperTranscriberWorker:
    """Transcribes audio chunks using Whisper model."""

    def __init__(self, model: whisper.Whisper, config: TranscriptionConfig, ...):
        self.model = model
        # ... other state

    async def run(self, transcription_queue: asyncio.Queue) -> None:
        """Main worker loop."""
        # Current transcriber_worker logic here
```

**Benefits**:
- Each worker is independently testable
- State is explicit (instance variables) not implicit (nonlocal)
- Easier to add new worker types
- Reduces function complexity (nested functions → classes)

**Effort**: Medium (8-10 hours for both backends)
**Impact**: High (major maintainability improvement)

---

### 3. Consolidate CLI Orchestration Logic

**Current State**: `main()` function (cli.py:656) is 237 lines with duplicated backend setup logic.

**Issues**:
- Whisper backend setup: lines 716-793 (77 lines)
- Realtime backend setup: lines 794-884 (90 lines)
- Similar finalization logic repeated twice
- Difficult to follow control flow

**Recommendation**:
```python
# cli.py refactor
class TranscriptionOrchestrator:
    """Orchestrates transcription session lifecycle."""

    def __init__(self, config: TranscriptionConfig):
        self.config = config
        self.session_logger = SessionLogger(...)
        self.collector = ChunkCollectorWithStitching(sys.stdout)

    def run_whisper_session(self) -> None:
        """Run Whisper backend session."""
        try:
            result = run_whisper_transcriber(...)
        finally:
            self._finalize_session(result)

    def run_realtime_session(self) -> None:
        """Run Realtime backend session."""
        try:
            result = run_realtime_transcriber(...)
        finally:
            self._finalize_session(result)

    def _finalize_session(self, result) -> None:
        """Common finalization logic (DRY)."""
        # Lines 748-793 and 828-884 consolidated here

def main(argv: list[str]) -> None:
    config = TranscriptionConfig.from_flags(FLAGS)
    orchestrator = TranscriptionOrchestrator(config)

    if FLAGS.backend == "whisper":
        orchestrator.run_whisper_session()
    else:
        orchestrator.run_realtime_session()
```

**Benefits**:
- Eliminates 50+ lines of duplication
- Clearer separation of concerns
- Easier to add new backends
- Testable orchestration logic

**Effort**: Medium (6-8 hours)
**Impact**: High (major readability improvement)

---

### 4. Extract Audio Utilities Module

**Current State**: Audio conversion/resampling logic duplicated across:
- `realtime_backend.py`: `float_to_pcm16()` (41), `resample_audio()` (48)
- `file_audio_source.py`: `_resample()` (260)
- `session_logger.py`: `_save_audio()` (299)

**Recommendation**:
```python
# src/transcribe_demo/audio_utils.py
import numpy as np

def float32_to_pcm16(audio: np.ndarray) -> bytes:
    """Convert float32 audio to PCM16 bytes."""
    # Consolidate from realtime_backend.py:41-45

def resample_audio(audio: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    """Resample audio using linear interpolation."""
    # Consolidate from realtime_backend.py:48-57 and file_audio_source.py:260-283

def save_audio_file(
    audio: np.ndarray,
    path: Path,
    sample_rate: int,
    format: str = "wav"
) -> None:
    """Save audio to file in specified format (wav/flac)."""
    # Consolidate from session_logger.py:299-328
```

**Benefits**:
- Single source of truth for audio operations
- Easier to add format support (e.g., mp3, opus)
- Reduces code by ~100 lines
- Centralized testing of audio operations

**Effort**: Low (3-4 hours)
**Impact**: Medium (code reduction, better testability)

---

## Medium-Priority Design Improvements

### 5. Extract Diff/Comparison Module

**Current State**: Transcription comparison logic in cli.py:
- `compute_transcription_diff()` (490)
- `print_transcription_summary()` (508)
- `_tokenize_with_original()` (571)
- `_colorize_token()` (582)
- `_format_diff_snippet()` (588)
- `_generate_diff_snippets()` (629)

**Issues**:
- ~200 lines of specialized logic mixed with CLI code
- Hard to reuse for other tools (e.g., testing, analysis)
- No unit tests for diff algorithm

**Recommendation**:
```python
# src/transcribe_demo/transcription_diff.py
@dataclass
class TranscriptionDiff:
    """Represents differences between two transcriptions."""
    similarity: float
    snippets: list[DiffSnippet]

    def format_for_display(self, use_color: bool = True) -> str:
        """Format diff for terminal display."""

    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization."""

def compare_transcriptions(
    text_a: str,
    text_b: str,
    tokenizer: Callable[[str], list[str]] | None = None
) -> TranscriptionDiff:
    """Compare two transcriptions and return detailed diff."""
    # Current _generate_diff_snippets logic
```

**Benefits**:
- Reusable across CLI, tests, and analysis tools
- Easier to unit test diff algorithm
- Cleaner cli.py (removes ~150 lines)
- Can swap tokenization strategies

**Effort**: Low (3-4 hours)
**Impact**: Medium (better separation of concerns)

---

### 6. Create VAD Abstraction Layer

**Current State**: `WebRTCVAD` class (whisper_backend.py:197) is the only VAD implementation, but TODO comments suggest adding Silero VAD (whisper_backend.py:146-195).

**Recommendation**:
```python
# src/transcribe_demo/vad/__init__.py
from abc import ABC, abstractmethod

class VADBackend(ABC):
    """Abstract base class for Voice Activity Detection backends."""

    @abstractmethod
    def is_speech(self, audio: np.ndarray) -> bool:
        """Detect if audio frame contains speech."""

    @property
    @abstractmethod
    def frame_size(self) -> int:
        """Required frame size in samples."""

# src/transcribe_demo/vad/webrtc.py
class WebRTCVAD(VADBackend):
    """WebRTC VAD implementation."""
    # Current implementation

# src/transcribe_demo/vad/silero.py (future)
class SileroVAD(VADBackend):
    """Silero VAD implementation (better for noisy environments)."""
    # Implementation from TODO comment
```

**Benefits**:
- Enables easy VAD backend switching
- Prepares for Silero VAD implementation (mentioned in TODO)
- Can A/B test different VAD algorithms
- Better testability (mock VAD for tests)

**Effort**: Low (2-3 hours)
**Impact**: Medium (enables future VAD improvements)

---

### 7. Formalize ChunkConsumer Protocol

**Current State**: Chunk consumer callbacks have inconsistent signatures:
- Whisper backend: `chunk_consumer(chunk_index, text, absolute_start, absolute_end, inference_seconds, is_partial)`
- Realtime backend: `chunk_consumer(chunk_index=..., text=..., absolute_start=..., ...)` (keyword args)

**Recommendation**:
```python
# src/transcribe_demo/types.py
from typing import Protocol

@dataclass
class TranscriptionChunk:
    """Represents a single transcribed chunk."""
    index: int
    text: str
    start_time: float
    end_time: float
    inference_seconds: float | None
    is_partial: bool = False
    audio: np.ndarray | None = None

class ChunkConsumer(Protocol):
    """Protocol for consuming transcription chunks."""

    def __call__(self, chunk: TranscriptionChunk) -> None:
        """Process a transcription chunk."""
```

**Benefits**:
- Type-safe chunk consumer interface
- Easier to pass additional metadata (e.g., confidence scores)
- Consistent signature across backends
- Better IDE autocomplete

**Effort**: Low (2-3 hours)
**Impact**: Medium (better API design)

---

## Code Quality and Maintainability

### 8. Extract Configuration Dataclasses

**Current State**: 50+ FLAGS definitions (cli.py:27-251) passed as individual arguments to backend functions.

**Recommendation**:
```python
# src/transcribe_demo/config.py
from dataclasses import dataclass

@dataclass
class VADConfig:
    """VAD-related configuration."""
    aggressiveness: int = 2
    min_silence_duration: float = 0.2
    min_speech_duration: float = 0.25
    speech_pad_duration: float = 0.2
    max_chunk_duration: float = 60.0

@dataclass
class WhisperConfig:
    """Whisper backend configuration."""
    model: str = "turbo"
    device: str = "auto"
    require_gpu: bool = False
    language: str = "en"
    enable_partial_transcription: bool = False
    partial_model: str = "base.en"
    vad: VADConfig = field(default_factory=VADConfig)

@dataclass
class RealtimeConfig:
    """Realtime API configuration."""
    model: str = "gpt-realtime-mini"
    endpoint: str = "wss://api.openai.com/v1/realtime"
    vad_threshold: float = 0.2
    vad_silence_duration_ms: int = 100
```

**Benefits**:
- Reduces function signatures from 20+ params to 2-3
- Type-checked configuration
- Easy to serialize/deserialize (JSON, YAML)
- Clear configuration hierarchy

**Effort**: Medium (4-5 hours)
**Impact**: High (major API improvement)

---

### 9. Reduce Nested Function Complexity

**Current State**: Several functions have high cyclomatic complexity:
- `_playback_loop()` (file_audio_source.py:285): 60+ lines, multiple conditionals
- `_audio_callback()` (audio_capture.py:67): nested conditionals
- `vad_worker()` (whisper_backend.py:376): 140+ lines

**Recommendation**: Extract helper methods from nested functions:
```python
# file_audio_source.py refactor
class FileAudioSource:
    def _playback_loop(self) -> None:
        """Feed audio chunks into the queue."""
        for frame_idx in range(self._get_total_frames()):
            if self.stop_event.is_set():
                break

            frame = self._get_frame(frame_idx)
            self._enqueue_frame(frame)
            self._check_duration_limit()
            self._sleep_for_realtime(frame_idx)

        self._signal_end_of_stream()

    def _get_frame(self, frame_idx: int) -> np.ndarray:
        """Extract single frame from loaded audio."""
        # Extract logic

    def _check_duration_limit(self) -> None:
        """Check if capture duration limit reached."""
        # Extract logic
```

**Benefits**:
- Each method does one thing (SRP)
- Easier to test individual behaviors
- Improved readability
- Lower cognitive load

**Effort**: Medium (6-8 hours)
**Impact**: Medium (better maintainability)

---

### 10. Consolidate Session Finalization Logic

**Current State**: Similar finalization code in:
- cli.py:748-793 (Whisper finalization)
- cli.py:828-884 (Realtime finalization)
- session_replay.py:413-425 (Whisper retranscription)
- session_replay.py:468-475 (Realtime retranscription)

**Recommendation**:
```python
# cli.py or new orchestrator module
def finalize_session(
    collector: ChunkCollectorWithStitching,
    result: TranscriptionResult,
    session_logger: SessionLogger,
    config: SessionConfig,
) -> None:
    """Finalize session with common logic for all backends."""
    final = collector.get_final_stitched()

    # Update cleaned chunk text
    for chunk_index, cleaned_text in collector.get_cleaned_chunks():
        session_logger.update_chunk_cleaned_text(chunk_index, cleaned_text)

    # Compute diff if available
    similarity, diff_snippets = None, None
    if result.full_audio_transcription:
        similarity, diff_snippets = compute_transcription_diff(
            final, result.full_audio_transcription
        )

    # Finalize logger
    session_logger.finalize(
        capture_duration=result.capture_duration,
        full_audio_transcription=result.full_audio_transcription,
        stitched_transcription=final,
        extra_metadata=result.metadata,
        min_duration=config.min_log_duration,
        transcription_similarity=similarity,
        transcription_diffs=diff_snippets,
    )

    # Display results
    if config.compare_transcripts:
        print_transcription_summary(sys.stdout, final, result.full_audio_transcription)
    else:
        _print_final_stitched(sys.stdout, final)
```

**Benefits**:
- Eliminates ~100 lines of duplication
- Single source of truth for finalization
- Easier to add new finalization steps

**Effort**: Low (2-3 hours)
**Impact**: Medium (DRY principle)

---

## Performance and Efficiency

### 11. Lazy Load Heavy Dependencies

**Current State**: All imports at module level, including heavy ones:
- `torch` (whisper_backend.py:15)
- `whisper` (whisper_backend.py:17)
- `sounddevice` (audio_capture.py:17 - already lazy!)

**Recommendation**: Lazy load Whisper and Torch when needed:
```python
# whisper_backend.py
# Remove top-level imports
# import torch
# import whisper

def load_whisper_model(...) -> tuple[Any, str, bool]:
    """Load Whisper model (lazy imports torch/whisper)."""
    import torch
    import whisper

    # Rest of function
```

**Benefits**:
- Faster CLI startup for `--help`, version info
- Realtime backend doesn't load unused Whisper deps
- Reduced memory if only using one backend

**Effort**: Low (1-2 hours)
**Impact**: Low (startup time optimization)

---

### 12. Optimize Buffer Operations

**Current State**: Multiple buffer concatenations in hot paths:
- whisper_backend.py:401: `buffer = np.concatenate((buffer, mono))`
- whisper_backend.py:402: `speech_pad_buffer = np.concatenate((speech_pad_buffer, mono))`

**Recommendation**: Use ring buffer or pre-allocated arrays for frequently updated buffers:
```python
class RingBuffer:
    """Efficient circular buffer for audio data."""

    def __init__(self, capacity: int):
        self._buffer = np.zeros(capacity, dtype=np.float32)
        self._write_pos = 0
        self._size = 0

    def append(self, data: np.ndarray) -> None:
        """Append data to buffer (O(1) when not full)."""
        # Efficient ring buffer implementation
```

**Benefits**:
- Reduces memory allocations in hot path
- Better cache locality
- Constant-time append operations

**Effort**: Medium (4-5 hours)
**Impact**: Low-Medium (performance in long sessions)

---

## Testing and Testability

### 13. Extract Test Fixtures to Shared Module

**Current State**: Test helpers duplicated across test files:
- `_generate_synthetic_audio()` appears in multiple test files
- `FakeAudioCaptureManager` duplicated in conftest.py and session_replay.py

**Recommendation**:
```python
# tests/fixtures/audio.py
def generate_synthetic_audio(
    duration_seconds: float,
    sample_rate: int = 16000,
    frequency: float = 440.0,
) -> np.ndarray:
    """Generate synthetic audio for testing."""
    # Consolidate from all test files

# tests/fixtures/audio_source.py
class FakeAudioCaptureManager:
    """Reusable fake audio source for testing."""
    # Consolidate from conftest.py and session_replay.py
```

**Benefits**:
- DRY principle for test code
- Consistent test fixtures
- Easier to enhance test utilities

**Effort**: Low (2-3 hours)
**Impact**: Medium (test maintainability)

---

### 14. Add Property-Based Tests for Audio Operations

**Current State**: Audio utilities tested with specific examples only.

**Recommendation**: Use Hypothesis for property-based testing:
```python
# tests/test_audio_utils.py
from hypothesis import given, strategies as st

@given(
    audio=st.lists(st.floats(min_value=-1.0, max_value=1.0), min_size=100),
    from_rate=st.integers(min_value=8000, max_value=48000),
    to_rate=st.integers(min_value=8000, max_value=48000),
)
def test_resample_preserves_duration(audio, from_rate, to_rate):
    """Resampling preserves audio duration within tolerance."""
    audio_arr = np.array(audio, dtype=np.float32)
    resampled = resample_audio(audio_arr, from_rate, to_rate)

    expected_duration = len(audio) / from_rate
    actual_duration = len(resampled) / to_rate

    assert abs(expected_duration - actual_duration) < 0.01
```

**Benefits**:
- Catches edge cases not covered by examples
- Better confidence in audio operations
- Documents expected properties

**Effort**: Low (2-3 hours)
**Impact**: Medium (test coverage)

---

## Documentation and Type Safety

### 15. Add Comprehensive Type Annotations

**Current State**: Some functions lack complete type hints:
- `_audio_callback()` (audio_capture.py:67): `status: Any`
- Various callback types use `Any` instead of specific types

**Recommendation**: Use specific types and Protocols:
```python
# audio_capture.py
from ctypes import Structure
import sounddevice as sd

def _audio_callback(
    self,
    indata: np.ndarray,
    frames: int,
    time_info: sd.CallbackTimeInfo,  # More specific than Structure
    status: sd.CallbackFlags,        # More specific than Any
) -> None:
    """Audio callback invoked by sounddevice."""
```

**Benefits**:
- Better type checking with Pyright
- Improved IDE autocomplete
- Self-documenting code

**Effort**: Low (2-3 hours)
**Impact**: Low-Medium (code quality)

---

### 16. Add Module-Level Docstrings

**Current State**: Most modules lack module-level docstrings explaining their purpose.

**Recommendation**: Add comprehensive module docstrings:
```python
# whisper_backend.py
"""
Whisper backend for local transcription with VAD-based chunking.

This module provides real-time audio transcription using OpenAI's Whisper model
running locally (CPU/GPU). It uses WebRTC VAD to intelligently chunk audio at
natural speech pauses.

Key Components:
- WebRTCVAD: Voice activity detection for intelligent chunking
- load_whisper_model(): Model loading with device auto-detection
- run_whisper_transcriber(): Main transcription loop with async workers

See DESIGN.md for architectural decisions and VAD chunking strategy.
"""
```

**Benefits**:
- Better navigability for new contributors
- Clear module boundaries
- Links to relevant documentation

**Effort**: Low (1-2 hours)
**Impact**: Low (documentation)

---

## Summary and Prioritization

### Recommended Implementation Order

**Phase 1: Foundation (1-2 weeks)**
1. Extract AudioSource Protocol (enables all audio source improvements)
2. Extract Audio Utilities Module (reduces duplication immediately)
3. Extract Configuration Dataclasses (improves all function signatures)
4. Extract Diff/Comparison Module (cleaner cli.py)

**Phase 2: Architecture (2-3 weeks)**
5. Consolidate CLI Orchestration Logic
6. Extract Backend Worker Classes
7. Consolidate Session Finalization Logic
8. Create VAD Abstraction Layer

**Phase 3: Polish (1 week)**
9. Formalize ChunkConsumer Protocol
10. Extract Test Fixtures
11. Reduce Nested Function Complexity
12. Add Type Annotations

**Phase 4: Optional Enhancements**
13. Lazy Load Dependencies
14. Property-Based Tests
15. Module Docstrings
16. Buffer Optimizations

### Effort Summary

| Priority | Effort | Impact | Count |
|----------|--------|--------|-------|
| High | Low | High | 2 |
| High | Medium | High | 3 |
| Medium | Low | Medium | 5 |
| Medium | Medium | Medium | 3 |
| Low | Low | Low-Medium | 3 |

**Total Estimated Effort**: 8-12 weeks for full implementation

---

## Risks and Considerations

### Refactoring Risks

1. **Breaking Changes**: Many refactorings change public APIs
   - Mitigation: Use deprecation warnings, maintain backward compatibility

2. **Test Coverage**: Need comprehensive tests before major refactors
   - Current coverage: Good for backends, could improve for CLI
   - Mitigation: Add integration tests for key workflows

3. **Coordination**: Multiple refactorings touch same files
   - Mitigation: Implement in order listed (dependencies first)

### When NOT to Refactor

- If adding urgent features or fixing critical bugs
- During feature freeze periods
- If changing underlying libraries (e.g., Whisper API changes)

---

## Conclusion

The transcribe_demo codebase is well-structured with good separation between backends and clear module boundaries. The main opportunities lie in:

1. **Reducing duplication** (orchestration, finalization, audio utils)
2. **Formalizing interfaces** (AudioSource, ChunkConsumer, VAD)
3. **Simplifying complex functions** (worker classes, configuration)

These refactorings would improve maintainability and testability while preparing the codebase for future extensions (Silero VAD, new backends, etc.).

**Next Steps**:
1. Review this document with team
2. Prioritize based on current roadmap
3. Create GitHub issues for approved refactorings
4. Implement Phase 1 (foundation) first
