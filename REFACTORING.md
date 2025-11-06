# Refactoring Opportunities

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
- Add tests to REFACTORING.md testing section
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

### 1.1 Extract Terminal Color Formatter (main.py)

**Issue**: ANSI color codes are hardcoded and duplicated in multiple locations.

**Locations**:
- `main.py:111-114` (ChunkCollectorWithStitching.__call__)
- `main.py:308-310` (main function - whisper backend)
- `main.py:342-344` (main function - realtime backend)

**Current Code Pattern**:
```python
if use_color:
    cyan = "\x1b[36m"
    green = "\x1b[32m"
    reset = "\x1b[0m"
    bold = "\x1b[1m"
    label_colored = f"{bold}{cyan}{label}{reset}"
```

**Proposed Solution**:
```python
class TerminalFormatter:
    """Handles terminal formatting with ANSI color codes."""

    def __init__(self, use_color: bool = True):
        self.use_color = use_color
        self._colors = {
            'cyan': '\x1b[36m',
            'green': '\x1b[32m',
            'reset': '\x1b[0m',
            'bold': '\x1b[1m',
        }

    def format_label(self, text: str, color: str, bold: bool = True) -> str:
        if not self.use_color:
            return text
        style = self._colors['bold'] if bold else ''
        return f"{style}{self._colors[color]}{text}{self._colors['reset']}"

    def format_chunk_label(self, label: str) -> str:
        return self.format_label(label, 'cyan', bold=True)

    def format_concat_label(self, label: str) -> str:
        return self.format_label(label, 'green', bold=True)
```

**Benefits**:
- Eliminates code duplication
- Centralizes color management
- Easy to add new colors or styles
- Testable in isolation

---

### 1.2 Extract Device Selection Logic (whisper_backend.py)

**Issue**: Device detection and selection logic (32 lines) is embedded in `run_whisper_transcriber()`.

**Location**: `whisper_backend.py:163-194`

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

---

### 1.3 Extract Duplicated Final Result Display (main.py)

**Issue**: Final concatenated result printing is duplicated identically in two places.

**Locations**:
- `main.py:303-313` (whisper backend)
- `main.py:337-347` (realtime backend)

**Proposed Solution**:
```python
def _print_final_result(
    collector: ChunkCollectorWithStitching,
    stream: TextIO = sys.stdout
) -> None:
    """Print the final concatenated transcription."""
    final = collector.get_final_stitched()
    if not final:
        return

    use_color = getattr(stream, "isatty", lambda: False)()
    if use_color:
        green = "\x1b[32m"
        reset = "\x1b[0m"
        bold = "\x1b[1m"
        label = f"\n{bold}{green}[FINAL CONCATENATED]{reset}"
    else:
        label = "\n[FINAL CONCATENATED]"

    print(f"{label} {final}\n", file=stream)

# Usage in main():
finally:
    _print_final_result(collector)
```

**Even Better**: Use the TerminalFormatter from 1.1:
```python
def _print_final_result(
    collector: ChunkCollectorWithStitching,
    stream: TextIO = sys.stdout
) -> None:
    """Print the final concatenated transcription."""
    final = collector.get_final_stitched()
    if not final:
        return

    formatter = TerminalFormatter(getattr(stream, "isatty", lambda: False)())
    label = formatter.format_concat_label("[FINAL CONCATENATED]")
    print(f"\n{label} {final}\n", file=stream)
```

**Benefits**:
- DRY principle
- Single point of maintenance
- Consistent formatting

---

### 1.4 Extract SSL Context Configuration (Both Backends)

**Issue**: SSL context setup is duplicated in both backends.

**Locations**:
- `whisper_backend.py:206-229`
- `realtime_backend.py:190-194`

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

### 2.1 Decompose `run_whisper_transcriber()` (whisper_backend.py)

**Issue**: Function is 258 lines with multiple responsibilities.

**Location**: `whisper_backend.py:147-405`

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
        self._concat_frequency = 3  # Show concatenated every N chunks

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

        if self._should_show_concatenated(chunk_index):
            self._display_concatenated()

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

    def _should_show_concatenated(self, chunk_index: int) -> bool:
        """Check if we should display concatenated result."""
        return (chunk_index + 1) % self._concat_frequency == 0

    def _display_concatenated(self) -> None:
        """Display concatenated result of all chunks so far."""
        cleaned_chunks = [
            self._clean_chunk_text(c.text, is_final_chunk=(i == len(self._chunks) - 1))
            for i, c in enumerate(self._chunks)
        ]
        concatenated = " ".join(chunk for chunk in cleaned_chunks if chunk)

        label = self._formatter.format_concat_label("[CONCATENATED]")
        self._stream.write(f"\n{label} {concatenated}\n\n")
        self._stream.flush()
```

**Benefits**:
- Each method has a single clear purpose
- Easier to test individual formatting logic
- More maintainable
- Reduced cyclomatic complexity

---

### 2.3 Decompose `run_realtime_transcriber()` (realtime_backend.py)

**Issue**: Function is 200 lines with 3 nested async functions.

**Location**: `realtime_backend.py:47-247`

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
        self.partials: Dict[str, str] = {}

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

## Priority 3: Configuration and Constants

### 3.1 Extract Magic Numbers to Named Constants

**Issue**: Magic numbers scattered throughout the code make intent unclear.

**Examples**:
- `main.py:125`: `(chunk_index + 1) % 3 == 0` - concatenation frequency
- `main.py:16`: `REALTIME_CHUNK_DURATION = 2.0` - good! (already done)
- `whisper_backend.py:241`: `2.0` - minimum chunk duration
- `whisper_backend.py:277`: `30` - VAD frame duration
- `realtime_backend.py:60`: `24000` - session sample rate
- `realtime_backend.py:103`: `100ms` - minimum audio (in comment)

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
REALTIME_MIN_AUDIO_MS = 100

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

### 4.2 Add Unit Tests for Missing Components

**Missing Test Coverage**:
- `ChunkCollectorWithStitching` (main.py)
- `_clean_chunk_text()` method
- Device selection logic
- SSL configuration
- Audio resampling (`resample_audio` in realtime_backend.py)
- PCM conversion (`float_to_pcm16` in realtime_backend.py)

**Proposed Tests**:
```python
# test_chunk_collector.py
def test_clean_chunk_text_removes_trailing_comma():
    text = "Hello, world,"
    cleaned = ChunkCollectorWithStitching._clean_chunk_text(text, is_final_chunk=False)
    assert cleaned == "Hello, world"

def test_clean_chunk_text_keeps_question_mark():
    text = "How are you?"
    cleaned = ChunkCollectorWithStitching._clean_chunk_text(text, is_final_chunk=False)
    assert cleaned == "How are you?"

def test_clean_chunk_text_preserves_final_punctuation():
    text = "Final sentence."
    cleaned = ChunkCollectorWithStitching._clean_chunk_text(text, is_final_chunk=True)
    assert cleaned == "Final sentence."


# test_device_selection.py
def test_device_config_auto_prefers_cuda(monkeypatch):
    """Test that auto mode prefers CUDA when available."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    config = DeviceConfig.detect_device("auto")
    assert config.device == "cuda"
    assert config.requires_fp16 is True

def test_device_config_require_gpu_raises_on_cpu_only(monkeypatch):
    """Test that require_gpu raises when no GPU available."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="GPU required"):
        DeviceConfig.detect_device("auto", require_gpu=True)


# test_audio_utils.py
def test_resample_audio_upsampling():
    """Test upsampling from 8kHz to 16kHz."""
    audio_8k = np.array([0.0, 0.5, 1.0, 0.5], dtype=np.float32)
    resampled = resample_audio(audio_8k, from_rate=8000, to_rate=16000)
    assert len(resampled) == 8
    assert resampled.dtype == np.float32

def test_float_to_pcm16_clipping():
    """Test that values outside [-1, 1] are clipped."""
    audio = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    pcm = float_to_pcm16(audio)
    assert len(pcm) == 10  # 5 samples * 2 bytes each
```

**Benefits**:
- Catch regressions early
- Document expected behavior
- Enable refactoring with confidence

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


