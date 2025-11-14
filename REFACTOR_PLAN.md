# Backend Class Refactoring Plan

## Goal
Simplify the `transcribe()` interface by absorbing configuration into backend and audio source classes.

## Current Architecture (Problems)
```python
transcribe(
    backend: str,  # 31 total parameters!
    model_name, sample_rate, channels, temp_file, ca_cert, ...
) -> Generator[TranscriptionChunk, None, Result]
```

## New Architecture (Solution)

### 1. Backend Protocol
```python
class TranscriptionBackend(Protocol):
    def run(
        self,
        audio_source: AudioSource,
        chunk_queue: queue.Queue[TranscriptionChunk | None],
    ) -> WhisperTranscriptionResult | RealtimeTranscriptionResult:
        ...
```

### 2. WhisperBackend Class
```python
@dataclass
class WhisperBackend:
    # Model config
    model_name: str = "turbo"
    device_preference: str = "auto"
    require_gpu: bool = False

    # VAD config
    vad_aggressiveness: int = 2
    vad_min_silence_duration: float = 0.2
    vad_min_speech_duration: float = 0.25
    vad_speech_pad_duration: float = 0.2
    max_chunk_duration: float = 60.0

    # Partial transcription config
    enable_partial_transcription: bool = False
    partial_model: str = "base.en"
    partial_interval: float = 1.0
    max_partial_buffer_seconds: float = 10.0

    # Session config
    language: str = "en"
    compare_transcripts: bool = True
    session_logger: SessionLogger | None = None
    min_log_duration: float = 0.0

    # SSL config
    ca_cert: Path | None = None
    disable_ssl_verify: bool = False
    temp_file: Path | None = None

    def run(self, audio_source, chunk_queue) -> WhisperTranscriptionResult:
        # Call the existing run_whisper_transcriber with self.* params
        ...
```

### 3. RealtimeBackend Class
```python
@dataclass
class RealtimeBackend:
    # API config
    api_key: str
    endpoint: str = "wss://api.openai.com/v1/realtime"
    model: str = "gpt-realtime-mini"
    instructions: str = "..."

    # VAD config
    vad_threshold: float = 0.3
    vad_silence_duration_ms: int = 200
    debug: bool = False

    # Session config
    language: str = "en"
    compare_transcripts: bool = True
    session_logger: SessionLogger | None = None
    min_log_duration: float = 0.0

    # SSL config
    disable_ssl_verify: bool = False

    def run(self, audio_source, chunk_queue) -> RealtimeTranscriptionResult:
        # Call the existing run_realtime_transcriber with self.* params
        ...
```

### 4. Simplified transcribe()
```python
def transcribe(
    backend: WhisperBackend | RealtimeBackend,
    audio_source: FileAudioSource | AudioCaptureManager,
) -> Generator[TranscriptionChunk, None, TranscriptionResult]:
    """
    Generator that yields transcription chunks.

    Args:
        backend: Configured backend instance
        audio_source: Configured audio source instance

    Yields:
        TranscriptionChunk objects

    Returns:
        Final transcription result
    """
    chunk_queue = queue.Queue()

    def worker():
        result = backend.run(audio_source, chunk_queue)
        result_container.append(result)
        chunk_queue.put(None)

    # ... rest of implementation
```

### 5. Updated main()
```python
def main(argv):
    # Create audio source
    if FLAGS.audio_file:
        audio_source = FileAudioSource(
            audio_file=FLAGS.audio_file,
            sample_rate=FLAGS.samplerate,
            channels=FLAGS.channels,
            max_capture_duration=FLAGS.max_capture_duration,
            collect_full_audio=FLAGS.compare_transcripts,
            playback_speed=FLAGS.playback_speed,
        )
    else:
        audio_source = AudioCaptureManager(
            sample_rate=FLAGS.samplerate,
            channels=FLAGS.channels,
            max_capture_duration=FLAGS.max_capture_duration,
            collect_full_audio=FLAGS.compare_transcripts,
        )

    # Create backend
    if FLAGS.backend == "whisper":
        backend = WhisperBackend(
            model_name=FLAGS.model,
            device_preference=FLAGS.device,
            require_gpu=FLAGS.require_gpu,
            vad_aggressiveness=FLAGS.vad_aggressiveness,
            # ... all whisper config
        )
    else:
        backend = RealtimeBackend(
            api_key=api_key,
            endpoint=FLAGS.realtime_endpoint,
            # ... all realtime config
        )

    # Run transcription
    for chunk in transcribe(backend, audio_source):
        collector(chunk)
```

## Benefits
1. **Clean separation**: Backend config in backend, audio config in audio source
2. **Type safety**: Backend classes are type-checkable
3. **Reusability**: Can create backend instances and reuse them
4. **Testability**: Easy to mock backend instances
5. **Simpler interface**: `transcribe(backend, audio_source)` vs 31 parameters!

## Implementation Steps
1. ✅ Create backend protocol in backend_protocol.py
2. Create WhisperBackend class wrapping run_whisper_transcriber
3. Create RealtimeBackend class wrapping run_realtime_transcriber
4. Update transcribe() to take backend + audio_source
5. Update main() to construct backends
6. Update tests
7. Commit and push
