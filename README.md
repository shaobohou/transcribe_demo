# Transcribe Demo

Real-time microphone transcription tool with support for two backends:
- **Local Whisper**: OpenAI's Whisper model running locally (CPU/GPU)
- **Realtime API**: Stream audio to OpenAI's Realtime API for lower-latency transcription

Built with the `uv` Python project manager.

## Features

- **Smart Voice Activity Detection (VAD)**: Automatically detects natural speech pauses using WebRTC VAD for intelligent chunking
- **GPU Acceleration**: Auto-detects CUDA or Apple Metal (MPS) for faster inference
- **Intelligent Stitching**: Cleans up punctuation at chunk boundaries for smooth transcription flow
- **Dual Backend Support**: Choose between local Whisper or cloud-based Realtime API
- **File & URL Support**: Simulate live transcription from audio files or URLs instead of microphone
- **Transcript Comparison**: Compares chunked vs. full-audio transcription with detailed diff output
- **Automatic Session Logging**: Every session is saved to disk with audio, transcriptions, and metadata
- **Flexible Configuration**: Fine-tune VAD sensitivity, chunk duration, capture limits, and more
- **Safety Features**: Automatic capture duration limits, memory warnings, and cost confirmations

## Setup

```bash
uv sync
```

### CPU-only environments (CI, sandboxes)

Fresh CI machines and other ephemeral environments can avoid downloading CUDA
artifacts by using the dedicated CPU workspace found in `ci/pyproject.toml`:

```bash
uv sync --project ci --refresh
# Run transcribe-demo in CPU-only mode
uv --project ci run transcribe-demo --audio_file audio.mp3
# Run tooling/tests against the CPU env
uv --project ci run ruff check
uv --project ci run python -m pytest
```

**Important**: All `uv run transcribe-demo` examples in this README will download
CUDA packages on fresh environments. In sandboxes/CI, **always use**
`uv --project ci run transcribe-demo` instead to avoid gigabyte downloads.

This workspace pins `torch==2.9.0+cpu`, installs the `vendor/triton-cpu-stub`
package, and pulls from the PyTorch CPU wheel index so dependency resolution
remains lightweight. Local development should keep using the default
`uv sync`, which resolves the official PyPI `torch` and `triton` wheels for full
GPU support.

## Usage

### Local Whisper Backend (Default)

The default backend uses Whisper's `turbo` model with VAD-based chunking:

```bash
uv run transcribe-demo
```

#### Model Selection

The default model is `turbo` (1.51GB), which requires a GPU for real-time performance.

**For CPU-only environments, use `base.en`:**

```bash
# Recommended for CI/testing and production without GPU
uv --project ci run transcribe-demo --model base.en
```

**Model recommendations:**
- **`base.en` (139MB)** - **STRONGLY RECOMMENDED for CPU-only**
  - 2.0x faster than real-time on CPU
  - 11x smaller download than turbo
  - Tested on 280s NPR newscast with good results
  - Use for: CI/testing, production without GPU, resource-constrained systems

- **`turbo` (1.51GB)** - Default for GPU
  - Highest accuracy
  - Requires GPU for real-time performance

- **NOT RECOMMENDED:**
  - `tiny.en` (72MB): Produces nonsensical errors ("stomp" vs "stop", "cliff face penalties")
  - `small.en` (461MB): Slower than real-time on CPU, marginal improvement over base.en

```bash
# Other models (for GPU use or experimentation)
uv run transcribe-demo --model small.en
uv run transcribe-demo --model tiny.en
```

#### VAD Configuration

Fine-tune voice activity detection for your environment:

**For responsive transcription with faster results:**
```bash
# Get 2-3x faster first results with minimal accuracy loss
uv run transcribe-demo --max_chunk_duration 20 --vad_min_silence_duration 0.6
```

**Standard VAD tuning options:**
```bash
# More aggressive pause detection (higher = more aggressive, 0-3)
uv run transcribe-demo --vad_aggressiveness 3

# Minimum silence duration to split chunks (default: 0.2s)
# Increase for slower chunking, decrease for faster response
uv run transcribe-demo --vad_min_silence_duration 0.5

# Minimum speech duration before transcribing (default: 0.25s)
uv run transcribe-demo --vad_min_speech_duration 0.5

# Padding before speech to avoid cutting words (default: 0.2s)
uv run transcribe-demo --vad_speech_pad_duration 0.3

# Maximum chunk duration (default: 60s)
# Increase if seeing duration warnings during long speech
uv run transcribe-demo --max_chunk_duration 90
```

**Tuning guide:**
- Increase `vad_aggressiveness` if missing speech
- Decrease `vad_aggressiveness` if capturing background noise
- Increase `vad_min_silence_duration` for slower, more deliberate chunking
- Decrease `max_chunk_duration` for more responsive transcription

### Realtime API Backend

For lower-latency transcription, stream audio to the OpenAI Realtime API:

```bash
export OPENAI_API_KEY=sk-...
uv run transcribe-demo --backend realtime
```

Customize the realtime backend:

```bash
uv run transcribe-demo --backend realtime \
  --realtime_model gpt-realtime-mini \
  --realtime_instructions "Your custom transcription instructions"
```

### GPU Support

The tool automatically detects and prefers GPU acceleration:

```bash
# Force specific device
uv run transcribe-demo --device cuda   # NVIDIA CUDA GPU
uv run transcribe-demo --device mps    # Apple Metal (Apple Silicon)
uv run transcribe-demo --device cpu    # CPU only

# Abort if no GPU available (don't fall back to CPU)
uv run transcribe-demo --require_gpu
```

### SSL/Certificate Handling

For corporate proxies with self-signed certificates:

```bash
# Provide custom certificate bundle
uv run transcribe-demo --ca_cert /path/to/corp-ca.pem

# Disable SSL certificate verification (insecure, use as last resort)
# Bypasses certificate issues for model downloads and Realtime API connections
uv run transcribe-demo --disable_ssl_verify
```

### Transcript Comparison & Capture Duration

Control how long the session runs and whether to compare transcriptions:

```bash
# Set max capture duration (default: 120s)
uv run transcribe-demo --max_capture_duration 300

# Unlimited duration
uv run transcribe-demo --max_capture_duration 0

# Disable transcript comparison (saves memory and API costs)
uv run transcribe-demo --nocompare_transcripts

# Shorter session without comparison
uv run transcribe-demo --max_capture_duration 60 --nocompare_transcripts
```

**Note**: Comparison is enabled by default. For Realtime API, this doubles API usage. See **[DESIGN.md](DESIGN.md#diff-tracking-design)** for rationale.

### Audio File Simulation

Instead of capturing from a microphone, you can simulate live transcription from audio files or URLs. This is useful for testing, development, or processing pre-recorded audio:

```bash
# Transcribe from local audio file
uv run transcribe-demo --audio_file path/to/audio.mp3

# Transcribe from URL (supports HTTP/HTTPS)
uv run transcribe-demo --audio_file http://example.com/audio.mp3

# Example: NPR newscast
uv run transcribe-demo --audio_file http://public.npr.org/anon.npr-mp3/npr/news/newscast.mp3

# Control playback speed (default: 1.0 = real-time)
uv run transcribe-demo --audio_file audio.wav --playback_speed 2.0    # 2x faster
uv run transcribe-demo --audio_file audio.mp3 --playback_speed 0.5    # 2x slower

# Works with all backends and configuration options
uv run transcribe-demo --audio_file audio.flac --backend realtime
uv run transcribe-demo --audio_file audio.wav --max_chunk_duration 30
```

**Supported formats**: WAV, FLAC, MP3, OGG, and any format supported by `libsndfile`.

**How it works**:
- Audio is loaded and resampled to target sample rate if needed
- Chunks are fed with realistic timing to simulate live capture
- URLs are automatically downloaded to temporary files (cleaned up after use)
- Playback speed can be adjusted for faster testing or slower analysis

### Language Control

Force a specific language or fall back to automatic detection:

```bash
# Default: English transcription
uv run transcribe-demo --language en

# Auto-detect spoken language
uv run transcribe-demo --language auto

# Force Spanish transcription
uv run transcribe-demo --language es
```

### Session Logging

Every transcription session is automatically saved to disk with comprehensive metadata. Sessions are organized by date and backend for easy browsing.

#### Directory Structure

```
session_logs/
└── 2025-11-06/                         # Organized by date
    ├── session_143052_whisper/         # Timestamp + backend name
    │   ├── session.json                # Complete metadata
    │   ├── full_audio.wav              # Raw mono audio (16kHz)
    │   ├── README.txt                  # Human-readable summary
    │   └── chunks/                     # Individual chunk audio files
    │       ├── chunk_000.wav
    │       ├── chunk_001.wav
    │       └── ...
    └── session_150234_realtime/
        └── ...
```

#### What Gets Logged

Each session includes:
- **Full audio**: Complete raw recording as WAV file
- **Chunk audio**: Individual audio segments for each transcribed chunk
- **Both transcriptions**: Stitched chunks + full audio transcription (for comparison)
- **Diff tracking**: Similarity scores and detailed diffs between stitched and complete transcriptions
- **Metadata**: Model, device, VAD parameters, timestamps, inference times
- **session.json**: Complete structured data in machine-readable JSON format
- **README.txt**: Human-readable summary with all transcriptions, diffs, and chunk details

See **[SESSION_LOGS.md](SESSION_LOGS.md)** for the complete session log format reference.

#### Configuration

```bash
# Change log directory (default: session_logs/)
uv run transcribe-demo --session_log_dir /path/to/logs

# Only save sessions longer than N seconds (default: 10.0s)
uv run transcribe-demo --min_log_duration 30

# Save very short sessions including test runs
uv run transcribe-demo --min_log_duration 0
```

**Note**: Sessions shorter than `--min_log_duration` are automatically discarded to avoid cluttering logs. See **[DESIGN.md](DESIGN.md#min-duration-filtering)** for rationale.

#### Session Replay Utility

List, inspect, and retranscribe previously logged sessions with different settings or backends:

```bash
# List all logged sessions
uv run transcribe-session --command=list

# Show details of a specific session
uv run transcribe-session --command=show --session_path=session_logs/2025-11-07/session_143052_whisper

# Retranscribe with different VAD settings
uv run transcribe-session --command=retranscribe \
  --session_path=session_logs/2025-11-07/session_143052_whisper \
  --model=small \
  --vad_aggressiveness=3

# Compare Whisper vs Realtime API
uv run transcribe-session --command=retranscribe \
  --session_path=session_logs/2025-11-07/session_143052_whisper \
  --retranscribe_backend=realtime
```

See **[SESSION_REPLAY.md](SESSION_REPLAY.md)** for complete documentation on listing, loading, and retranscribing sessions.

## How It Works

### VAD-Based Chunking

The Whisper backend uses WebRTC Voice Activity Detection (VAD) to intelligently chunk audio at natural speech pauses:

1. Audio is captured in real-time from your microphone
2. VAD analyzes 30ms frames to detect speech vs. silence
3. Chunks are created when silence is detected or max duration is reached
4. Whisper transcribes each chunk using the `--language` setting (English by default)
5. Chunks are intelligently stitched with automatic punctuation cleanup at boundaries

This minimizes transcription errors by chunking at natural pauses instead of arbitrary time intervals.

See **[DESIGN.md](DESIGN.md#vad-based-chunking-strategy)** for detailed design rationale and parameter defaults.

### Timeout Behavior

When the time limit is reached, capture stops immediately and all buffered audio is transcribed, including incomplete chunks. This ensures zero data loss—audio captured before the timeout is never discarded.

See **[DESIGN.md](DESIGN.md#timeout-behavior-design)** for implementation details.

### Output Format

Each chunk displays chunk number, timestamp, duration, inference time (Whisper only), and transcribed text. Stitched results are shown every 3 chunks and at session end.

See **[DESIGN.md](DESIGN.md#output-display-strategy)** for display strategy rationale.

## Testing

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_vad.py

# Verbose output
uv run pytest -v
```

## Architecture

- **main.py**: Entry point and orchestration, chunk collection, stitching, and comparison
- **whisper_backend.py**: Local Whisper transcription with WebRTC VAD
- **realtime_backend.py**: OpenAI Realtime API integration via WebSocket
- **audio_capture.py**: Microphone audio capture manager
- **file_audio_source.py**: File/URL audio source for simulation

See **[DESIGN.md](DESIGN.md#architecture-overview)** for detailed architecture and design decisions.

## Development

See [SITEMAP.md](SITEMAP.md) for a complete guide to all project documentation, including development workflow, architecture, and refactoring opportunities.
