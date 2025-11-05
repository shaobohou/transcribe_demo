# Transcribe Demo

Real-time microphone transcription tool with support for two backends:
- **Local Whisper**: OpenAI's Whisper model running locally (CPU/GPU)
- **Realtime API**: Stream audio to OpenAI's Realtime API for lower-latency transcription

Built with the `uv` Python project manager.

## Features

- **Smart Voice Activity Detection (VAD)**: Automatically detects natural speech pauses using WebRTC VAD for intelligent chunking
- **GPU Acceleration**: Auto-detects CUDA or Apple Metal (MPS) for faster inference
- **Intelligent Concatenation**: Cleans up punctuation at chunk boundaries for smooth transcription flow
- **Dual Backend Support**: Choose between local Whisper or cloud-based Realtime API
- **Flexible Configuration**: Fine-tune VAD sensitivity, chunk duration, and more

## Setup

```bash
uv sync
```

## Usage

### Local Whisper Backend (Default)

The default backend uses Whisper's `turbo` model with VAD-based chunking:

```bash
uv run transcribe-demo
```

#### Model Selection

```bash
# Use different Whisper models
uv run transcribe-demo --model small.en
uv run transcribe-demo --model base.en
uv run transcribe-demo --model tiny.en
```

#### VAD Configuration

Fine-tune voice activity detection for your environment:

```bash
# More aggressive pause detection (higher = more aggressive, 0-3)
uv run transcribe-demo --vad-aggressiveness 3

# Minimum silence duration to split chunks (default: 0.2s)
uv run transcribe-demo --vad-min-silence-duration 0.5

# Minimum speech duration before transcribing (default: 0.25s)
uv run transcribe-demo --vad-min-speech-duration 0.5

# Padding before speech to avoid cutting words (default: 0.2s)
uv run transcribe-demo --vad-speech-pad-duration 0.3

# Maximum chunk duration (default: 60s)
uv run transcribe-demo --max-chunk-duration 90
```

### Realtime API Backend

For lower-latency transcription, stream audio to the OpenAI Realtime API:

```bash
export OPENAI_API_KEY=sk-...
uv run transcribe-demo --backend realtime
```

Customize the realtime backend:

```bash
uv run transcribe-demo --backend realtime \
  --realtime-model gpt-realtime-mini \
  --realtime-instructions "Your custom transcription instructions"
```

### GPU Support

The tool automatically detects and prefers GPU acceleration:

```bash
# Force specific device
uv run transcribe-demo --device cuda   # NVIDIA CUDA GPU
uv run transcribe-demo --device mps    # Apple Metal (Apple Silicon)
uv run transcribe-demo --device cpu    # CPU only

# Abort if no GPU available (don't fall back to CPU)
uv run transcribe-demo --require-gpu
```

### SSL/Certificate Handling

For corporate proxies with self-signed certificates:

```bash
# Provide custom certificate bundle
uv run transcribe-demo --ca-cert /path/to/corp-ca.pem

# Skip certificate verification (insecure, use as last resort)
uv run transcribe-demo --insecure-downloads
```

## How It Works

### VAD-Based Chunking

The Whisper backend uses WebRTC Voice Activity Detection (VAD) to intelligently chunk audio at natural speech pauses:

1. Audio is captured in real-time from your microphone
2. VAD analyzes 30ms frames to detect speech vs. silence
3. Chunks are created when:
   - Minimum speech duration is reached (default 0.25s), AND
   - Continuous silence is detected (default 0.2s), OR
   - Maximum chunk duration is reached (default 60s)
4. Whisper transcribes each chunk with `language="en"` to prevent foreign language hallucinations
5. Chunks are intelligently concatenated with automatic punctuation cleanup at boundaries

This approach minimizes transcription errors by chunking at natural pauses instead of arbitrary time intervals.

### Output Format

Each chunk displays:
- Chunk number
- Audio duration
- Inference time (Whisper backend only)
- Transcribed text

Concatenated results are shown every 3 chunks and at the end of the session.

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

- **main.py**: Entry point and orchestration, chunk collection and stitching
- **whisper_backend.py**: Local Whisper transcription with WebRTC VAD
- **realtime_backend.py**: OpenAI Realtime API integration via WebSocket

See `CLAUDE.md` for detailed architecture documentation and development guidelines.
