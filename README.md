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
- **Transcript Comparison**: Compares chunked vs. full-audio transcription with detailed diff output
- **Automatic Session Logging**: Every session is saved to disk with audio, transcriptions, and metadata
- **Flexible Configuration**: Fine-tune VAD sensitivity, chunk duration, capture limits, and more
- **Safety Features**: Automatic capture duration limits, memory warnings, and cost confirmations

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

**Note**: When `--compare_transcripts` is enabled (default), the program captures audio for post-session comparison. For Realtime API, this doubles your API usage cost.

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

**Note**: Sessions shorter than `--min_log_duration` are automatically discarded to avoid cluttering logs with test runs.

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
3. Chunks are created when:
   - Minimum speech duration is reached (default 0.25s), AND
   - Continuous silence is detected (default 0.2s), OR
   - Maximum chunk duration is reached (default 60s)
4. Whisper transcribes each chunk using the `--language` setting (English by default)
5. Chunks are intelligently stitched with automatic punctuation cleanup at boundaries

This approach minimizes transcription errors by chunking at natural pauses instead of arbitrary time intervals.

### Timeout Behavior

**When the time limit is reached, capture stops immediately and all buffered audio is transcribed, including incomplete chunks.**

This ensures zero data loss—the chunk that pushes over the timeout limit is included in the transcription:

```
Time:  2.0s         2.1s         2.2s         2.3s (timeout)   2.4s
       |            |            |            |                |
Device: [chunk1] -> [chunk2] -> [chunk3] -> [chunk4]         [chunk5]
       |            |            |            |                |
Queue:  [put]       [put]       [put]       [put+None]       [rejected]
       |            |            |            |                |
Buffer: [add]       [add]       [add]       [add]
                                             [TRANSCRIBE ALL]
```

Audio arriving after the timeout (chunk5) is discarded, but all previously captured audio is transcribed immediately without waiting for VAD silence detection or minimum chunk size requirements.

### Output Format

Each chunk displays:
- Chunk number
- Absolute timestamp from session start
- Audio duration
- Inference time (Whisper backend only)
- Transcribed text

Stitched results are shown every 3 chunks and at the end of the session.

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

## Development

- See `CLAUDE.md` for key implementation rules and default values
- See `REFACTORING.md` for detailed refactoring opportunities and architectural improvements
