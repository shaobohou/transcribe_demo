# Session Replay Utility

The session replay utility allows you to list, inspect, and retranscribe previously logged transcription sessions.

## Quick Start

```bash
# List all logged sessions
uv run transcribe-session --command=list

# Show details of a specific session
uv run transcribe-session --command=show --session_path=session_logs/2025-11-07/session_143052_whisper

# Retranscribe a session with different settings
uv run transcribe-session --command=retranscribe \
  --session_path=session_logs/2025-11-07/session_143052_whisper \
  --retranscribe_backend=whisper \
  --model=small \
  --vad_aggressiveness=3
```

## Session Completion Marker

All successfully finalized sessions contain a `.complete` marker file. This file is created as the last step of session finalization and indicates that:
- All audio files have been written
- All metadata has been saved
- The session was not interrupted

By default, the session replay utility only lists and loads complete sessions (those with the `.complete` marker). This protects against:
- Incomplete sessions from crashes or interruptions
- Partially written session data
- Corrupted session directories

To work with incomplete sessions, use the `--include_incomplete` flag for listing or `--allow_incomplete` flag for loading/retranscribing.

## Commands

### `list` - List All Sessions

List all available sessions with optional filtering.

**Basic usage:**
```bash
uv run transcribe-session --command=list
```

**Filter by backend:**
```bash
uv run transcribe-session --command=list --backend=whisper
uv run transcribe-session --command=list --backend=realtime
```

**Filter by date range:**
```bash
uv run transcribe-session --command=list --start_date=2025-11-01 --end_date=2025-11-07
```

**Filter by minimum duration:**
```bash
uv run transcribe-session --command=list --min_duration=60.0
```

**Show verbose details:**
```bash
uv run transcribe-session --command=list --verbose
```

**Include incomplete sessions:**
```bash
# Show all sessions including those without completion marker
uv run transcribe-session --command=list --include_incomplete
```

**Example output (compact):**
```
Found 5 session(s):

  session_143052_whisper                   | whisper  | 45.2s    | 8 chunks
  session_142830_realtime                  | realtime | 30.1s    | 15 chunks
  session_141500_whisper                   | whisper  | 120.5s   | 20 chunks
  session_140000_whisper                   | whisper  | 15.2s    | 3 chunks [INCOMPLETE]
```

Note: Sessions marked `[INCOMPLETE]` are missing the `.complete` marker and may be corrupted or partially written.

### `show` - Show Session Details

Display detailed information about a specific session.

**Usage:**
```bash
uv run transcribe-session --command=show --session_path=session_logs/2025-11-07/session_143052_whisper
```

**Load incomplete session:**
```bash
# Force loading session without completion marker
uv run transcribe-session --command=show \
  --session_path=session_logs/2025-11-07/session_143052_whisper \
  --allow_incomplete
```

**Example output:**
```
Session: session_143052_whisper
============================================================
Timestamp: 2025-11-07T14:30:52.123456
Backend: whisper
Duration: 45.20s
Sample Rate: 16000 Hz
Channels: 1
Total Chunks: 8
Model: turbo
Device: cuda
Language: en

VAD Parameters:
  Aggressiveness: 2
  Min Silence: 0.2s
  Min Speech: 0.25s

============================================================
STITCHED TRANSCRIPTION:
============================================================
Hello world. How are you today? I'm doing great thanks for asking...

============================================================
CHUNKS (8):
============================================================
  Chunk 000 [0.00s - 2.30s]: Hello world....
  Chunk 001 [2.30s - 4.10s]: How are you?...
  ... and 6 more chunks
```

### `retranscribe` - Retranscribe a Session

Retranscribe a logged session with different backend or settings.

**Basic usage (Whisper):**
```bash
uv run transcribe-session --command=retranscribe \
  --session_path=session_logs/2025-11-07/session_143052_whisper \
  --retranscribe_backend=whisper
```

**Use different Whisper model:**
```bash
uv run transcribe-session --command=retranscribe \
  --session_path=session_logs/2025-11-07/session_143052_whisper \
  --retranscribe_backend=whisper \
  --model=small \
  --device=cuda
```

**Adjust VAD settings:**
```bash
uv run transcribe-session --command=retranscribe \
  --session_path=session_logs/2025-11-07/session_143052_whisper \
  --retranscribe_backend=whisper \
  --vad_aggressiveness=3 \
  --vad_min_silence_duration=0.3
```

**Retranscribe with Realtime backend:**
```bash
export OPENAI_API_KEY=your_api_key_here
uv run transcribe-session --command=retranscribe \
  --session_path=session_logs/2025-11-07/session_143052_whisper \
  --retranscribe_backend=realtime \
  --realtime_model=gpt-realtime-mini
```

**Custom output directory:**
```bash
uv run transcribe-session --command=retranscribe \
  --session_path=session_logs/2025-11-07/session_143052_whisper \
  --output_dir=./retranscriptions \
  --retranscribe_backend=whisper \
  --model=turbo
```

## Command-Line Flags Reference

### Common Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--command` | (required) | Command to execute: `list`, `show`, or `retranscribe` |
| `--session_log_dir` | `./session_logs` | Directory containing session logs |

### List Command Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--backend` | None | Filter by backend (`whisper` or `realtime`) |
| `--start_date` | None | Filter sessions on or after this date (YYYY-MM-DD) |
| `--end_date` | None | Filter sessions on or before this date (YYYY-MM-DD) |
| `--min_duration` | None | Filter sessions with duration >= this value (seconds) |
| `--verbose` | False | Show detailed information |
| `--include_incomplete` | False | Include sessions without completion marker |

### Show/Retranscribe Command Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--session_path` | (required) | Path to session directory |
| `--allow_incomplete` | False | Allow loading sessions without completion marker |

### Retranscribe Command Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--retranscribe_backend` | `whisper` | Backend to use (`whisper` or `realtime`) |
| `--output_dir` | `./session_logs` | Output directory for results |
| `--model` | `turbo` | Whisper model (turbo, small, base, etc.) |
| `--device` | `auto` | Device (`auto`, `cpu`, `cuda`, `mps`) |
| `--language` | `en` | Language code |
| `--vad_aggressiveness` | 2 | VAD aggressiveness (0-3) |
| `--vad_min_silence_duration` | 0.2 | Min silence duration (seconds) |
| `--api_key` | $OPENAI_API_KEY | OpenAI API key (for realtime backend) |
| `--realtime_model` | `gpt-realtime-mini` | Realtime model name |
| `--audio_format` | `flac` | Audio format (`wav` or `flac`) |

## Use Cases

### Compare Different VAD Settings

Test how different VAD settings affect chunking:

```bash
# Original session with VAD aggressiveness 2
uv run transcribe-demo --backend=whisper --vad_aggressiveness=2

# Retranscribe with more aggressive VAD
uv run transcribe-session --command=retranscribe \
  --session_path=session_logs/2025-11-07/session_143052_whisper \
  --vad_aggressiveness=3 \
  --vad_min_silence_duration=0.1
```

### Test Different Models

Compare transcription quality across different Whisper models:

```bash
# Get the session path from listing
SESSION=session_logs/2025-11-07/session_143052_whisper

# Try turbo (fastest)
uv run transcribe-session --command=retranscribe --session_path=$SESSION --model=turbo

# Try small (more accurate)
uv run transcribe-session --command=retranscribe --session_path=$SESSION --model=small

# Try base (balanced)
uv run transcribe-session --command=retranscribe --session_path=$SESSION --model=base
```

### Compare Backends

Compare Whisper vs Realtime API:

```bash
SESSION=session_logs/2025-11-07/session_143052_whisper

# Retranscribe with Whisper
uv run transcribe-session --command=retranscribe \
  --session_path=$SESSION \
  --retranscribe_backend=whisper

# Retranscribe with Realtime API
uv run transcribe-session --command=retranscribe \
  --session_path=$SESSION \
  --retranscribe_backend=realtime
```

### Batch Processing

Process multiple sessions with a script:

```bash
#!/bin/bash
# retranscribe_all.sh - Retranscribe all whisper sessions with a different model

for session_dir in session_logs/*/session_*_whisper; do
  echo "Processing $session_dir..."
  uv run transcribe-session --command=retranscribe \
    --session_path="$session_dir" \
    --model=small \
    --output_dir=./retranscriptions_small
done
```

## Python API

You can also use the session replay functionality programmatically:

```python
from pathlib import Path
from transcribe_demo.session_replay import (
    list_sessions,
    load_session,
    retranscribe_session,
)

# List sessions (only complete ones by default)
sessions = list_sessions(
    log_dir="./session_logs",
    backend="whisper",
    min_duration=10.0,
)

# Include incomplete sessions
all_sessions = list_sessions(
    log_dir="./session_logs",
    include_incomplete=True,
)

for session in sessions:
    print(f"{session.session_id}: {session.capture_duration:.1f}s")

# Load a specific session (must be complete)
loaded = load_session("session_logs/2025-11-07/session_143052_whisper")
print(f"Loaded session with {loaded.metadata.total_chunks} chunks")
print(f"Audio duration: {loaded.metadata.capture_duration:.2f}s")

# Load incomplete session (use with caution)
incomplete_loaded = load_session(
    "session_logs/2025-11-07/session_incomplete",
    allow_incomplete=True,
)

# Retranscribe with different settings
result_path = retranscribe_session(
    loaded_session=loaded,
    output_dir="./retranscriptions",
    backend="whisper",
    backend_kwargs={
        "model": "small",
        "device": "cuda",
        "vad_aggressiveness": 3,
    },
)
print(f"Results saved to: {result_path}")
```

## Output Directory Structure

Retranscribed sessions are saved in the same format as original sessions:

```
session_logs/
└── YYYY-MM-DD/
    └── retranscribe_HHMMSS_<backend>_from_<original_session_id>/
        ├── .complete              # Completion marker
        ├── session.json           # New metadata and chunks
        ├── full_audio.flac        # Same audio as original
        ├── chunks/                # New chunk audio files
        │   ├── chunk_000.flac
        │   └── ...
        └── README.txt             # Human-readable summary
```

The retranscribed session directory name includes:
- Timestamp of retranscription
- Backend used for retranscription
- Reference to original session ID

This allows you to easily track which sessions are retranscriptions and compare them with originals.

## See Also

- **SESSION_LOGS.md**: Session log format specification
- **README.md**: User-facing documentation and usage examples
- **DESIGN.md**: Architecture and design rationale
- **CLAUDE.md**: Development workflow and critical implementation rules
