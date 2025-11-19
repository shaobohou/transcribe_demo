# Session Logging and Replay

This document describes the complete session logging system in transcribe-demo. Every transcription session is automatically saved with comprehensive metadata, audio files, and diff tracking information. You can also replay and retranscribe sessions with different settings.

## Quick Start

```bash
# Sessions are automatically logged during transcription
uv run transcribe-demo                # Creates session log automatically

# List all logged sessions
uv run transcribe-session --subcommand=list

# Show details of a specific session
uv run transcribe-session --subcommand=show --subcommand.session_path=session_logs/2025-11-07/session_143052_whisper

# Retranscribe with different settings
uv run transcribe-session --subcommand=retranscribe \
  --subcommand.session_path=session_logs/2025-11-07/session_143052_whisper \
  --subcommand.retranscribe_backend=whisper \
  --subcommand.whisper.model=small \
  --subcommand.whisper.vad.aggressiveness=3
```

---

## Table of Contents

1. [Session Log Format](#session-log-format)
2. [Session Replay Utility](#session-replay-utility)
3. [Using Session Data](#using-session-data)
4. [Configuration](#configuration)

---

## Session Log Format

### Directory Structure

```
session_logs/
└── YYYY-MM-DD/                           # Organized by date
    └── session_HHMMSS_<backend>/         # Timestamp + backend (whisper/realtime)
        ├── .complete                      # Completion marker (indicates successful finalization)
        ├── session.json                   # Complete structured metadata (machine-readable)
        ├── full_audio.wav                 # Complete raw audio (mono, 16kHz, int16)
        ├── README.txt                     # Human-readable summary
        └── chunks/                        # Individual chunk audio files
            ├── chunk_000.wav              # Audio for chunk 0
            ├── chunk_001.wav              # Audio for chunk 1
            └── ...
```

**Completion Marker (`.complete`):**
This hidden file is created as the last step of session finalization. Its presence guarantees that:
- All audio files have been written successfully
- All metadata has been saved to disk
- The session was not interrupted or crashed

The session replay utility uses this marker to filter out incomplete or corrupted sessions by default.

### session.json Format

The `session.json` file contains all session data in structured JSON format for programmatic analysis.

#### Top-Level Structure

```json
{
  "metadata": { ... },
  "chunks": [ ... ]
}
```

#### Metadata Object

Complete session-level information:

```json
{
  "session_id": "session_143052_whisper",
  "timestamp": "2025-11-06T14:30:52.123456",
  "backend": "whisper",
  "sample_rate": 16000,
  "channels": 1,
  "capture_duration": 45.2,
  "total_chunks": 8,

  // Whisper-specific (null for realtime backend)
  "model": "turbo",
  "device": "cuda",
  "language": "en",
  "vad_aggressiveness": 2,
  "vad_min_silence_duration": 0.2,
  "vad_min_speech_duration": 0.25,
  "vad_speech_pad_duration": 0.2,
  "max_chunk_duration": 60.0,

  // Realtime-specific (null for whisper backend)
  "realtime_endpoint": "wss://api.openai.com/v1/realtime",
  "realtime_instructions": "You are a high-accuracy transcription service...",

  // Transcriptions
  "stitched_transcription": "Hello world. How are you today?",
  "full_audio_transcription": "Hello world. How are you today?",

  // Diff tracking (new in latest version)
  "transcription_similarity": 1.0,           // 0.0 to 1.0 (100% = exact match)
  "transcription_diffs": []                  // Empty array means no differences
}
```

#### Metadata Fields Reference

| Field | Type | Description |
|-------|------|-------------|
| `session_id` | string | Unique session identifier |
| `timestamp` | string | ISO 8601 timestamp when session ended |
| `backend` | string | `"whisper"` or `"realtime"` |
| `sample_rate` | int | Audio sample rate (Hz), typically 16000 |
| `channels` | int | Number of audio channels, always 1 (mono) |
| `capture_duration` | float | Total session duration (seconds) |
| `total_chunks` | int | Number of transcription chunks |
| `model` | string\|null | Whisper model name (e.g., "turbo", "small.en") |
| `device` | string\|null | Inference device ("cpu", "cuda", "mps") |
| `language` | string\|null | Language code ("en", "es", "auto") |
| `vad_aggressiveness` | int\|null | VAD level 0-3 (higher = more aggressive) |
| `vad_min_silence_duration` | float\|null | Min silence to split chunks (seconds) |
| `vad_min_speech_duration` | float\|null | Min speech before transcribing (seconds) |
| `vad_speech_pad_duration` | float\|null | Padding before speech (seconds) |
| `max_chunk_duration` | float\|null | Max chunk length (seconds) |
| `realtime_endpoint` | string\|null | Realtime API WebSocket endpoint |
| `realtime_instructions` | string\|null | Instructions sent to realtime model |
| `stitched_transcription` | string\|null | Transcription from stitched chunks |
| `full_audio_transcription` | string\|null | Transcription from complete audio (if comparison enabled) |
| `transcription_similarity` | float\|null | Similarity ratio 0.0-1.0 between stitched and complete |
| `transcription_diffs` | array\|null | Detailed diff snippets (see below) |

#### Transcription Diff Format

When `--compare_transcripts` is enabled, diffs show where stitched and complete transcriptions differ:

```json
"transcription_diffs": [
  {
    "tag": "replace",
    "stitched": "Hello, world How are",
    "complete": "Hello world. How are"
  },
  {
    "tag": "delete",
    "stitched": "... extra [[text]] here ...",
    "complete": "... ∅ here ..."
  }
]
```

**Diff Tags:**
- `"replace"`: Text differs between versions
- `"insert"`: Text appears only in complete
- `"delete"`: Text appears only in stitched

Context words (3 before/after) are shown around differences. The special symbol `∅` indicates missing text.

#### Chunks Array

Each chunk contains transcription and timing information:

```json
"chunks": [
  {
    "index": 0,
    "text": "Hello, world.",
    "cleaned_text": "Hello, world",
    "start_time": 0.0,
    "end_time": 2.3,
    "duration": 2.3,
    "inference_seconds": 0.15,
    "audio_filename": "chunk_000.wav"
  },
  {
    "index": 1,
    "text": "How are you?",
    "cleaned_text": "How are you?",
    "start_time": 2.3,
    "end_time": 4.1,
    "duration": 1.8,
    "inference_seconds": 0.12,
    "audio_filename": "chunk_001.wav"
  }
]
```

#### Chunk Fields Reference

| Field | Type | Description |
|-------|------|-------------|
| `index` | int | Chunk number (0-indexed) |
| `text` | string | Original transcription from model |
| `cleaned_text` | string\|null | Text after stitching punctuation cleanup |
| `start_time` | float | Chunk start time from session start (seconds) |
| `end_time` | float | Chunk end time from session start (seconds) |
| `duration` | float | Chunk duration (seconds) |
| `inference_seconds` | float\|null | Inference time (null for realtime backend) |
| `audio_filename` | string\|null | Audio file in chunks/ directory (if saved) |

**Punctuation Cleanup:**
The `cleaned_text` field shows text after removing trailing `,` and `.` characters for smoother stitching. Question marks and exclamation points are preserved. The final chunk keeps all punctuation.

### README.txt Format

Human-readable text summary of the session. Sections include:

#### Header
```
Transcription Session: session_143052_whisper
============================================================

Timestamp: 2025-11-06T14:30:52.123456
Backend: whisper
Sample Rate: 16000 Hz
Channels: 1
Capture Duration: 45.20s
Total Chunks: 8

Model: turbo
Device: cuda
Language: en
```

#### VAD Parameters (Whisper only)
```
VAD Parameters:
  Aggressiveness: 2
  Min Silence Duration: 0.2s
  Min Speech Duration: 0.25s
  Speech Pad Duration: 0.2s
  Max Chunk Duration: 60.0s
```

#### Stitched Transcription
```
============================================================
STITCHED TRANSCRIPTION (from chunks):
============================================================
Hello world. How are you today? I'm doing great thanks for asking.
```

#### Full Audio Transcription
```
============================================================
FULL AUDIO TRANSCRIPTION (complete audio):
============================================================
Hello world. How are you today? I'm doing great, thanks for asking.
```

#### Transcription Comparison (when available)
```
============================================================
TRANSCRIPTION COMPARISON:
============================================================
Similarity: 98.50%
Differences found: 1

Diff 1 (replace):
  Stitched: doing great thanks for
  Complete: doing great, thanks for
```

Or, when transcriptions match:
```
============================================================
TRANSCRIPTION COMPARISON:
============================================================
Similarity: 100.00%
Transcriptions match exactly.
```

#### Chunks
```
============================================================
CHUNKS:
============================================================

Chunk 000 [0.00s - 2.30s]:
  Inference: 0.15s
  Text (original): Hello, world.
  Text (cleaned): Hello, world
  Audio: chunks/chunk_000.wav

Chunk 001 [2.30s - 4.10s]:
  Inference: 0.12s
  Text (original): How are you?
  Text (cleaned): How are you?
  Audio: chunks/chunk_001.wav
```

### Audio Files

#### full_audio.wav

Complete recording of the entire session:
- **Format**: WAV (RIFF)
- **Encoding**: 16-bit signed integer PCM
- **Channels**: 1 (mono)
- **Sample Rate**: 16000 Hz (or as configured with `--samplerate`)
- **Duration**: Matches `capture_duration` in metadata

#### chunks/*.wav

Individual chunk audio files (when saved):
- Same format as full_audio.wav
- Filename format: `chunk_NNN.wav` (zero-padded to 3 digits)
- Duration matches chunk `duration` in metadata
- Not all sessions save chunk audio (depends on `save_chunk_audio` setting)

---

## Session Replay Utility

The session replay utility allows you to list, inspect, and retranscribe previously logged transcription sessions.

### Session Completion Marker

All successfully finalized sessions contain a `.complete` marker file. This file is created as the last step of session finalization and indicates that:
- All audio files have been written
- All metadata has been saved
- The session was not interrupted

By default, the session replay utility only lists and loads complete sessions (those with the `.complete` marker). This protects against:
- Incomplete sessions from crashes or interruptions
- Partially written session data
- Corrupted session directories

To work with incomplete sessions, use the `--include_incomplete` flag for listing or `--allow_incomplete` flag for loading/retranscribing.

### Commands

#### `list` - List All Sessions

List all available sessions with optional filtering.

**Basic usage:**
```bash
uv run transcribe-session --subcommand=list
```

**Filter by backend:**
```bash
uv run transcribe-session --subcommand=list --subcommand.backend=whisper
uv run transcribe-session --subcommand=list --subcommand.backend=realtime
```

**Filter by date range:**
```bash
uv run transcribe-session --subcommand=list --subcommand.start_date=2025-11-01 --subcommand.end_date=2025-11-07
```

**Filter by minimum duration:**
```bash
uv run transcribe-session --subcommand=list --subcommand.min_duration=60.0
```

**Show verbose details:**
```bash
uv run transcribe-session --subcommand=list --subcommand.verbose=true
```

**Include incomplete sessions:**
```bash
# Show all sessions including those without completion marker
uv run transcribe-session --subcommand=list --subcommand.include_incomplete=true
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

#### `show` - Show Session Details

Display detailed information about a specific session.

**Usage:**
```bash
uv run transcribe-session --subcommand=show --subcommand.session_path=session_logs/2025-11-07/session_143052_whisper
```

**Load incomplete session:**
```bash
# Force loading session without completion marker
uv run transcribe-session --subcommand=show \
  --subcommand.session_path=session_logs/2025-11-07/session_143052_whisper \
  --subcommand.allow_incomplete=true
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

#### `retranscribe` - Retranscribe a Session

Retranscribe a logged session with different backend or settings.

**Basic usage (Whisper):**
```bash
uv run transcribe-session --subcommand=retranscribe \
  --subcommand.session_path=session_logs/2025-11-07/session_143052_whisper \
  --subcommand.retranscribe_backend=whisper
```

**Use different Whisper model:**
```bash
uv run transcribe-session --subcommand=retranscribe \
  --subcommand.session_path=session_logs/2025-11-07/session_143052_whisper \
  --subcommand.retranscribe_backend=whisper \
  --subcommand.whisper.model=small \
  --subcommand.whisper.device=cuda
```

**Adjust VAD settings:**
```bash
uv run transcribe-session --subcommand=retranscribe \
  --subcommand.session_path=session_logs/2025-11-07/session_143052_whisper \
  --subcommand.retranscribe_backend=whisper \
  --subcommand.whisper.vad.aggressiveness=3 \
  --subcommand.whisper.vad.min_silence_duration=0.3
```

**Retranscribe with Realtime backend:**
```bash
export OPENAI_API_KEY=your_api_key_here
uv run transcribe-session --subcommand=retranscribe \
  --subcommand.session_path=session_logs/2025-11-07/session_143052_whisper \
  --subcommand.retranscribe_backend=realtime \
  --subcommand.realtime.model=gpt-realtime-mini
```

**Custom output directory:**
```bash
uv run transcribe-session --subcommand=retranscribe \
  --subcommand.session_path=session_logs/2025-11-07/session_143052_whisper \
  --subcommand.output_dir=./retranscriptions \
  --subcommand.retranscribe_backend=whisper \
  --subcommand.whisper.model=turbo
```

### Command-Line Flags Reference

#### Common Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--command` | (required) | Command to execute: `list`, `show`, or `retranscribe` |
| `--session_log_dir` | `./session_logs` | Directory containing session logs |

#### List Command Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--backend` | None | Filter by backend (`whisper` or `realtime`) |
| `--start_date` | None | Filter sessions on or after this date (YYYY-MM-DD) |
| `--end_date` | None | Filter sessions on or before this date (YYYY-MM-DD) |
| `--min_duration` | None | Filter sessions with duration >= this value (seconds) |
| `--verbose` | False | Show detailed information |
| `--include_incomplete` | False | Include sessions without completion marker |

#### Show/Retranscribe Command Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--session_path` | (required) | Path to session directory |
| `--allow_incomplete` | False | Allow loading sessions without completion marker |

#### Retranscribe Command Flags

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

### Use Cases

#### Compare Different VAD Settings

Test how different VAD settings affect chunking:

```bash
# Original session with VAD aggressiveness 2
uv run transcribe-demo --backend whisper --aggressiveness 2

# Retranscribe with more aggressive VAD
uv run transcribe-session --command=retranscribe \
  --session_path=session_logs/2025-11-07/session_143052_whisper \
  --vad_aggressiveness=3 \
  --vad_min_silence_duration=0.1
```

#### Test Different Models

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

#### Compare Backends

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

#### Batch Processing

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

### Output Directory Structure

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

---

## Using Session Data

### Python API

You can use the session replay functionality programmatically:

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

### Python Examples

**Load session data:**
```python
import json
from pathlib import Path

session_path = Path("session_logs/2025-11-06/session_143052_whisper/session.json")
with open(session_path) as f:
    data = json.load(f)

print(f"Backend: {data['metadata']['backend']}")
print(f"Duration: {data['metadata']['capture_duration']:.1f}s")
print(f"Chunks: {data['metadata']['total_chunks']}")
print(f"Similarity: {data['metadata']['transcription_similarity']:.2%}")
```

**Analyze chunks:**
```python
for chunk in data["chunks"]:
    print(f"Chunk {chunk['index']}: {chunk['start_time']:.1f}s - {chunk['end_time']:.1f}s")
    print(f"  Original: {chunk['text']}")
    print(f"  Cleaned:  {chunk['cleaned_text']}")
    if chunk['inference_seconds']:
        print(f"  Inference: {chunk['inference_seconds']:.2f}s")
```

**Find sessions with differences:**
```python
import json
from pathlib import Path

logs_dir = Path("session_logs")
for session_json in logs_dir.rglob("session.json"):
    with open(session_json) as f:
        data = json.load(f)

    similarity = data["metadata"].get("transcription_similarity")
    if similarity is not None and similarity < 1.0:
        print(f"{session_json.parent.name}: {similarity:.2%} similarity")
        diffs = data["metadata"]["transcription_diffs"]
        print(f"  {len(diffs)} differences found")
```

**Load audio:**
```python
import wave
import numpy as np

with wave.open("session_logs/2025-11-06/session_143052_whisper/full_audio.wav") as wav:
    sample_rate = wav.getframerate()
    n_frames = wav.getnframes()
    audio_bytes = wav.readframes(n_frames)
    audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

print(f"Sample rate: {sample_rate} Hz")
print(f"Duration: {len(audio) / sample_rate:.1f}s")
```

### Command-Line Examples

**Find all sessions:**
```bash
find session_logs -name "session.json" | sort
```

**Count total sessions:**
```bash
find session_logs -name "session.json" | wc -l
```

**Extract transcriptions:**
```bash
jq -r '.metadata.stitched_transcription' session_logs/*/session_*/session.json
```

**Find sessions with low similarity:**
```bash
find session_logs -name "session.json" -exec \
  jq -r 'select(.metadata.transcription_similarity < 0.95) | .metadata.session_id' {} \;
```

**Get summary statistics:**
```bash
jq -s 'map(.metadata) | {
  total: length,
  avg_duration: (map(.capture_duration) | add / length),
  avg_chunks: (map(.total_chunks) | add / length),
  avg_similarity: (map(.transcription_similarity // 0) | add / length)
}' session_logs/*/session_*/session.json
```

---

## Configuration

Session logging is always enabled and controlled by these flags:

```bash
# Change output directory
uv run transcribe-demo --session_log_dir /path/to/logs

# Minimum duration to save (discard short test runs)
uv run transcribe-demo --session.min_log_duration 10.0

# Save all sessions including very short ones
uv run transcribe-demo --session.min_log_duration 0

# Disable comparison (no similarity/diffs, saves memory and API costs)
uv run transcribe-demo --session.compare_transcripts false
```

**Note:** Sessions shorter than `--min_log_duration` are automatically deleted to avoid cluttering logs.

---

## Version History

### Current Version (2025-11-06)
- ✅ Added `cleaned_text` field to chunks showing punctuation cleanup
- ✅ Added `transcription_similarity` to metadata
- ✅ Added `transcription_diffs` with detailed diff snippets
- ✅ Enhanced README.txt with comparison section
- ✅ Chunks now show both original and cleaned text

### Previous Version
- Basic session logging with transcriptions
- Full audio and chunk audio files
- Metadata for model, device, VAD parameters
- Human-readable README.txt

---

## Related Documentation

See [SITEMAP.md](SITEMAP.md) for a complete guide to all documentation.

---

*Last Updated: 2025-11-15*
