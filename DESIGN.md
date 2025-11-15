# Design Decisions & Architecture

**Last Updated**: 2025-11-15

This document explains the key design decisions and architectural patterns in the transcribe-demo codebase. For user-facing documentation, see **README.md**. For development workflow, see **CLAUDE.md**. For implementation improvements, see **TODO.md**.

---

## Table of Contents

1. [Design Philosophy](#design-philosophy)
2. [Architecture Overview](#architecture-overview)
3. [Backend Design](#backend-design)
4. [VAD-Based Chunking Strategy](#vad-based-chunking-strategy)
5. [Stitching & Punctuation Cleanup](#stitching--punctuation-cleanup)
6. [Session Logging Design](#session-logging-design)
7. [Session Replay Design](#session-replay-design)

---

## Design Philosophy

### Core Principles

1. **Accuracy Over Speed**: Prioritize transcription quality by using natural speech boundaries for chunking
2. **Debuggability**: Comprehensive session logging preserves all data needed to diagnose issues
3. **Flexibility**: Support multiple backends (local Whisper, cloud Realtime API) with consistent interfaces
4. **Testability**: Design for automated testing despite inherent unpredictability of VAD and audio processing
5. **Safety First**: Default to conservative settings that prevent hallucinations and ensure data integrity

### Design Goals

**Primary Goals:**
- Real-time transcription with minimal latency
- Accurate chunking at natural speech pauses (no mid-word splits)
- Smooth stitching across chunks without artifacts
- Complete audit trail for debugging and analysis

**Non-Goals:**
- Perfect real-time factor (we prioritize accuracy over speed)
- Speaker diarization (out of scope)
- Background noise transcription (we use VAD to filter non-speech)
- Live editing/correction of transcripts

---

## Architecture Overview

### High-Level Component Structure

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              cli.py                                     │
│  ┌──────────────────────┐  ┌────────────────────────────────────────┐ │
│  │  chunk_collector.py  │  │     transcript_diff.py                 │ │
│  │  ChunkCollector      │  │  - compute_transcription_diff()        │ │
│  │  - Collects chunks   │  │  - print_transcription_summary()       │ │
│  │  - Punctuation       │  │  - Tokenization & diff generation      │ │
│  │    cleanup           │  └────────────────────────────────────────┘ │
│  └──────────────────────┘                                              │
│              │                                                          │
│              │         ┌────────────────────────────────────┐          │
│              │         │    backend_protocol.py             │          │
│              │         │  - TranscriptionChunk              │          │
│              │         │  - ChunkConsumer Protocol          │          │
│              │         │  - TranscriptionBackend Protocol   │          │
│              │         │  - TranscriptionResult Protocol    │          │
│              │         └────────────────────────────────────┘          │
│              │                       │                                 │
│              ▼                       ▼                                 │
│  ┌────────────────────┐   ┌──────────────────────────────┐           │
│  │ whisper_backend.py │   │   realtime_backend.py        │           │
│  │  ┌──────────────┐  │   │    ┌──────────────┐          │           │
│  │  │  WebRTCVAD   │  │   │    │  WebSocket   │          │           │
│  │  │  Chunking    │  │   │    │  Streaming   │          │           │
│  │  └──────────────┘  │   │    └──────────────┘          │           │
│  └────────────────────┘   └──────────────────────────────┘           │
│              │                       │                                 │
│              │         ┌────────────────────────────────────┐          │
│              │         │    backend_config.py               │          │
│              │         │  - VADConfig                       │          │
│              │         │  - WhisperConfig                   │          │
│              │         │  - RealtimeConfig                  │          │
│              │         │  - PartialTranscriptionConfig      │          │
│              │         └────────────────────────────────────┘          │
│              │                                                          │
│              └───────────┬──────────────────────────────────┘          │
│                          ▼                                              │
│              ┌───────────────────────┐                                 │
│              │   Audio Source        │                                 │
│              │   (Protocol)          │                                 │
│              └───────────┬───────────┘                                 │
│                          │                                              │
│         ┌────────────────┴────────────────┐                            │
│         ▼                                 ▼                            │
│  ┌──────────────────┐           ┌──────────────────────┐              │
│  │ audio_capture.py │           │ file_audio_source.py │              │
│  │ (Microphone)     │           │ (File/URL)           │              │
│  │ - sounddevice    │           │ - soundfile          │              │
│  │ - Real-time      │           │ - Download from URL  │              │
│  │                  │           │ - Simulate playback  │              │
│  └──────────────────┘           └──────────────────────┘              │
│                          │                                              │
│                          ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │        session_logger.py                                        │  │
│  │  SessionLogger                                                  │  │
│  │  - Persist audio (WAV/FLAC)                                     │  │
│  │  - Save metadata & transcriptions                               │  │
│  │  - Diff tracking                                                │  │
│  └─────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│              session_replay.py / session_replay_cli.py      │
│  - List sessions                                            │
│  - Load previous sessions                                   │
│  - Retranscribe with different settings/backends            │
└─────────────────────────────────────────────────────────────┘
```

### Module Responsibilities

| Module | Responsibility | Key Classes/Functions |
|--------|----------------|----------------------|
| `cli.py` | Orchestration, CLI args | `ChunkCollectorWithStitching`, `cli_main()`, `_finalize_transcription_session()` |
| `backend_protocol.py` | Type-safe backend interfaces | `TranscriptionChunk`, `ChunkConsumer`, `TranscriptionBackend`, `TranscriptionResult` |
| `backend_config.py` | Configuration dataclasses | `VADConfig`, `WhisperConfig`, `RealtimeConfig`, `PartialTranscriptionConfig` |
| `whisper_backend.py` | Local Whisper + VAD chunking | `WebRTCVAD`, `run_whisper_transcriber()` |
| `realtime_backend.py` | OpenAI Realtime API streaming | `run_realtime_transcriber()` |
| `audio_capture.py` | Microphone audio capture | `AudioCaptureManager` |
| `file_audio_source.py` | File/URL audio simulation | `FileAudioSource` |
| `transcript_diff.py` | Transcription comparison utilities | `compute_transcription_diff()`, `print_transcription_summary()` |
| `chunk_collector.py` | Chunk display and stitching | `ChunkCollectorWithStitching` |
| `session_logger.py` | Session persistence | `SessionLogger` |
| `session_replay.py` | Session loading & retranscription | `load_session()`, `retranscribe_session()` |
| `session_replay_cli.py` | Session replay CLI | CLI for listing, showing, retranscribing sessions |

### Audio Source Design

**Problem**: Need to support both live microphone capture and file-based simulation for testing/development.

**Solution**: Define a common interface that both `AudioCaptureManager` and `FileAudioSource` implement:

```python
class AudioSource(Protocol):
    """Common interface for audio sources."""
    audio_queue: queue.Queue[np.ndarray | None]
    stop_event: threading.Event
    capture_limit_reached: threading.Event

    def start() -> None: ...
    def stop() -> None: ...
    def wait_until_stopped() -> None: ...
    def close() -> None: ...
    def get_full_audio() -> np.ndarray: ...
    def get_capture_duration() -> float: ...
```

**AudioCaptureManager** (microphone):
- Uses `sounddevice` for real-time audio capture
- Feeds audio frames directly from hardware
- User can press Enter or Ctrl+C to stop

**FileAudioSource** (files/URLs):
- Loads audio from local files or downloads from HTTP/HTTPS URLs
- Simulates real-time playback by feeding chunks with realistic timing
- Supports playback speed control (e.g., 2.0x for faster testing)
- Automatically resamples audio to target sample rate
- Cleans up temporary files for URL downloads

**Key Benefits**:
- Backends don't need to know if audio is live or simulated
- Same chunking/transcription logic works for both sources
- Easy to test with pre-recorded audio
- Session replay can use either live or file sources

---

## Backend Design

### Why Two Backends?

We support both local Whisper and cloud-based Realtime API to serve different use cases:

**Whisper Backend** (default):
- **Pros**: Privacy (local), no API costs, GPU acceleration, fine-grained VAD control
- **Cons**: Slower (GPU needed for real-time), requires model downloads
- **Best for**: Privacy-sensitive work, offline use, GPU available

**Realtime API Backend**:
- **Pros**: Lower latency, no local GPU needed, minimal setup
- **Cons**: API costs, requires internet, less VAD control (fixed 2.0s chunks)
- **Best for**: Quick setup, no GPU available, cloud workflows

### Backend Interface Design

Both backends implement a common pattern:

```python
def run_backend(
    # Audio settings
    sample_rate: int,
    channels: int,

    # Backend-specific config
    ...,

    # Common hooks
    chunk_consumer: Optional[ChunkConsumer] = None,
    max_capture_duration: float = 0,

    # SSL/Security
    ca_cert: Optional[Path] = None,
    insecure: bool = False,
) -> None:
    """
    Run transcription backend.

    Emits chunks via chunk_consumer callback:
        chunk_consumer(chunk_index, text, start, end, inference_time)

    Respects max_capture_duration for automatic stopping.
    """
```

**Key Design Decision**: The `chunk_consumer` callback pattern allows `cli.py` to collect chunks from either backend without knowing implementation details. This enables:
- Consistent display formatting
- Unified session logging
- Backend-agnostic testing

---

## VAD-Based Chunking Strategy

### The Core Insight

**Problem**: Arbitrary time-based chunking (e.g., every 5 seconds) can split words or sentences mid-utterance, degrading transcription quality.

**Solution**: Use Voice Activity Detection (VAD) to detect natural speech pauses and chunk there instead.

### How VAD Chunking Works

1. **Frame-Level Detection**: Audio is analyzed in 30ms frames
2. **Speech Classification**: Each frame is classified as speech or silence
3. **Chunk Triggers**:
   - **Minimum speech detected** (default 0.25s) AND
   - **Silence threshold reached** (default 0.2s) OR
   - **Maximum duration exceeded** (default 60s)

### Why These Defaults?

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| VAD aggressiveness | 2 | Balance between missing speech (0-1) and capturing noise (3) |
| Min silence duration | 0.2s | Natural pause between phrases; shorter = too many chunks |
| Min speech duration | 0.25s | Filter out clicks/coughs; long enough for meaningful content |
| Speech pad duration | 0.2s | Prevents cutting off the start of words |
| Max chunk duration | 60s | Safety limit; Whisper quality degrades on very long audio |

### Key Design Decision: No Audio Overlap

**Important**: VAD chunks split at silence = **no overlap between chunks**.

- **Advantage**: Simpler logic, no duplicate audio, clean concatenation
- **Trade-off**: Context loss across chunk boundaries (addressed via stitching)

**Alternative Considered**: Sliding window with overlap (rejected due to complexity and redundant transcription)

### Timeout Behavior Design

**Goal**: When time limit is reached, ensure zero data loss while stopping immediately.

**Design Decision**: Stop capture immediately but transcribe all buffered audio, including incomplete chunks.

**Implementation**:

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

**Key Points**:
- Audio arriving after timeout (chunk5) is discarded
- All previously captured audio is transcribed immediately
- No waiting for VAD silence detection or minimum chunk size requirements
- Ensures the chunk that pushes over the limit is included

**Rationale**: Users expect time limits to be respected, but losing partial audio data would be surprising. This design provides deterministic cutoff while maximizing data preservation.

### Language Parameter Default

**Default**: `language="en"`

**Rationale**:
- Prevents hallucinations on silence/background noise
- Auto-detect (`language=None`) can hallucinate foreign language text on noise
- English-only constraint focuses the model and improves accuracy

**When to change**: Use `--language auto` only if transcribing multiple languages in same session

---

## Stitching & Punctuation Cleanup

### The Problem

Whisper adds punctuation assuming each chunk is a complete sentence. When VAD splits mid-sentence, this creates artifacts:

```
Chunk 1: "Hello, world."
Chunk 2: "How are you today."
Stitched: "Hello, world. How are you today."  ← Incorrect (should be one sentence)
```

### The Solution: Selective Punctuation Stripping

**Rule**: Strip trailing `,` and `.` from intermediate chunks, but preserve `?` and `!`

**Rationale**:
- Periods and commas are likely artifacts of incomplete sentences
- Questions and exclamations are intentional and should be preserved
- Final chunk keeps all punctuation (it's genuinely complete)

**Implementation** (`cli.py:ChunkCollectorWithStitching._clean_chunk_text()`):

```python
def _clean_chunk_text(self, text: str, is_final_chunk: bool = False) -> str:
    """Remove trailing punctuation that interferes with stitching."""
    if is_final_chunk:
        return text.strip()

    # Strip trailing commas and periods (but keep ? and !)
    cleaned = text.rstrip()
    while cleaned and cleaned[-1] in ",。":
        cleaned = cleaned[:-1].rstrip()

    return cleaned
```

**Key Design Decision**: This is a heuristic that works well in practice. Edge cases exist (intentional mid-sentence pauses), but they're rare.

### Output Display Strategy

**Design**: Show stitched results every 3 chunks and at session end.

**Rationale**:
- **Every chunk**: Too verbose, hard to read continuous transcription
- **Only at end**: No intermediate feedback during long sessions
- **Every 3 chunks**: Balances visibility with readability

**Implementation**: `ChunkCollectorWithStitching` displays individual chunks as they arrive, then shows the stitched result periodically to provide context.

**User Benefit**: Users can monitor transcription progress and quality in real-time without overwhelming output.

---

## Session Logging Design

### Design Goals

1. **Complete Audit Trail**: Save everything needed to reproduce or debug a session
2. **Human & Machine Readable**: Both JSON (programmatic) and README.txt (browsable)
3. **Minimal Disk Usage**: FLAC compression for audio, optional chunk audio
4. **Diff Tracking**: Automatically track differences between chunked and full-audio transcription

### Directory Structure Design

```
session_logs/
└── YYYY-MM-DD/                    # Date-based organization
    └── session_HHMMSS_<backend>/  # Timestamp + backend identifier
        ├── .complete              # Completion marker
        ├── session.json           # Machine-readable metadata
        ├── full_audio.wav         # Complete session audio
        ├── README.txt             # Human-readable summary
        └── chunks/                # Individual chunk audio (optional)
            ├── chunk_000.wav
            └── ...
```

**Rationale for Structure**:
- **Date folders**: Easy browsing, prevents directory bloat
- **Timestamp in name**: Unique IDs, chronological sorting
- **Backend suffix**: Quickly identify which backend was used
- **`.complete` marker**: Distinguishes successful sessions from interrupted ones
- **Dual formats** (JSON + TXT): Serves both automation and manual inspection

### Completion Marker Pattern

**Problem**: Crashes or Ctrl+C can leave partial session data on disk.

**Solution**: Write `.complete` marker as final step of session finalization.

**Benefits**:
- Session replay utility can filter incomplete sessions by default
- Prevents processing corrupted or partial data
- Clear indicator of session success

### Diff Tracking Design

**Goal**: Quantify how much chunking affects transcription accuracy.

**Implementation**:
1. Transcribe chunks in real-time (stitched result)
2. Transcribe full audio at end (complete result)
3. Compute token-level diff using `difflib.SequenceMatcher`
4. Save similarity ratio and diff snippets

**Stored Data**:
- `transcription_similarity`: Float 0.0-1.0 (percentage match)
- `transcription_diffs`: Array of `{tag, stitched, complete}` snippets

**Key Design Decision**: Token-level (not character-level) comparison better reflects semantic differences:

```python
def _tokenize_with_original(text: str) -> list[str]:
    """Split on whitespace + punctuation while preserving original characters."""
    return re.findall(r'\S+|\s+', text)
```

**Cost Consideration**: For Realtime API backend, comparison doubles API usage since the full audio must be transcribed separately. Users can disable with `--nocompare_transcripts` if cost is a concern. For Whisper backend, there is no additional cost.

**Design Rationale**: Comparison is enabled by default because the debugging value outweighs the cost for most use cases. The similarity scores and diffs help tune VAD settings and validate transcription quality.

### Min Duration Filtering

**Feature**: Sessions shorter than `--min_log_duration` (default 10s) are automatically discarded.

**Rationale**:
- Prevents cluttering logs with test runs (e.g., "testing 1 2 3")
- Most useful sessions are longer conversations
- Configurable via `--min_log_duration 0` to keep everything

---

## Session Replay Design

### Purpose

Enable experimentation with different backends, models, or VAD settings on the same audio without re-recording.

### Key Features

1. **List Sessions**: Browse all logged sessions with filtering
2. **Show Details**: Inspect session metadata and transcriptions
3. **Retranscribe**: Re-run transcription with different settings

### Completion Marker Integration

**Default Behavior**: Only load/list complete sessions (those with `.complete` marker)

**Safety Flags**:
- `--include_incomplete`: List incomplete sessions
- `--allow_incomplete`: Load incomplete sessions (use with caution)

**Rationale**: Protect users from accidentally processing corrupted data while still allowing recovery of interrupted sessions if needed.

### Retranscription Workflow

```python
# 1. Load original session (includes audio + metadata)
loaded_session = load_session("session_logs/.../session_143052_whisper")

# 2. Retranscribe with new settings
result_path = retranscribe_session(
    loaded_session=loaded_session,
    output_dir="./session_logs",
    backend="whisper",
    backend_kwargs={"model": "small", "vad_aggressiveness": 3},
)

# 3. Results saved to:
#    session_logs/YYYY-MM-DD/retranscribe_HHMMSS_whisper_from_session_143052_whisper/
```

**Key Design Decision**: Retranscribed sessions reference original via directory name, creating clear lineage.

---

## Future Design Considerations

### Potential Improvements

1. **Sliding Window Refinement** (see `cli.py:242-266` TODO)
   - Use 3-chunk sliding window for better context
   - Requires word-level timestamps (not currently implemented)
   - Trade-off: 3x inference time, 1-chunk latency

2. **Silero VAD Backend** (see `whisper_backend.py:124-173` TODO)
   - More robust to background noise/music
   - Trade-off: Additional dependency (PyTorch model)

3. **Speaker Diarization**
   - Identify different speakers
   - Trade-off: Significant complexity, slower processing

4. **Real-time WER (Word Error Rate) Calculation**
   - Compute WER during session for live feedback
   - Trade-off: Requires reference transcription

### Design Constraints to Maintain

1. **Separate backends**: Don't tightly couple Whisper and Realtime logic
2. **Callback pattern**: Keep `chunk_consumer` as primary interface
3. **Session logging completeness**: Never lose data (always save full audio + metadata)
4. **Test independence**: Keep unit tests fast (<0.5s), integration tests moderate (<5s)

---

## Related Documentation

See [SITEMAP.md](SITEMAP.md) for a complete guide to all documentation.

**This document** answers "why did we design it this way?" (architecture, design rationale)

---

## Document Maintenance

**Review Schedule**: Update when making architectural changes or adding major features

**How to Update**:
1. Document new design decisions with rationale
2. Keep examples in sync with actual code
3. Reference specific line numbers for key implementations
4. Move implementation details to TODO.md if appropriate

**Last Major Changes**:
- 2025-11-07: Initial comprehensive design document
