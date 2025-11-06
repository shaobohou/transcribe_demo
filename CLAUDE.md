# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**⚠️ IMPORTANT: Keep this file updated when making significant changes to the codebase, especially:**
- Changes to default settings or configuration
- New features or major architectural changes
- Changes to command-line arguments
- New test files or testing approaches
- Updates to key design decisions
- Refactoring opportunities or code cleanup tasks (update REFACTORING.md)

## Project Overview

Transcribe Demo is a real-time microphone transcription tool that supports two backends:
1. **Local Whisper**: Uses OpenAI's Whisper model running locally (CPU/GPU)
2. **Realtime API**: Streams audio to OpenAI's Realtime API for lower-latency transcription

The project uses `uv` as its Python project manager.

## Common Commands

### Setup
```bash
uv sync
```

### Running the Application
```bash
# Local Whisper backend (default: turbo model with VAD)
uv run transcribe-demo

# Specify different model
uv run transcribe-demo --model small.en

# OpenAI Realtime API backend
export OPENAI_API_KEY=sk-...
uv run transcribe-demo --backend realtime

# Force specific device
uv run transcribe-demo --device cuda    # CUDA GPU
uv run transcribe-demo --device mps     # Apple Metal
uv run transcribe-demo --device cpu     # CPU only
uv run transcribe-demo --require-gpu    # Abort if no GPU

# VAD configuration
uv run transcribe-demo --vad-aggressiveness 3              # More aggressive pause detection (default: 2)
uv run transcribe-demo --vad-min-silence-duration 0.5      # Minimum silence to split chunk (default: 0.2s)
uv run transcribe-demo --vad-min-speech-duration 0.5       # Minimum speech before transcribing (default: 0.25s)
uv run transcribe-demo --vad-speech-pad-duration 0.5       # Padding before speech (default: 0.2s)
uv run transcribe-demo --max-chunk-duration 90             # Max chunk length with VAD (default: 60s)
```

### Testing
```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_vad.py        # WebRTC VAD tests

# Run with verbose output
uv run pytest -v
```

### Git Hooks
```bash
git config core.hooksPath .githooks
UV_CACHE_DIR=$(pwd)/.uv-cache uv sync --group dev
```
- Pre-commit hook runs `uv run python -m pytest` and blocks commits when tests fail.

## Architecture

### Core Components

**main.py** - Entry point and orchestration
- `ChunkCollectorWithStitching`: Collects transcription chunks and displays them
- `parse_args()`: CLI argument parsing with support for both backends
- `main()`: Orchestrates the transcription flow based on selected backend

**whisper_backend.py** - Local Whisper transcription engine
- `TranscriptionChunk`: Data class storing transcription with timing and segment info
- `WebRTCVAD`: Wrapper class for WebRTC Voice Activity Detection (speech vs. silence detection)
- `run_whisper_transcriber()`: Main loop that captures audio, runs Whisper inference, and manages GPU/CPU device selection
- **VAD-based chunking**: Uses WebRTC VAD to detect speech pauses and chunk at natural boundaries (2-60s chunks, no overlap)

**realtime_backend.py** - OpenAI Realtime API integration
- `run_realtime_transcriber()`: Async WebSocket client that streams PCM16 audio to OpenAI
- Uses server-side VAD (voice activity detection) for automatic speech segmentation
- Handles audio resampling from input rate to 24kHz (required by API)

### Transcript Stitching

**VAD-based chunking:**
- **Simple stitching**: Chunks split at natural speech pauses have no audio overlap
- **Punctuation cleanup**: Trailing commas and periods are stripped from intermediate chunks (but kept on the final chunk)
  - Whisper adds punctuation as if each chunk is complete, but VAD can split mid-sentence
  - Question marks and exclamation points are preserved as they're more intentional
- Chunks are joined with spaces and labeled as `[STITCHED]` in output
- Minimizes transcription errors at boundaries by chunking at natural speech pauses

**Key insight:** By using VAD to detect natural speech pauses, chunks don't have overlapping audio. This eliminates the need for complex overlap resolution algorithms and reduces transcription errors at chunk boundaries. Automatic punctuation cleanup ensures smooth stitching even when VAD splits occur mid-sentence.

### Audio Processing Flow

**VAD-based chunking:**
1. **Capture**: `sounddevice.InputStream` captures audio in real-time
2. **VAD Processing**: WebRTC VAD analyzes 30ms frames to detect speech vs. silence
3. **Buffering**: Audio accumulates until:
   - Minimum speech duration reached (default 0.25s, configurable with `--vad-min-speech-duration`), AND
   - 0.2s of continuous silence detected (configurable with `--vad-min-silence-duration`), OR
   - Maximum chunk duration reached (default 60s, configurable with `--max-chunk-duration`)
4. **Max Duration Warning**: When max chunk duration is exceeded, a warning is logged to stderr suggesting to increase the limit or check for continuous speech without pauses
5. **Speech Padding**: Adds 0.2s padding before speech segments to avoid cutting word beginnings (configurable with `--vad-speech-pad-duration`)
6. **Transcription**: Whisper runs inference with `language="en"` to prevent foreign language hallucinations
7. **Display**: Chunks shown with both audio duration and inference time (e.g., `audio: 5.23s | inference: 2.15s`)
8. **Punctuation Cleanup**: Trailing commas and periods are stripped from intermediate chunks before stitching
9. **Stitching**: Simple joining of cleaned chunks with spaces (no overlap to resolve)

### Device Selection Logic

Auto-detection preference order:
1. CUDA (if available)
2. Apple Metal/MPS (if available)
3. CPU (fallback)

The `--require-gpu` flag causes immediate abort if no GPU is detected.

### SSL/Certificate Handling

Both backends support:
- `--ca-cert`: Custom certificate bundle for corporate proxies
- `--insecure-downloads`: Skip certificate verification (for model downloads in Whisper, WebSocket connections in Realtime)

## Key Design Decisions

- **Default model**: `turbo` - Balanced speed/accuracy (significantly better than `tiny.en`, faster than `small`)
- **VAD-based chunking**: Uses WebRTC VAD to chunk at natural speech pauses (2-60s variable length)
  - Eliminates boundary transcription errors by splitting at natural pauses instead of fixed intervals
  - WebRTC VAD aggressiveness defaults to 2 (moderate/aggressive) - balances speech detection with false positive prevention
  - Minimum chunk: 2s (avoids transcribing brief noise bursts)
  - **Minimum speech duration**: 0.25s (250ms) - filters out short noise bursts (based on research)
  - **Minimum silence duration**: 0.2s (200ms) - responsive to natural speech pauses for quick chunking
  - **Speech padding**: 0.2s (200ms) - adds context before speech to avoid cutting word beginnings
  - **Maximum chunk duration**: 60s - balances Whisper's 30s optimal window with practical need for longer continuous speech; warns when exceeded
- **English-only constraint**: `language="en"` parameter prevents foreign language hallucinations on silence/noise
- **Duration logging**: Both audio duration and inference time displayed for performance monitoring
- **Server-side VAD in Realtime**: The API handles voice activity detection automatically; manual audio buffer commits are not needed
- **Realtime chunk duration**: Fixed at 2.0s for lower latency (Realtime API doesn't use VAD chunking)

## Testing Strategy

**`tests/test_vad.py`** - WebRTC VAD tests (20 tests):
- **Initialization tests**: Valid/invalid sample rates, frame durations, aggressiveness levels
- **Speech detection tests**: Silence, noise, simulated speech, edge cases (NaN, inf, large values)
- **Integration tests**: Continuous frames, alternating speech/silence, different sample rates
- **Edge case handling**: Ensures robust error handling for malformed audio input

When modifying VAD logic, ensure all tests pass and consider adding test cases for new scenarios.

## Code Quality and Refactoring

**See REFACTORING.md** for refactoring opportunities, guidelines on when/how to refactor, and common cleanup patterns. Update it when you identify code quality issues or complete refactorings.
