# Realtime Backend Calibration Guide

## Overview

This document provides calibration guidance for the OpenAI Realtime API backend to match the transcription quality of the Whisper backend with VAD chunking.

## Baseline: Whisper Backend Transcription

**Test Audio**: NPR News audio (http://public.npr.org/anon.npr-mp3/npr/news/newscast.mp3)
**Duration**: 120 seconds (limited from 280s source)
**Model**: whisper-turbo

### Whisper Backend Configuration

- **VAD Aggressiveness**: 2 (moderate)
- **Min Silence Duration**: 0.2s (200ms)
- **Padding Duration**: 0.2s (200ms)
- **Min Speech Duration**: 0.25s (250ms)
- **Max Chunk Duration**: 60.0s

### Whisper Baseline Results

**Chunk Pattern**:
- Chunk 0: 60.00s (hit max duration limit - continuous speech)
- Chunk 1: 22.04s
- Chunk 2: 4.22s
- Chunk 3: 4.58s
- Chunk 4: 4.64s
- Chunk 5: 6.26s
- Chunk 6: 18.41s

**Key Observations**:
1. First chunk hit the 60s max duration limit (continuous speech, no natural pauses)
2. Subsequent chunks ranged from 4-22 seconds
3. VAD successfully detected natural speech pauses
4. Stitched transcription achieved 99.38% similarity with full-audio transcription
5. Only minor differences in grammar ("and moving" vs "are moving", "winds across" vs "winds cross")

### Whisper Final Transcription

```
Live from NPR News in Washington, I'm Nora Rahm. Today is day 40 of the government shutdown...
[Full transcription showing high accuracy with proper punctuation and grammar]
```

## Realtime Backend Configuration

### Current Server VAD Settings

Located in `realtime_backend.py:_create_session_update()`:

```python
"turn_detection": {
    "type": "server_vad",
    "threshold": 0.3,              # Speech detection sensitivity (0.0-1.0)
    "prefix_padding_ms": 200,      # Audio before speech starts
    "silence_duration_ms": 300,    # Silence before ending chunk
}
```

### Audio Processing

- **Sample Rate**: Resampled from 16kHz to 24kHz for API
- **Chunk Duration**: ~2.0s chunks (configurable in main.py, default REALTIME_CHUNK_DURATION=2.0)
- **Audio Format**: PCM16 (16-bit PCM)

## Calibration Strategy

### Problem: Different Chunking Approaches

**Whisper Backend**:
- Uses WebRTC VAD to detect natural speech pauses
- Creates variable-length chunks (4-60s)
- Chunks align with sentence/phrase boundaries

**Realtime Backend**:
- Uses server-side VAD (OpenAI's implementation)
- Processes audio in ~2s chunks sent to the API
- Server VAD determines when to emit transcriptions based on turn detection

### Key Calibration Parameters

#### 1. Server VAD Threshold (Current: 0.3)

**Purpose**: Controls sensitivity for detecting speech vs silence

- **Lower values (0.0-0.2)**: More sensitive, detects more audio as speech
  - Pro: Captures quiet speech, filler words
  - Con: May transcribe background noise

- **Higher values (0.4-1.0)**: Less sensitive, stricter speech detection
  - Pro: Filters background noise better
  - Con: May miss quiet speech or cut off words

**Recommendation for NPR-style audio**:
- Start with 0.3 (current default)
- If missing quiet speech: Lower to 0.2
- If transcribing background noise: Raise to 0.4

#### 2. Silence Duration (Current: 300ms)

**Purpose**: How much continuous silence before the server ends a turn/chunk

Comparison with Whisper baseline:
- Whisper uses 200ms min silence duration
- Realtime uses 300ms silence duration

**Impact**:
- **Lower values (100-200ms)**: Creates more, shorter chunks
  - Pro: Faster partial transcripts, better for real-time display
  - Con: May split mid-sentence, more API calls

- **Higher values (400-600ms)**: Creates fewer, longer chunks
  - Pro: More complete sentences, fewer API calls
  - Con: Slower to display transcripts

**Recommendation**:
- For matching Whisper baseline: Use 200ms (matches Whisper VAD)
- For news/podcast audio: 300-400ms works well
- For conversational audio: 200-300ms for faster responses

#### 3. Prefix Padding (Current: 200ms)

**Purpose**: Audio included before detected speech starts

Comparison with Whisper baseline:
- Whisper uses 200ms padding
- Realtime uses 200ms prefix padding

**Impact**:
- Prevents cutting off the beginning of words
- Higher values (300-400ms) help with words that start softly
- Lower values (100-150ms) reduce latency

**Recommendation**:
- Keep at 200ms (matches Whisper and is industry standard)

## Calibration Procedure

### Step 1: Baseline Test

Run realtime backend on the same NPR audio:

```bash
uv --project ci run transcribe-demo \
  --backend realtime \
  --audio_file http://public.npr.org/anon.npr-mp3/npr/news/newscast.mp3 \
  --max_capture_duration 120 \
  --compare_transcripts=true
```

**Note**: If you encounter SSL certificate errors in sandboxed/CI environments, use the `--insecure_downloads` flag:

```bash
uv --project ci run transcribe-demo \
  --backend realtime \
  --audio_file http://public.npr.org/anon.npr-mp3/npr/news/newscast.mp3 \
  --max_capture_duration 120 \
  --insecure_downloads
```

### Step 2: Adjust VAD Parameters

Use CLI flags to adjust VAD parameters (no code changes needed):

```bash
uv --project ci run transcribe-demo \
  --backend realtime \
  --audio_file http://public.npr.org/anon.npr-mp3/npr/news/newscast.mp3 \
  --max_capture_duration 120 \
  --realtime_vad_threshold 0.25 \
  --realtime_vad_silence_duration_ms 250 \
  --realtime_vad_prefix_padding_ms 200
```

Available VAD flags:
- `--realtime_vad_threshold`: Speech detection threshold (0.0-1.0, default: 0.3)
- `--realtime_vad_prefix_padding_ms`: Padding before speech starts (default: 200ms)
- `--realtime_vad_silence_duration_ms`: Silence before ending a turn (default: 300ms)

### Step 3: Test Different Audio Types

- **News/Podcasts**: Higher silence duration (300-400ms), threshold 0.3
- **Conversations**: Lower silence duration (200-250ms), threshold 0.2-0.3
- **Noisy environments**: Higher threshold (0.4-0.5), longer silence (400ms)

### Step 4: Compare Results

Use the built-in comparison feature:

```bash
--compare_transcripts=true
```

This will show:
- Similarity percentage between chunked and full-audio transcription
- Character-level diff showing differences
- Helps identify if VAD settings are causing transcription issues

## Expected Differences vs Whisper

Even with perfect calibration, some differences are expected:

1. **Model Differences**:
   - Whisper backend uses local Whisper model
   - Realtime backend uses OpenAI's hosted Whisper-1 model
   - May have slight version/tuning differences

2. **Chunking Strategy**:
   - Whisper: Large variable chunks (4-60s)
   - Realtime: Smaller fixed chunks (~2s) + server VAD
   - Different context windows may affect grammar/punctuation

3. **Latency vs Accuracy Tradeoff**:
   - Whisper optimized for accuracy (post-processing)
   - Realtime optimized for low latency (streaming)

## Troubleshooting

### Issue: Poor Transcription Quality

**Check**:
- Is the audio too quiet? → Lower threshold to 0.2
- Is background noise being transcribed? → Raise threshold to 0.4
- Are sentences being cut mid-word? → Increase silence_duration_ms to 400

### Issue: Slow/Laggy Transcripts

**Check**:
- Is silence_duration_ms too high? → Reduce to 200-250ms
- Is chunk_duration too large? → Reduce REALTIME_CHUNK_DURATION to 1.5s

### Issue: Missing Words

**Check**:
- Is threshold too high? → Lower to 0.2-0.3
- Is prefix_padding too short? → Increase to 250-300ms

## Recommended Settings by Audio Type

### Professional Audio (NPR, Podcasts)

```python
"threshold": 0.3,
"prefix_padding_ms": 200,
"silence_duration_ms": 300,
```

### Conversational Audio (Interviews, Meetings)

```python
"threshold": 0.25,
"prefix_padding_ms": 200,
"silence_duration_ms": 250,
```

### Noisy Environments

```python
"threshold": 0.4,
"prefix_padding_ms": 250,
"silence_duration_ms": 350,
```

### Low Latency Requirements (Live Captions)

```python
"threshold": 0.3,
"prefix_padding_ms": 150,
"silence_duration_ms": 200,
```

## Future Improvements

1. ~~**Make VAD parameters configurable via CLI flags**~~ ✅ **Implemented**
   - Added `--realtime_vad_threshold`, `--realtime_vad_prefix_padding_ms`, `--realtime_vad_silence_duration_ms`
   - See Step 2 in Calibration Procedure above

2. **Add automatic calibration**:
   - Run test audio through both backends
   - Compare results and suggest optimal parameters

3. **Audio type detection**:
   - Analyze audio characteristics (SNR, speech rate, pauses)
   - Automatically select optimal VAD parameters

## References

- Whisper VAD: `src/transcribe_demo/whisper_backend.py` (WebRTCVAD class)
- Realtime VAD: `src/transcribe_demo/realtime_backend.py` (_create_session_update)
- OpenAI Realtime API: https://platform.openai.com/docs/guides/realtime
