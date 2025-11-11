# Transcribe Demo Web App

A tasteful web interface for the transcribe-demo CLI tool, providing real-time audio transcription through your browser.

## Features

- **Real-time Audio Transcription**: Record audio directly from your browser and see transcriptions appear in real-time
- **Dual Backend Support**:
  - **Whisper**: Local processing with GPU/CPU support
  - **Realtime**: Cloud-based transcription via Anthropic's API
- **Configurable Settings**:
  - Multiple Whisper models (tiny, base, small, medium, large, turbo)
  - Language selection (English, Spanish, French, German, etc.)
  - VAD (Voice Activity Detection) tuning
  - Partial transcription support
- **Modern UI**: Clean, responsive design with dark mode and real-time status indicators
- **WebSocket-based**: Low-latency bidirectional communication for smooth audio streaming

## Installation

1. Install the webapp dependencies:
   ```bash
   uv sync --group webapp
   ```

   For CPU-only environments (CI/sandboxes):
   ```bash
   uv sync --project ci --group webapp --refresh
   ```

2. (Optional) For Realtime backend, set your Anthropic API key:
   ```bash
   export ANTHROPIC_API_KEY=your_api_key_here
   ```

## Usage

### Starting the Server

From the repository root:

```bash
# With GPU support (local development)
uv run python webapp/app.py

# CPU-only (CI/sandboxes)
uv --project ci run python webapp/app.py
```

The server will start on `http://localhost:5000`

### Using the Web Interface

1. **Open your browser** and navigate to `http://localhost:5000`

2. **Configure settings**:
   - Choose backend (Whisper for local, Realtime for cloud)
   - Select a model (smaller = faster, larger = more accurate)
   - Set language (default: English)
   - (Optional) Adjust advanced settings like VAD aggressiveness

3. **Start recording**:
   - Click "Start Recording"
   - Allow microphone access when prompted
   - Speak into your microphone
   - Watch transcriptions appear in real-time

4. **Stop recording**:
   - Click "Stop" when finished
   - Transcription will be finalized

## Configuration Options

### Backend Selection

- **Whisper**:
  - Runs locally on your machine
  - Requires CPU/GPU resources
  - Free to use
  - Models: tiny.en, base.en, small.en, medium.en, large, turbo

- **Realtime**:
  - Cloud-based API transcription
  - Requires ANTHROPIC_API_KEY
  - Lower latency (<200ms)
  - Fixed 2.0s chunks

### Advanced Settings

- **VAD Aggressiveness** (Whisper only): Controls how aggressively voice is detected
  - 0-3, default: 2
  - Higher values = more strict speech detection

- **Min Silence Duration**: Minimum silence before chunking
  - Default: 0.2 seconds
  - Range: 0.1 - 2.0 seconds

- **Max Chunk Duration**: Maximum duration for each audio chunk
  - Default: 60 seconds
  - Range: 10 - 300 seconds

- **Partial Transcription** (Whisper only): Show intermediate transcription results
  - Provides real-time feedback as speech is processed
  - May show incomplete sentences

## Architecture

### Components

1. **Flask Server** (`app.py`):
   - Serves HTML interface
   - Manages WebSocket connections
   - Coordinates transcription sessions

2. **WebSocket Communication**:
   - `connect/disconnect`: Session management
   - `start_transcription`: Initiate new session with config
   - `audio_chunk`: Stream audio data from browser
   - `stop_transcription`: End session
   - `transcription_chunk`: Receive transcription results
   - `transcription_error`: Handle errors

3. **Frontend** (`templates/index.html`):
   - Modern, responsive UI
   - Real-time audio capture via Web Audio API
   - Socket.IO for bidirectional communication
   - Dynamic status indicators

### Audio Pipeline

```
Browser Microphone
  ↓ (Web Audio API)
16kHz, 16-bit PCM Audio
  ↓ (WebSocket)
Flask Server
  ↓ (Audio Queue)
Whisper/Realtime Backend
  ↓ (Transcription)
WebSocket → Browser Display
```

## Browser Requirements

- Modern browser with Web Audio API support:
  - Chrome 74+
  - Firefox 25+
  - Safari 14.1+
  - Edge 79+
- Microphone access permissions

## Performance Notes

### Whisper Backend

- First chunk latency: 0.5-2 seconds
- Subsequent chunks: 0.4-0.8 seconds
- CPU usage varies by model size
- GPU acceleration recommended for larger models

### Realtime Backend

- Latency: <200ms typical
- Fixed 2.0-second chunks
- Requires stable internet connection
- API costs apply

## Troubleshooting

### "Failed to access microphone"

- Check browser permissions (usually in address bar)
- Ensure no other application is using the microphone
- Try HTTPS if on remote server (required for microphone access)

### "Transcription error"

- **Whisper**: Check that the model is downloaded and CPU/GPU is available
- **Realtime**: Verify ANTHROPIC_API_KEY is set and valid
- Check server logs for detailed error messages

### SSL/Certificate Issues

If you encounter SSL errors with Realtime backend:

```bash
# Use custom CA certificate
export SSL_CERT_FILE=/path/to/cert.pem

# Or disable SSL verification (development only, insecure!)
export PYTHONHTTPSVERIFY=0
```

### Performance Issues

- Try a smaller model (e.g., tiny.en or base.en)
- Close other browser tabs
- Check CPU/GPU usage
- For Whisper: Use `--project ci` for CPU-only mode

## Development

### Running with Debug Mode

```bash
# Edit app.py and set debug=True
socketio.run(app, host="0.0.0.0", port=5000, debug=True)
```

### Testing

The webapp integrates with existing backends, so use the main test suite:

```bash
uv run python -m pytest
```

## Related Documentation

- **CLAUDE.md**: Development workflow and implementation rules
- **README.md**: Main project documentation
- **DESIGN.md**: Architecture and technical design
- **SITEMAP.md**: Complete documentation guide

## License

Same as parent transcribe-demo project.
