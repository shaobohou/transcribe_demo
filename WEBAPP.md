# Web App Quick Start Guide

A tasteful web interface for real-time audio transcription, mimicking the functionality of the CLI tool.

## Quick Start

1. **Install dependencies**:
   ```bash
   uv sync --project ci --refresh
   ```

2. **Start the server**:
   ```bash
   uv --project ci run python webapp/app.py
   ```

   Or use the convenient script:
   ```bash
   cd webapp && ./run-webapp.sh --cpu
   ```

3. **Open your browser**:
   Navigate to `http://localhost:5000`

4. **Start transcribing**:
   - Select your backend (Whisper/Realtime)
   - Choose a model
   - Click "Start Recording"
   - Allow microphone access
   - Speak into your microphone
   - Click "Stop" when done
   - View your transcription!

## Features

### Dual Backend Support
- **Whisper**: Local processing (CPU/GPU)
- **Realtime**: Cloud-based via OpenAI Realtime API (requires `OPENAI_API_KEY`)

### Modern Interface
- Clean, responsive dark mode design
- Real-time status indicators
- Recording animation
- Collapsible advanced settings
- Mobile-friendly

### Configuration Options
- **Backend**: Choose between Whisper (local) or Realtime (cloud)
- **Model**: Multiple Whisper models available (tiny, base, small, medium, large, turbo)
- **Language**: Support for English, Spanish, French, German, Italian, Portuguese, Chinese, Japanese, Korean
- **Advanced Settings**:
  - VAD Aggressiveness (0-3)
  - Min Silence Duration (0.1-2.0s)
  - Max Chunk Duration (10-300s)

## How It Works

1. **Recording Phase**:
   - Browser captures audio via Web Audio API (16kHz, 16-bit PCM)
   - Audio chunks are streamed to Flask server via WebSocket
   - Server buffers audio chunks to a temporary WAV file

2. **Transcription Phase**:
   - When recording stops, server transcribes the buffered audio
   - Uses FileAudioSource with high playback speed (10x) for fast processing
   - Transcription chunks are sent back to browser in real-time
   - Results displayed as they arrive

3. **Cleanup**:
   - Temporary audio files are automatically deleted
   - Session state is cleaned up on disconnect

## Architecture

```
Browser (Web Audio API)
  ↓ WebSocket
Flask Server
  ↓ Temporary WAV File
Whisper/Realtime Backend
  ↓ WebSocket
Browser Display
```

## Requirements

- Modern browser with Web Audio API support:
  - Chrome 74+
  - Firefox 25+
  - Safari 14.1+
  - Edge 79+
- Microphone access permissions
- For Realtime backend: `OPENAI_API_KEY` environment variable

## Differences from CLI

| Feature | CLI | Web App |
|---------|-----|---------|
| Audio Input | Microphone (live) | Microphone → Buffer → Playback |
| Transcription | Real-time streaming | Buffered (after recording stops) |
| Output | Terminal | Web UI |
| Configuration | CLI flags | Web forms |
| Session Logs | Optional file output | Not implemented yet |
| File Input | Supported | Not implemented yet |

## Future Enhancements

- [ ] True real-time streaming transcription (no buffering)
- [ ] File upload support
- [ ] Session history and export
- [ ] User authentication
- [ ] Multiple concurrent sessions
- [ ] Advanced VAD visualization
- [ ] Audio waveform display
- [ ] Transcript editing and export

## Troubleshooting

**"Failed to access microphone"**
- Check browser permissions
- Ensure HTTPS if on remote server
- Try a different browser

**"OPENAI_API_KEY not set" (Realtime backend)**
```bash
export OPENAI_API_KEY=your_api_key_here
```

**Port already in use**
```bash
./run-webapp.sh --port 5001
```

## Development

Edit `webapp/app.py` for backend logic or `webapp/templates/index.html` for frontend.

Restart the server to see changes:
```bash
# Press Ctrl+C to stop
# Then restart with:
uv --project ci run python webapp/app.py
```

## Related Documentation

- **webapp/README.md**: Detailed technical documentation
- **CLAUDE.md**: Development workflow
- **README.md**: Main project documentation
- **DESIGN.md**: Architecture overview

---

*Created: 2025-11-11*
