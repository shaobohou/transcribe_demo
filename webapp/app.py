"""
Flask web server for transcribe-demo web app.

Provides a web interface that mimics CLI functionality with real-time transcription.
"""

import logging
import os
import sys
import tempfile
import threading
import wave
from pathlib import Path
from typing import Any

import numpy as np
from absl import app as absl_app
from absl import flags
from flask import Flask, render_template, request as flask_request
from flask_socketio import SocketIO, emit

from transcribe_demo.realtime_backend import run_realtime_transcriber
from transcribe_demo.whisper_backend import run_whisper_transcriber

# Define command-line flags
FLAGS = flags.FLAGS

flags.DEFINE_string("host", "0.0.0.0", "Host to bind to")
flags.DEFINE_integer("port", 5000, "Port to bind to")
flags.DEFINE_boolean("debug", False, "Enable debug mode")
flags.DEFINE_enum(
    "device",
    "auto",
    ["auto", "cpu", "cuda", "mps"],
    "Device to run Whisper on. 'auto' prefers CUDA, then MPS, otherwise CPU.",
)
flags.DEFINE_boolean(
    "require_gpu",
    False,
    "Exit immediately if CUDA is unavailable instead of falling back to CPU.",
)

app = Flask(__name__)

# Flask secret key configuration
# IMPORTANT: Set FLASK_SECRET_KEY environment variable in production
# Using os.urandom() generates a new key on each restart, invalidating sessions
# This is acceptable for development but problematic for production
flask_secret = os.environ.get("FLASK_SECRET_KEY")
if not flask_secret:
    # Generate temporary key (won't persist across restarts)
    flask_secret = os.urandom(24).hex()
    # Log warning that will be visible when server starts
    print(
        "WARNING: Using temporary Flask secret key. Sessions will not persist across server restarts.",
        file=sys.stderr,
    )
    print(
        "For production, set FLASK_SECRET_KEY: python -c 'import os; print(os.urandom(24).hex())'",
        file=sys.stderr,
    )
app.config["SECRET_KEY"] = flask_secret

# CORS configuration
# Default: Allow only same-origin requests (secure by default)
# Set CORS_ALLOWED_ORIGINS environment variable to allow specific origins in production
# Example: export CORS_ALLOWED_ORIGINS="https://yourdomain.com,https://app.yourdomain.com"
# Use CORS_ALLOWED_ORIGINS="*" only for development (insecure)
cors_origins_env = os.environ.get("CORS_ALLOWED_ORIGINS")
if cors_origins_env:
    cors_origins = cors_origins_env
    if cors_origins == "*":
        print(
            "WARNING: CORS set to allow all origins (*). This is insecure for production.",
            file=sys.stderr,
        )
else:
    # Default: same-origin only (most secure)
    cors_origins = []
socketio = SocketIO(app, cors_allowed_origins=cors_origins, max_http_buffer_size=10 * 1024 * 1024)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Active transcription sessions
active_sessions: dict[str, dict[str, Any]] = {}

# Counter for unique transcription IDs
_transcription_counter = 0
_transcription_counter_lock = threading.Lock()


@app.route("/")
def index() -> str:
    """Serve the main web interface."""
    return render_template("index.html")


@socketio.on("connect")
def handle_connect() -> None:
    """Handle client connection."""
    session_id: str = flask_request.sid  # type: ignore[attr-defined]
    logger.info(f"Client connected: {session_id}")
    emit("connected", {"message": "Connected to transcription server"})


@socketio.on("disconnect")
def handle_disconnect() -> None:
    """Handle client disconnection."""
    session_id: str = flask_request.sid  # type: ignore[attr-defined]
    logger.info(f"Client disconnected: {session_id}")
    # Stop any active transcription for this client
    stop_transcription(session_id)


@socketio.on("start_transcription")
def handle_start_transcription(data: dict[str, Any]) -> None:
    """Start a new transcription session."""
    session_id: str = flask_request.sid  # type: ignore[attr-defined]
    logger.info(f"Starting transcription for session {session_id} with config: {data}")

    # Stop any existing session
    stop_transcription(session_id)

    # Extract configuration
    backend = data.get("backend", "whisper")
    model = data.get("model", "base.en")
    language = data.get("language", "en")
    vad_aggressiveness = data.get("vad_aggressiveness", 2)
    min_silence_duration = data.get("min_silence_duration", 0.2)
    max_chunk_duration = data.get("max_chunk_duration", 60.0)
    partial_transcription = data.get("partial_transcription", False)

    # Create temporary WAV file for audio buffering
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, mode="wb")
    temp_path = Path(temp_file.name)

    # Initialize WAV file
    sample_rate = 16000
    channels = 1
    wav_writer = wave.open(temp_file, "wb")
    wav_writer.setnchannels(channels)
    wav_writer.setsampwidth(2)  # 16-bit audio
    wav_writer.setframerate(sample_rate)

    # Generate unique transcription ID
    global _transcription_counter
    with _transcription_counter_lock:
        _transcription_counter += 1
        transcription_id = _transcription_counter

    # Store session info including configuration
    active_sessions[session_id] = {
        "transcription_id": transcription_id,
        "wav_writer": wav_writer,
        "temp_path": temp_path,
        "backend": backend,
        "model": model,
        "language": language,
        "vad_aggressiveness": vad_aggressiveness,
        "min_silence_duration": min_silence_duration,
        "max_chunk_duration": max_chunk_duration,
        "partial_transcription": partial_transcription,
        "is_transcribing": False,  # Track if background transcription is active
        "stop_requested": False,  # Flag to signal thread should stop
        "lock": threading.Lock(),
    }

    emit("transcription_started", {"message": "Transcription started"})


@socketio.on("audio_chunk")
def handle_audio_chunk(data: dict[str, Any]) -> None:
    """Receive audio chunk from client."""
    session_id: str = flask_request.sid  # type: ignore[attr-defined]

    if session_id not in active_sessions:
        logger.warning(f"Received audio chunk for inactive session: {session_id}")
        return

    session = active_sessions[session_id]

    # Get audio data (binary ArrayBuffer from browser)
    audio_data = data.get("audio")
    if not audio_data:
        return

    # Handle both binary data (new format) and JSON array (legacy format for backwards compatibility)
    if isinstance(audio_data, bytes):
        audio_bytes = audio_data
    elif isinstance(audio_data, list):
        # Legacy format: convert from list to bytes
        audio_bytes = np.array(audio_data, dtype=np.int16).tobytes()
    else:
        logger.warning(f"Unexpected audio data type: {type(audio_data)}")
        return

    # Write to WAV file
    with session["lock"]:
        try:
            session["wav_writer"].writeframes(audio_bytes)
        except Exception as e:
            logger.error(f"Error writing audio chunk: {e}")


@socketio.on("stop_transcription")
def handle_stop_transcription() -> None:
    """Stop recording and transcribe the complete audio."""
    session_id: str = flask_request.sid  # type: ignore[attr-defined]
    logger.info(f"Stopping transcription for session {session_id}")

    if session_id not in active_sessions:
        emit("transcription_error", {"error": "No active session"})
        return

    session = active_sessions[session_id]

    # Close WAV file
    with session["lock"]:
        session["wav_writer"].close()

    # Extract configuration from session
    transcription_id = session.get("transcription_id", 0)
    backend = session.get("backend", "whisper")
    model = session.get("model", "base.en")
    language = session.get("language", "en")
    vad_aggressiveness = session.get("vad_aggressiveness", 2)
    min_silence_duration = session.get("min_silence_duration", 0.2)
    max_chunk_duration = session.get("max_chunk_duration", 60.0)
    partial_transcription = session.get("partial_transcription", False)

    # Start transcription in background thread
    def run_transcription() -> None:
        # Mark as transcribing to prevent premature cleanup
        current_session = active_sessions.get(session_id)
        if current_session and current_session.get("transcription_id") == transcription_id:
            current_session["is_transcribing"] = True

        try:
            # Check if stop was requested before starting
            current_session = active_sessions.get(session_id)
            if not current_session or current_session.get("stop_requested"):
                return

            socketio.emit("transcription_status", {"message": "Processing audio..."}, to=session_id)

            # Run appropriate backend
            if backend == "realtime":
                # Realtime backend uses keyword-only ChunkConsumer protocol
                def realtime_chunk_consumer(
                    *,
                    chunk_index: int,
                    text: str,
                    absolute_start: float,
                    absolute_end: float,
                    inference_seconds: float | None,
                ) -> None:
                    """Send transcription chunks to client."""
                    # Check if this is still the active transcription and stop wasn't requested
                    current_session = active_sessions.get(session_id)
                    if not current_session or current_session.get("transcription_id") != transcription_id:
                        return
                    if current_session.get("stop_requested"):
                        return

                    socketio.emit(
                        "transcription_chunk",
                        {
                            "text": text,
                            "is_final": True,
                            "index": chunk_index,
                            "start_time": absolute_start,
                            "end_time": absolute_end,
                        },
                        to=session_id,
                    )

                # Use OpenAI API for realtime transcription
                api_key = os.environ.get("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OPENAI_API_KEY not set for realtime transcription")

                run_realtime_transcriber(
                    api_key=api_key,
                    endpoint="wss://api.openai.com/v1/realtime",
                    model=model if model != "base.en" else "gpt-realtime-mini",
                    sample_rate=16000,
                    channels=1,
                    chunk_duration=2.0,
                    instructions="Transcribe the audio accurately.",
                    disable_ssl_verify=False,
                    chunk_consumer=realtime_chunk_consumer,
                    compare_transcripts=False,
                    max_capture_duration=0.0,
                    language=language,
                    session_logger=None,
                    min_log_duration=0.0,
                    audio_file=session["temp_path"],
                    playback_speed=10.0,  # Process quickly
                    vad_threshold=0.3,
                    vad_silence_duration_ms=200,
                    debug=False,
                )
            else:  # whisper
                # Whisper backend uses positional parameters
                def whisper_chunk_consumer(
                    chunk_index: int,
                    chunk_text: str,
                    start_time: float,
                    end_time: float,
                    inference_seconds: float | None,
                    is_final: bool,
                ) -> None:
                    """Send transcription chunks to client."""
                    # Check if this is still the active transcription and stop wasn't requested
                    current_session = active_sessions.get(session_id)
                    if not current_session or current_session.get("transcription_id") != transcription_id:
                        return
                    if current_session.get("stop_requested"):
                        return

                    socketio.emit(
                        "transcription_chunk",
                        {
                            "text": chunk_text,
                            "is_final": is_final,
                            "index": chunk_index,
                            "start_time": start_time,
                            "end_time": end_time,
                        },
                        to=session_id,
                    )

                run_whisper_transcriber(
                    model_name=model,
                    sample_rate=16000,
                    channels=1,
                    temp_file=None,
                    ca_cert=None,
                    disable_ssl_verify=False,
                    device_preference=FLAGS.device,
                    require_gpu=FLAGS.require_gpu,
                    chunk_consumer=whisper_chunk_consumer,
                    vad_aggressiveness=vad_aggressiveness,
                    vad_min_silence_duration=min_silence_duration,
                    vad_min_speech_duration=0.25,
                    vad_speech_pad_duration=0.2,
                    max_chunk_duration=max_chunk_duration,
                    compare_transcripts=False,
                    max_capture_duration=0.0,
                    language=language,
                    session_logger=None,
                    min_log_duration=0.0,
                    audio_file=session["temp_path"],
                    playback_speed=10.0,  # Process quickly
                    enable_partial_transcription=partial_transcription,
                )

            socketio.emit("transcription_complete", {"message": "Transcription finished"}, to=session_id)

        except Exception as e:
            logger.error(f"Transcription error: {e}", exc_info=True)
            socketio.emit("transcription_error", {"error": str(e)}, to=session_id)
        finally:
            # Clean up temp file
            try:
                session["temp_path"].unlink()
            except Exception as e:
                logger.error(f"Error cleaning up temp file: {e}")

            # Only remove session if this is still the active transcription
            # (prevents old threads from interfering with new recordings)
            current_session = active_sessions.get(session_id)
            if current_session and current_session.get("transcription_id") == transcription_id:
                active_sessions.pop(session_id, None)

    thread = threading.Thread(target=run_transcription, daemon=True)
    thread.start()

    emit("transcription_stopped", {"message": "Recording stopped, processing..."})


def stop_transcription(session_id: str) -> None:
    """Stop transcription for a specific session."""
    session = active_sessions.get(session_id)
    if session is None:
        return

    # Mark session as stopped (signal to background thread)
    session["stop_requested"] = True

    try:
        with session["lock"]:
            if "wav_writer" in session:
                session["wav_writer"].close()
    except Exception as e:
        logger.error(f"Error closing WAV writer: {e}")

    # Only delete temp file and remove session if transcription is not currently running
    # Background transcription thread will handle cleanup when it finishes
    if not session.get("is_transcribing", False):
        try:
            if "temp_path" in session and session["temp_path"].exists():
                session["temp_path"].unlink()
        except Exception as e:
            logger.error(f"Error cleaning up temp file: {e}")

        # Remove the session now that cleanup is done
        active_sessions.pop(session_id, None)


def main(argv: list[str]) -> None:
    """Main entry point for the web server."""
    del argv  # Unused
    logger.info(f"Starting server on {FLAGS.host}:{FLAGS.port} (debug={FLAGS.debug})")
    socketio.run(app, host=FLAGS.host, port=FLAGS.port, debug=FLAGS.debug)


if __name__ == "__main__":
    absl_app.run(main)
