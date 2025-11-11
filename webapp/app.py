"""
Flask web server for transcribe-demo web app.

Provides a web interface that mimics CLI functionality with real-time transcription.
"""

import logging
import os
import tempfile
import threading
import wave
from pathlib import Path
from typing import Any

import numpy as np
from flask import Flask, render_template, request as flask_request
from flask_socketio import SocketIO, emit

from transcribe_demo.realtime_backend import run_realtime_transcriber
from transcribe_demo.whisper_backend import run_whisper_transcriber

app = Flask(__name__)
app.config["SECRET_KEY"] = "transcribe-demo-secret-key"
socketio = SocketIO(app, cors_allowed_origins="*", max_http_buffer_size=10 * 1024 * 1024)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Active transcription sessions
active_sessions: dict[str, dict[str, Any]] = {}


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

    # Store session info including configuration
    active_sessions[session_id] = {
        "wav_writer": wav_writer,
        "temp_path": temp_path,
        "backend": backend,
        "model": model,
        "language": language,
        "vad_aggressiveness": vad_aggressiveness,
        "min_silence_duration": min_silence_duration,
        "max_chunk_duration": max_chunk_duration,
        "transcribing": False,
        "audio_chunks": [],
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

    # Get audio data (list of int16 samples)
    audio_data = data.get("audio")
    if not audio_data:
        return

    # Convert from list to bytes
    if isinstance(audio_data, list):
        audio_bytes = np.array(audio_data, dtype=np.int16).tobytes()
    else:
        audio_bytes = audio_data

    # Write to WAV file
    with session["lock"]:
        try:
            session["wav_writer"].writeframes(audio_bytes)
            session["audio_chunks"].append(audio_bytes)
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
    backend = session.get("backend", "whisper")
    model = session.get("model", "base.en")
    language = session.get("language", "en")
    vad_aggressiveness = session.get("vad_aggressiveness", 2)
    min_silence_duration = session.get("min_silence_duration", 0.2)
    max_chunk_duration = session.get("max_chunk_duration", 60.0)

    # Start transcription in background thread
    def run_transcription() -> None:
        try:
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

                api_key = os.environ.get("ANTHROPIC_API_KEY")
                if not api_key:
                    raise ValueError("ANTHROPIC_API_KEY not set")

                run_realtime_transcriber(
                    api_key=api_key,
                    endpoint="wss://api.anthropic.com/v1/messages",
                    model=model,
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
                    device_preference="cpu",
                    require_gpu=False,
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
                )

            socketio.emit("transcription_complete", {"message": "Transcription finished"}, to=session_id)

        except Exception as e:
            logger.error(f"Transcription error: {e}", exc_info=True)
            socketio.emit("transcription_error", {"error": str(e)}, to=session_id)
        finally:
            # Clean up
            try:
                session["temp_path"].unlink()
            except Exception as e:
                logger.error(f"Error cleaning up temp file: {e}")

            if session_id in active_sessions:
                del active_sessions[session_id]

    thread = threading.Thread(target=run_transcription, daemon=True)
    thread.start()

    emit("transcription_stopped", {"message": "Recording stopped, processing..."})


def stop_transcription(session_id: str) -> None:
    """Stop transcription for a specific session."""
    if session_id in active_sessions:
        session = active_sessions[session_id]
        try:
            with session["lock"]:
                if "wav_writer" in session:
                    session["wav_writer"].close()
        except Exception as e:
            logger.error(f"Error closing WAV writer: {e}")

        try:
            if "temp_path" in session and session["temp_path"].exists():
                session["temp_path"].unlink()
        except Exception as e:
            logger.error(f"Error cleaning up temp file: {e}")

        del active_sessions[session_id]


if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)
