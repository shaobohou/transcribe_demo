from __future__ import annotations

from absl import flags

FLAGS = flags.FLAGS

# Backend configuration
flags.DEFINE_enum(
    "backend",
    "whisper",
    ["whisper", "realtime"],
    "Transcription backend to use.",
)

# API configuration
flags.DEFINE_string(
    "api_key",
    None,
    "OpenAI API key for realtime transcription. Defaults to OPENAI_API_KEY.",
)

# Whisper model configuration
flags.DEFINE_string(
    "model",
    "turbo",
    "Whisper checkpoint to load. Recommended: 'turbo' (default, GPU, multilingual) or 'base.en' (CPU-friendly, English-only).",
)
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

flags.DEFINE_string(
    "language",
    "en",
    "Preferred language code for transcription (e.g., en, es). Use 'auto' to let the model detect.",
)

# Partial transcription configuration (Whisper backend only)
flags.DEFINE_boolean(
    "enable_partial_transcription",
    False,
    "Enable real-time partial transcription of accumulating audio chunks using a fast model. "
    "Only applies to Whisper backend.",
)
flags.DEFINE_string(
    "partial_model",
    "base.en",
    "Whisper model for partial transcription (should be faster than main model, e.g., base.en, tiny.en).",
)
flags.DEFINE_float(
    "partial_interval",
    1.0,
    "Interval (seconds) between partial transcription updates.",
    lower_bound=0.1,
    upper_bound=10.0,
)
flags.DEFINE_float(
    "max_partial_buffer_seconds",
    10.0,
    "Segment duration (in seconds) for partial transcription. "
    "Audio is divided into fixed-duration segments. Each segment is continuously "
    "transcribed as new audio accumulates, with updates printed on separate lines.",
    lower_bound=1.0,
    upper_bound=60.0,
)

# Audio configuration
flags.DEFINE_integer(
    "samplerate",
    16000,
    "Input sample rate expected by the model.",
)
flags.DEFINE_integer(
    "channels",
    1,
    "Number of microphone input channels.",
)
flags.DEFINE_string(
    "audio_file",
    None,
    "Path or URL to audio file for simulating live transcription (MP3, WAV, FLAC, etc.). "
    "Supports local files and HTTP/HTTPS URLs. "
    "If provided, audio will be read from file/URL instead of microphone.",
)
flags.DEFINE_float(
    "playback_speed",
    1.0,
    "Playback speed multiplier when using --audio-file (1.0 = real-time, 2.0 = 2x speed).",
    lower_bound=0.1,
    upper_bound=10.0,
)

# File configuration
flags.DEFINE_string(
    "temp_file",
    None,
    "Optional path to persist audio chunks for inspection.",
)

# SSL/Certificate configuration
flags.DEFINE_string(
    "ca_cert",
    None,
    "Custom certificate bundle to trust when downloading Whisper models.",
)
flags.DEFINE_boolean(
    "disable_ssl_verify",
    False,
    "Disable SSL certificate verification for all network operations, including model downloads "
    "and Realtime API connections. Use this to bypass certificate issues in restricted networks. "
    "WARNING: This is insecure and not recommended for production use.",
)

# VAD configuration
flags.DEFINE_integer(
    "vad_aggressiveness",
    2,
    "WebRTC VAD aggressiveness level: 0=least aggressive, 3=most aggressive.",
    lower_bound=0,
    upper_bound=3,
)
flags.DEFINE_float(
    "vad_min_silence_duration",
    0.2,
    "Minimum duration of silence (seconds) to trigger chunk split.",
)
flags.DEFINE_float(
    "vad_min_speech_duration",
    0.25,
    "Minimum duration of speech (seconds) required before transcribing.",
)
flags.DEFINE_float(
    "vad_speech_pad_duration",
    0.2,
    "Padding duration (seconds) added before speech to avoid cutting words.",
)
flags.DEFINE_float(
    "max_chunk_duration",
    60.0,
    "Maximum chunk duration in seconds when using VAD.",
)

# Feature flags
flags.DEFINE_boolean(
    "refine_with_context",
    False,
    "[NOT YET IMPLEMENTED] Use 3-chunk sliding window to refine middle chunk transcription.",
)

# Realtime API configuration
flags.DEFINE_string(
    "realtime_model",
    "gpt-realtime-mini",
    "Realtime model to use with the OpenAI Realtime API.",
)
flags.DEFINE_string(
    "realtime_endpoint",
    "wss://api.openai.com/v1/realtime",
    "Realtime websocket endpoint (advanced).",
)
flags.DEFINE_string(
    "realtime_instructions",
    (
        "You are a high-accuracy transcription service. "
        "Return a concise verbatim transcript of the most recent audio buffer. "
        "Do not add commentary or speaker labels."
    ),
    "Instruction prompt sent to the realtime model.",
)
flags.DEFINE_float(
    "realtime_vad_threshold",
    0.2,
    "Server VAD threshold for turn detection (0.0-1.0). Lower = more sensitive. "
    "Default 0.2 works well for continuous speech (news, podcasts). "
    "Only applies when using realtime backend.",
    lower_bound=0.0,
    upper_bound=1.0,
)
flags.DEFINE_integer(
    "realtime_vad_silence_duration_ms",
    100,
    "Silence duration in milliseconds required to detect turn boundary. "
    "Lower values = more frequent chunks. Default 100ms works well for fast-paced content. "
    "Only applies when using realtime backend.",
    lower_bound=100,
    upper_bound=2000,
)
flags.DEFINE_boolean(
    "realtime_debug",
    False,
    "Enable debug logging for realtime transcription events (shows delta and completed events).",
)

# Comparison and capture configuration
flags.DEFINE_boolean(
    "compare_transcripts",
    True,
    "Compare chunked transcription with full-audio transcription at session end. "
    "Note: For Realtime API, this doubles API usage cost.",
)
flags.DEFINE_float(
    "max_capture_duration",
    120.0,
    "Maximum duration (seconds) to run the transcription session. "
    "Program will gracefully stop after this duration. Set to 0 for unlimited duration.",
)

# Session logging configuration (always enabled)
flags.DEFINE_string(
    "session_log_dir",
    "./session_logs",
    "Directory to save session logs. All sessions are logged with full audio, chunk audio, and metadata.",
)
flags.DEFINE_float(
    "min_log_duration",
    10.0,
    "Minimum session duration (seconds) required to save logs. Sessions shorter than this are discarded.",
)
flags.DEFINE_enum(
    "audio_format",
    "flac",
    ["wav", "flac"],
    "Audio format for saved session files. 'flac' provides lossless compression (~50-60% smaller), 'wav' is uncompressed.",
)

# Validators
flags.register_validator(
    "max_capture_duration",
    lambda value: value >= 0.0,
    message="--max_capture_duration must be >= 0 (set to 0 for unlimited duration)",
)
