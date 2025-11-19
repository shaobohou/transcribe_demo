"""Tests for backend protocol and configuration classes."""

from __future__ import annotations

import numpy as np
import pytest

from transcribe_demo.backend_config import (
    PartialTranscriptionConfig,
    RealtimeConfig,
    RealtimeVADConfig,
    VADConfig,
    WhisperConfig,
)
from transcribe_demo.backend_protocol import (
    ChunkConsumer,
    TranscriptionChunk,
    TranscriptionResult,
    _adapt_legacy_consumer,
)
from transcribe_demo.realtime_backend import RealtimeTranscriptionResult
from transcribe_demo.whisper_backend import WhisperTranscriptionResult


class TestTranscriptionChunk:
    """Test TranscriptionChunk dataclass."""

    def test_create_chunk(self) -> None:
        """Test creating a basic transcription chunk."""
        chunk = TranscriptionChunk(
            index=0,
            text="Hello world",
            start_time=0.0,
            end_time=2.0,
            inference_seconds=0.5,
        )

        assert chunk.index == 0
        assert chunk.text == "Hello world"
        assert chunk.start_time == 0.0
        assert chunk.end_time == 2.0
        assert chunk.inference_seconds == 0.5
        assert chunk.is_partial is False
        assert chunk.audio is None

    def test_chunk_with_partial_flag(self) -> None:
        """Test chunk marked as partial."""
        chunk = TranscriptionChunk(
            index=0,
            text="Hello",
            start_time=0.0,
            end_time=1.0,
            is_partial=True,
        )

        assert chunk.is_partial is True
        assert chunk.inference_seconds is None

    def test_chunk_with_audio(self) -> None:
        """Test chunk with audio data."""
        audio = np.zeros(480, dtype=np.float32)
        chunk = TranscriptionChunk(
            index=0,
            text="Hello",
            start_time=0.0,
            end_time=1.0,
            audio=audio,
        )

        assert chunk.audio is not None
        assert chunk.audio.shape == (480,)


class TestChunkConsumerProtocol:
    """Test ChunkConsumer protocol."""

    def test_chunk_consumer_callable(self) -> None:
        """Test that callable accepting chunk satisfies protocol."""
        received_chunks: list[TranscriptionChunk] = []

        def consumer(chunk: TranscriptionChunk) -> None:
            received_chunks.append(chunk)

        # Type check: this should satisfy the protocol
        _: ChunkConsumer = consumer

        # Test functionality
        test_chunk = TranscriptionChunk(
            index=0,
            text="Test",
            start_time=0.0,
            end_time=1.0,
        )
        consumer(test_chunk)

        assert len(received_chunks) == 1
        assert received_chunks[0].text == "Test"

    def test_chunk_consumer_class(self) -> None:
        """Test that class with __call__ satisfies protocol."""

        class ConsumerClass:
            def __init__(self) -> None:
                self.chunks: list[TranscriptionChunk] = []

            def __call__(self, chunk: TranscriptionChunk) -> None:
                self.chunks.append(chunk)

        consumer = ConsumerClass()
        _: ChunkConsumer = consumer

        chunk = TranscriptionChunk(index=0, text="Test", start_time=0.0, end_time=1.0)
        consumer(chunk)

        assert len(consumer.chunks) == 1
        assert consumer.chunks[0].text == "Test"


class TestLegacyConsumerAdapter:
    """Test _adapt_legacy_consumer function."""

    def test_adapt_none(self) -> None:
        """Test adapting None returns None."""
        result = _adapt_legacy_consumer(legacy_consumer=None)
        assert result is None

    def test_adapt_legacy_consumer(self) -> None:
        """Test adapting legacy consumer to new protocol."""
        received_args: list[tuple] = []

        def legacy_consumer(
            chunk_index: int,
            text: str,
            absolute_start: float,
            absolute_end: float,
            inference_seconds: float | None,
            is_partial: bool,
        ) -> None:
            received_args.append((chunk_index, text, absolute_start, absolute_end, inference_seconds, is_partial))

        adapted = _adapt_legacy_consumer(legacy_consumer=legacy_consumer)
        assert adapted is not None

        # Use adapted consumer with new TranscriptionChunk interface
        chunk = TranscriptionChunk(
            index=5,
            text="Hello world",
            start_time=1.0,
            end_time=3.0,
            inference_seconds=0.5,
            is_partial=True,
        )
        adapted(chunk)

        assert len(received_args) == 1
        idx, text, start, end, inf, partial = received_args[0]
        assert idx == 5
        assert text == "Hello world"
        assert start == 1.0
        assert end == 3.0
        assert inf == 0.5
        assert partial is True


class TestTranscriptionResultProtocol:
    """Test that existing result types implement TranscriptionResult protocol."""

    def test_whisper_result_implements_protocol(self) -> None:
        """Test WhisperTranscriptionResult implements protocol."""
        result = WhisperTranscriptionResult(
            full_audio_transcription="Hello world",
            capture_duration=5.0,
            metadata={"model": "turbo", "device": "cuda"},
        )

        # Type check: should satisfy protocol
        _: TranscriptionResult = result

        # Verify protocol properties
        assert result.capture_duration == 5.0
        assert result.full_audio_transcription == "Hello world"
        assert result.metadata == {"model": "turbo", "device": "cuda"}

    def test_whisper_result_default_metadata_is_empty_dict(self) -> None:
        """Test that default metadata is an empty dict."""
        result = WhisperTranscriptionResult(
            full_audio_transcription=None,
            capture_duration=0.0,
        )

        assert result.metadata == {}

    def test_realtime_result_implements_protocol(self) -> None:
        """Test RealtimeTranscriptionResult implements protocol."""
        result = RealtimeTranscriptionResult(
            full_audio=np.zeros(1000, dtype=np.float32),
            sample_rate=16000,
            capture_duration=2.0,
            metadata={"model": "gpt-realtime-mini"},
            full_audio_transcription="Test transcription",
        )

        # Type check: should satisfy protocol
        _: TranscriptionResult = result

        # Verify protocol properties
        assert result.capture_duration == 2.0
        assert result.full_audio_transcription == "Test transcription"
        assert result.metadata == {"model": "gpt-realtime-mini"}

    def test_realtime_result_default_metadata_is_empty_dict(self) -> None:
        """Test that default metadata is an empty dict."""
        result = RealtimeTranscriptionResult(
            full_audio=np.zeros(100, dtype=np.float32),
            sample_rate=16000,
        )

        assert result.metadata == {}


class TestVADConfig:
    """Test VADConfig validation."""

    def test_default_vad_config(self) -> None:
        """Test default VAD configuration."""
        config = VADConfig()

        assert config.aggressiveness == 2
        assert config.min_silence_duration == 0.2
        assert config.min_speech_duration == 0.25
        assert config.speech_pad_duration == 0.2
        assert config.max_chunk_duration == 60.0

    def test_custom_vad_config(self) -> None:
        """Test custom VAD configuration."""
        config = VADConfig(
            aggressiveness=3,
            min_silence_duration=0.5,
            min_speech_duration=0.3,
            speech_pad_duration=0.1,
            max_chunk_duration=30.0,
        )

        assert config.aggressiveness == 3
        assert config.min_silence_duration == 0.5

    def test_vad_config_invalid_aggressiveness(self) -> None:
        """Test VAD config rejects invalid aggressiveness."""
        with pytest.raises(ValueError, match="aggressiveness must be 0-3"):
            VADConfig(aggressiveness=4)

        with pytest.raises(ValueError, match="aggressiveness must be 0-3"):
            VADConfig(aggressiveness=-1)

    def test_vad_config_invalid_durations(self) -> None:
        """Test VAD config rejects invalid durations."""
        with pytest.raises(ValueError, match="min_silence_duration must be positive"):
            VADConfig(min_silence_duration=0.0)

        with pytest.raises(ValueError, match="min_speech_duration must be positive"):
            VADConfig(min_speech_duration=-0.1)

        with pytest.raises(ValueError, match="speech_pad_duration must be non-negative"):
            VADConfig(speech_pad_duration=-0.1)

        with pytest.raises(ValueError, match="max_chunk_duration must be positive"):
            VADConfig(max_chunk_duration=0.0)


class TestPartialTranscriptionConfig:
    """Test PartialTranscriptionConfig validation."""

    def test_default_partial_config(self) -> None:
        """Test default partial transcription config."""
        config = PartialTranscriptionConfig()

        assert config.enabled is False
        assert config.model == "base.en"
        assert config.interval == 1.0
        assert config.max_buffer_seconds == 10.0

    def test_enabled_partial_config(self) -> None:
        """Test enabled partial transcription config."""
        config = PartialTranscriptionConfig(
            enabled=True,
            model="tiny.en",
            interval=0.5,
            max_buffer_seconds=5.0,
        )

        assert config.enabled is True
        assert config.model == "tiny.en"

    def test_partial_config_invalid_interval(self) -> None:
        """Test partial config rejects invalid interval."""
        with pytest.raises(ValueError, match="interval must be positive"):
            PartialTranscriptionConfig(interval=0.0)

    def test_partial_config_invalid_buffer_seconds(self) -> None:
        """Test partial config rejects invalid buffer seconds."""
        with pytest.raises(ValueError, match="max_buffer_seconds must be 1.0-60.0"):
            PartialTranscriptionConfig(max_buffer_seconds=0.5)

        with pytest.raises(ValueError, match="max_buffer_seconds must be 1.0-60.0"):
            PartialTranscriptionConfig(max_buffer_seconds=100.0)


class TestWhisperConfig:
    """Test WhisperConfig."""

    def test_default_whisper_config(self) -> None:
        """Test default Whisper configuration."""
        config = WhisperConfig()

        assert config.model == "turbo"
        assert config.device == "auto"
        assert config.require_gpu is False
        assert config.vad.aggressiveness == 2
        assert config.partial.enabled is False

    def test_custom_whisper_config(self) -> None:
        """Test custom Whisper configuration."""
        config = WhisperConfig(
            model="small",
            device="cuda",
            require_gpu=True,
            vad=VADConfig(aggressiveness=3),
            partial=PartialTranscriptionConfig(enabled=True, model="tiny.en"),
        )

        assert config.model == "small"
        assert config.device == "cuda"
        assert config.require_gpu is True
        assert config.vad.aggressiveness == 3
        assert config.partial.enabled is True


class TestRealtimeVADConfig:
    """Test RealtimeVADConfig validation."""

    def test_default_realtime_vad_config(self) -> None:
        """Test default Realtime VAD configuration."""
        config = RealtimeVADConfig()

        assert config.threshold == 0.2
        assert config.silence_duration_ms == 100

    def test_custom_realtime_vad_config(self) -> None:
        """Test custom Realtime VAD configuration."""
        config = RealtimeVADConfig(threshold=0.5, silence_duration_ms=500)

        assert config.threshold == 0.5
        assert config.silence_duration_ms == 500

    def test_realtime_vad_invalid_threshold(self) -> None:
        """Test Realtime VAD config rejects invalid threshold."""
        with pytest.raises(ValueError, match="threshold must be 0.0-1.0"):
            RealtimeVADConfig(threshold=1.5)

        with pytest.raises(ValueError, match="threshold must be 0.0-1.0"):
            RealtimeVADConfig(threshold=-0.1)

    def test_realtime_vad_invalid_silence_duration(self) -> None:
        """Test Realtime VAD config rejects invalid silence duration."""
        with pytest.raises(ValueError, match="silence_duration_ms must be 100-2000"):
            RealtimeVADConfig(silence_duration_ms=50)

        with pytest.raises(ValueError, match="silence_duration_ms must be 100-2000"):
            RealtimeVADConfig(silence_duration_ms=3000)


class TestRealtimeConfig:
    """Test RealtimeConfig."""

    def test_realtime_config_with_api_key(self) -> None:
        """Test Realtime configuration with API key."""
        config = RealtimeConfig(api_key="test-key-123")

        assert config.api_key == "test-key-123"
        assert config.model == "gpt-realtime-mini"
        assert config.chunk_duration == 2.0
        assert config.vad.threshold == 0.2

    def test_realtime_config_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test Realtime config does NOT read API key from environment (cli.py does)."""
        monkeypatch.setenv("OPENAI_API_KEY", "env-key-456")

        # Config should NOT automatically read from environment
        config = RealtimeConfig()
        assert config.api_key == ""  # Empty string, not None

    def test_realtime_config_missing_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test Realtime config allows missing API key (validation happens in factory)."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        # Config creation should succeed - validation happens in factory
        config = RealtimeConfig()
        assert config.api_key == ""  # Empty string, not None

    def test_custom_realtime_config(self) -> None:
        """Test custom Realtime configuration."""
        config = RealtimeConfig(
            api_key="test",
            model="gpt-realtime-preview",
            endpoint="wss://custom.endpoint.com",
            debug=True,
            vad=RealtimeVADConfig(threshold=0.5, silence_duration_ms=200),
        )

        assert config.model == "gpt-realtime-preview"
        assert config.endpoint == "wss://custom.endpoint.com"
        assert config.debug is True
        assert config.vad.threshold == 0.5
