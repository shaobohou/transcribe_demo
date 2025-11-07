"""Pytest configuration and fixtures for transcribe_demo tests.

This module mocks sounddevice and AudioCaptureManager at import time to prevent
audio device access in CI environments where no audio hardware is available.
"""

from __future__ import annotations

import queue as queue_module
import sys
import threading
import types

import numpy as np
import pytest

# Mock sounddevice BEFORE any test modules are imported
# This must happen at module level (not in a fixture) because pytest imports
# test modules during collection, before any fixtures run.


# Mock InputStream class - needs to be a class that can be instantiated
class MockInputStream:
    """Mock sounddevice InputStream to prevent audio device access in tests."""

    def __init__(self, *args, **kwargs):
        # Store callback for potential inspection
        self.callback = kwargs.get('callback', None)
        self.channels = kwargs.get('channels', 1)
        self.samplerate = kwargs.get('samplerate', 16000)
        self.dtype = kwargs.get('dtype', 'float32')

    def start(self):
        """Mock start - does nothing."""
        pass

    def stop(self):
        """Mock stop - does nothing."""
        pass

    def close(self):
        """Mock close - does nothing."""
        pass

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        self.close()
        return False


# Mock PortAudioError exception - this needs to be a real exception class
class MockPortAudioError(Exception):
    """Mock exception for PortAudio errors."""
    pass


# Create a simple fake module without using unittest.mock
# This avoids any potential issues with Mock object behavior
class FakeSoundDeviceModule(types.ModuleType):
    """Fake sounddevice module for testing without audio hardware."""

    def __init__(self):
        super().__init__('sounddevice')
        self.InputStream = MockInputStream
        self.PortAudioError = MockPortAudioError

    def query_devices(self, *args, **kwargs):
        """Mock query_devices - returns empty list."""
        return []

    def default_device(self, *args, **kwargs):
        """Mock default device query - returns None."""
        return None


# Inject the mock into sys.modules
# This happens as soon as conftest.py is imported by pytest
mock_sd = FakeSoundDeviceModule()
sys.modules['sounddevice'] = mock_sd
sys.modules['sd'] = mock_sd  # Also mock 'sd' in case it's imported differently


# Now that sounddevice is mocked, we can safely import audio_capture
from transcribe_demo import audio_capture


# Create a dummy AudioCaptureManager that won't hang in CI
class DummyAudioCaptureManager:
    """
    Dummy AudioCaptureManager for CI environments.

    This provides a safe default that won't try to access audio hardware.
    Individual tests should override this with FakeAudioCaptureManager using monkeypatch.
    """

    def __init__(
        self,
        sample_rate: int,
        channels: int,
        max_capture_duration: float = 0.0,
        collect_full_audio: bool = True,
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.max_capture_duration = max_capture_duration
        self.collect_full_audio = collect_full_audio
        self.audio_queue: queue_module.Queue[np.ndarray | None] = queue_module.Queue()
        self.stop_event = threading.Event()
        self.capture_limit_reached = threading.Event()

        # Immediately signal that we're stopped - prevents hanging
        self.stop_event.set()

    def start(self) -> None:
        """Mock start - immediately stops to prevent hanging."""
        self.stop_event.set()
        # Put None to signal end of stream
        self.audio_queue.put(None)

    def stop(self) -> None:
        """Mock stop."""
        self.stop_event.set()

    def wait_until_stopped(self) -> None:
        """Mock wait - returns immediately since we're always stopped."""
        pass

    def close(self) -> None:
        """Mock close."""
        pass

    def get_full_audio(self) -> np.ndarray:
        """Return empty audio."""
        return np.zeros(0, dtype=np.float32)

    def get_capture_duration(self) -> float:
        """Return zero duration."""
        return 0.0

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        self.close()
        return False


# Replace AudioCaptureManager with the dummy version
audio_capture.AudioCaptureManager = DummyAudioCaptureManager


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """
    Session-level fixture to ensure the test environment is properly configured.

    This is autouse=True to ensure it runs before any tests, providing a safe
    environment where AudioCaptureManager won't try to access audio hardware.
    """
    # The mocking is already done at module level above
    # This fixture just provides a hook for any additional setup if needed
    yield

