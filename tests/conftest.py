"""Pytest configuration and fixtures for transcribe_demo tests.

This module mocks sounddevice at import time to prevent audio device access
in CI environments where no audio hardware is available.
"""

from __future__ import annotations

import sys
import tempfile
import types
from pathlib import Path

import pytest

# Mock sounddevice BEFORE any test modules are imported
# This must happen at module level (not in a fixture) because pytest imports
# test modules during collection, before any fixtures run. When test modules
# import from transcribe_demo, audio_capture.py tries to import sounddevice,
# which fails in CI environments without audio hardware.


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


# Inject the mock into sys.modules at module level
# This happens immediately when conftest.py is imported by pytest, before test
# collection begins. This ensures sounddevice is already mocked when test modules
# import backend code (which imports audio_capture, which imports sounddevice).
mock_sd = FakeSoundDeviceModule()
sys.modules['sounddevice'] = mock_sd
sys.modules['sd'] = mock_sd  # Also mock 'sd' in case it's imported differently

# NOTE: We do NOT replace AudioCaptureManager here. Backend modules import the
# audio_capture module (not the class directly) with:
#   from transcribe_demo import audio_capture as audio_capture_lib
# This allows tests to monkeypatch audio_capture_lib.AudioCaptureManager with
# FakeAudioCaptureManager. With sounddevice mocked above, the real AudioCaptureManager
# can be imported safely without triggering audio hardware access.


@pytest.fixture(autouse=True)
def mock_stdin_for_audio_capture(monkeypatch):
    """
    Ensure stdin.isatty() returns False to prevent stdin listener threads.

    This prevents AudioCaptureManager from starting a stdin listener thread
    that would block on input() in CI environments. Applies to all tests
    automatically via autouse=True.
    """
    # Create a mock stdin that always reports it's not a TTY
    class MockStdin:
        def isatty(self):
            return False

    monkeypatch.setattr(sys, 'stdin', MockStdin())


@pytest.fixture
def temp_session_dir():
    """
    Provide a temporary directory for session logging in tests.

    The directory is automatically cleaned up after the test completes,
    ensuring no session logs are left on disk after tests run.

    Usage:
        def test_something(temp_session_dir):
            session_logger = SessionLogger(
                output_dir=temp_session_dir,
                ...
            )
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)
