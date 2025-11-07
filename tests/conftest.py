"""Pytest configuration and fixtures for transcribe_demo tests.

This module mocks sounddevice at import time to prevent audio device access
in CI environments where no audio hardware is available.
"""

from __future__ import annotations

import sys
import types

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


# Inject the mock into sys.modules
# This happens as soon as conftest.py is imported by pytest
mock_sd = FakeSoundDeviceModule()
sys.modules['sounddevice'] = mock_sd
sys.modules['sd'] = mock_sd  # Also mock 'sd' in case it's imported differently
