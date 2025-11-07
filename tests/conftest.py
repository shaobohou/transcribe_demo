"""Pytest configuration and fixtures for transcribe_demo tests.

This module mocks sounddevice at import time to prevent audio device access
in CI environments where no audio hardware is available.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

# Mock sounddevice BEFORE any test modules are imported
# This must happen at module level (not in a fixture) because pytest imports
# test modules during collection, before any fixtures run. When test modules
# import from transcribe_demo, audio_capture.py tries to import sounddevice,
# which fails in CI environments without audio hardware.

mock_sd = MagicMock()


# Mock InputStream class - needs to be a class that can be instantiated
class MockInputStream:
    def __init__(self, *args, **kwargs):
        pass

    def start(self):
        pass

    def stop(self):
        pass

    def close(self):
        pass


mock_sd.InputStream = MockInputStream


# Mock PortAudioError exception - this needs to be a real exception class
class MockPortAudioError(Exception):
    pass


mock_sd.PortAudioError = MockPortAudioError

# Inject the mock into sys.modules
# This happens as soon as conftest.py is imported by pytest
sys.modules['sounddevice'] = mock_sd
