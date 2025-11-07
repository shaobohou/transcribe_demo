"""Pytest configuration and fixtures for transcribe_demo tests."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest


@pytest.fixture(scope="session", autouse=True)
def mock_sounddevice():
    """
    Mock sounddevice globally to prevent audio device access in CI environments.

    This fixture is autouse=True with session scope, meaning it runs once before
    any tests and applies the mock globally. This prevents sounddevice from trying
    to query audio devices during module imports, which fails in CI environments
    that don't have audio hardware.

    Individual tests that need sounddevice functionality should use monkeypatch
    to replace audio_capture.AudioCaptureManager with FakeAudioCaptureManager.
    """
    # Create a mock for sounddevice
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

    # Inject the mock into sys.modules before sounddevice is imported
    # This needs to happen before any test module imports audio_capture
    sys.modules['sounddevice'] = mock_sd

    yield mock_sd

    # Cleanup: restore original sounddevice if it was imported
    # (though in practice we don't need to restore for tests)
