"""Pytest configuration for transcribe_demo tests."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

# Mock sounddevice at import time to prevent audio device queries in CI
# This must happen before any transcribe_demo modules are imported
if 'sounddevice' not in sys.modules:
    mock_sd = MagicMock()

    # Mock InputStream as a class that can be instantiated
    mock_stream_class = MagicMock()
    mock_stream_instance = MagicMock()
    mock_stream_class.return_value = mock_stream_instance
    mock_sd.InputStream = mock_stream_class

    # Mock common functions that might be called
    mock_sd.query_devices = MagicMock(return_value=[])
    mock_sd.default = MagicMock()

    # Replace sounddevice in sys.modules before other imports
    sys.modules['sounddevice'] = mock_sd
