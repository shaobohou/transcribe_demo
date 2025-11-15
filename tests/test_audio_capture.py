"""Tests for AudioCaptureManager."""

from __future__ import annotations

from unittest.mock import Mock

import numpy as np

from transcribe_demo.audio_capture import AudioCaptureManager


def test_audio_callback_accepts_positional_args():
    """
    Test that _audio_callback accepts positional arguments.

    This is critical because sounddevice.InputStream calls callbacks with
    positional arguments. If _audio_callback is made keyword-only (with *),
    it will fail at runtime with:
    TypeError: _audio_callback() takes 1 positional argument but 5 were given

    This test ensures the callback signature remains compatible.
    """
    manager = AudioCaptureManager(
        sample_rate=16000,
        channels=1,
        max_capture_duration=0.0,
        collect_full_audio=False,
    )

    # Create dummy callback arguments like sounddevice would pass
    indata = np.zeros((480, 1), dtype=np.float32)
    frames = 480
    time_info = Mock()  # Mock the ctypes.Structure object
    status = None

    # This should NOT raise TypeError
    # sounddevice calls: callback(indata, frames, time, status)
    # NOT: callback(indata=indata, frames=frames, time=time, status=status)
    manager._audio_callback(indata, frames, time_info, status)

    # Verify the callback worked (queue should have received the data)
    assert not manager.audio_queue.empty()
    received = manager.audio_queue.get_nowait()
    assert received is not None
    assert isinstance(received, np.ndarray)
    assert received.shape == (480, 1)
