import numpy as np

from transcribe_demo.realtime_backend import float_to_pcm16


def test_float_to_pcm16_clamps_and_scales():
    samples = np.array([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5], dtype=np.float32)
    pcm = np.frombuffer(float_to_pcm16(audio=samples), dtype=np.int16)

    # Values beyond range should be clipped (accounting for symmetric rounding)
    assert pcm[0] <= -32767
    assert pcm[-1] >= 32767

    # Representative midpoints
    assert pcm[2] == -16383
    assert pcm[4] == 16383
