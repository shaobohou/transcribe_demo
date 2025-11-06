from __future__ import annotations

from types import ModuleType

_SOUNDDEVICE_MODULE: ModuleType | None = None


def get_sounddevice() -> ModuleType:
    """Return the cached ``sounddevice`` module, importing it on demand.

    The ``sounddevice`` package raises ``OSError`` when PortAudio is not
    available on the host. Import the module lazily so unit tests that only
    exercise non-audio code paths can run without PortAudio, and surface a
    clear runtime error when microphone capture is actually requested.
    """

    global _SOUNDDEVICE_MODULE
    if _SOUNDDEVICE_MODULE is not None:
        return _SOUNDDEVICE_MODULE

    try:
        import sounddevice as sd  # pylint: disable=import-outside-toplevel
    except (ImportError, OSError) as exc:  # pragma: no cover - depends on host
        raise RuntimeError(
            "sounddevice/PortAudio is required for microphone capture. "
            "Install PortAudio (e.g. `apt-get install libportaudio2`) and "
            "retry."
        ) from exc

    _SOUNDDEVICE_MODULE = sd
    return sd
