"""Tests for the lazy ``sounddevice`` import helper."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType

import pytest


def _load_helper():
    """Import the helper module and reset its cache."""

    module = importlib.import_module("transcribe_demo.sound_device")
    module._SOUNDDEVICE_MODULE = None  # type: ignore[attr-defined]
    return module


def test_get_sounddevice_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    """The helper should cache the imported module."""

    fake_module = ModuleType("sounddevice")
    monkeypatch.setitem(sys.modules, "sounddevice", fake_module)

    helper = _load_helper()

    first = helper.get_sounddevice()
    monkeypatch.setitem(sys.modules, "sounddevice", ModuleType("sounddevice"))

    assert first is helper.get_sounddevice()


def test_get_sounddevice_missing_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    """A clear RuntimeError should be raised when PortAudio is unavailable."""

    helper = _load_helper()

    def _raise_import_error(*args, **kwargs):  # noqa: D401
        raise OSError("PortAudio library not found")

    monkeypatch.setitem(sys.modules, "sounddevice", None)
    monkeypatch.setattr(importlib, "import_module", _raise_import_error)

    with pytest.raises(RuntimeError, match="PortAudio"):
        helper.get_sounddevice()
