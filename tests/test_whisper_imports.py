"""Tests for lazy imports in ``whisper_backend``."""

from __future__ import annotations

import importlib
import sys

import pytest


def _reload_backend(monkeypatch: pytest.MonkeyPatch):
    """Reload ``transcribe_demo.whisper_backend`` with a clean module cache."""

    monkeypatch.delitem(sys.modules, "transcribe_demo.whisper_backend", raising=False)
    return importlib.import_module("transcribe_demo.whisper_backend")


def test_import_without_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Importing should succeed even when torch/whisper are unavailable."""

    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "whisper", None)

    module = _reload_backend(monkeypatch)

    # The module should expose WebRTCVAD without importing torch/whisper eagerly.
    assert hasattr(module, "WebRTCVAD")


def test_run_requires_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    """The runtime function should raise a helpful error when torch is missing."""

    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "whisper", None)

    module = _reload_backend(monkeypatch)

    with pytest.raises(RuntimeError, match="PyTorch is required"):
        module.run_whisper_transcriber(
            model_name="tiny",
            sample_rate=16000,
            channels=1,
            temp_file=None,
            ca_cert=None,
            insecure_downloads=False,
            device_preference="auto",
            require_gpu=False,
            chunk_consumer=None,
        )
