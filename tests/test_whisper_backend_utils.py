import os
import ssl

import numpy as np
import pytest

from transcribe_demo import whisper_backend


class DummyModel:
    def __init__(self, expected_audio):
        self.calls = []
        self.expected_audio = expected_audio

    def transcribe(self, audio, **kwargs):
        # Record a shallow copy to verify parameters without mutating caller data.
        self.calls.append({"audio": np.array(audio), "kwargs": kwargs})
        assert np.array_equal(audio, self.expected_audio)
        return {"text": "dummy transcript"}


def test_transcribe_full_audio_returns_empty_for_no_samples():
    assert (
        whisper_backend.transcribe_full_audio(
            audio=np.zeros(0, dtype=np.float32),
            sample_rate=16000,
            model_name="tiny",
            device_preference="auto",
            require_gpu=False,
            ca_cert=None,
            disable_ssl_verify=False,
        )
        == ""
    )


def test_transcribe_full_audio_invokes_model(monkeypatch):
    expected_audio = np.array([0.1, -0.2, 0.05], dtype=np.float32)
    dummy_model = DummyModel(expected_audio)

    def fake_load_model(**kwargs):
        # Return dummy model, report cpu device and fp16 disabled.
        return dummy_model, "cpu", False

    monkeypatch.setattr(whisper_backend, "load_whisper_model", fake_load_model)

    result = whisper_backend.transcribe_full_audio(
        audio=np.array(expected_audio, copy=True),
        sample_rate=16000,
        model_name="small",
        device_preference="cpu",
        require_gpu=False,
        ca_cert=None,
        disable_ssl_verify=False,
        language="en",
    )
    assert result == "dummy transcript"
    assert dummy_model.calls
    call = dummy_model.calls[0]
    assert call["kwargs"]["language"] == "en"
    assert not call["kwargs"]["fp16"]


def test_load_whisper_model_prefers_cuda(monkeypatch):
    loaded_model = object()

    monkeypatch.setattr(whisper_backend.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(whisper_backend, "_mps_available", lambda: False)
    monkeypatch.setattr(whisper_backend.whisper, "load_model", lambda name, device: loaded_model)

    model, device, fp16 = whisper_backend.load_whisper_model(
        model_name="base",
        device_preference="auto",
        require_gpu=False,
        ca_cert=None,
        disable_ssl_verify=False,
    )
    assert model is loaded_model
    assert device == "cuda"
    assert fp16 is True


def test_load_whisper_model_requires_gpu(monkeypatch):
    monkeypatch.setattr(whisper_backend.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(whisper_backend, "_mps_available", lambda: False)

    with pytest.raises(RuntimeError, match="GPU expected"):
        whisper_backend.load_whisper_model(
            model_name="tiny",
            device_preference="auto",
            require_gpu=True,
            ca_cert=None,
            disable_ssl_verify=False,
        )


def test_load_whisper_model_sets_ca_and_ssl(tmp_path, monkeypatch):
    monkeypatch.setattr(whisper_backend.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(whisper_backend, "_mps_available", lambda: False)

    loaded_model = object()

    def fake_load_model(name, device):
        # Simulate download path invoking the HTTPS context.
        whisper_backend.ssl._create_default_https_context()
        return loaded_model

    monkeypatch.setattr(whisper_backend.whisper, "load_model", fake_load_model)

    ca_file = tmp_path / "ca.pem"
    ca_file.write_text("certificate data")

    original_context = ssl._create_default_https_context
    context_calls = []

    def fake_unverified_context():
        context_calls.append("called")
        return original_context()

    monkeypatch.setattr(ssl, "_create_unverified_context", fake_unverified_context)
    monkeypatch.setattr(whisper_backend.ssl, "_create_unverified_context", fake_unverified_context)

    model, device, fp16 = whisper_backend.load_whisper_model(
        model_name="tiny",
        device_preference="cpu",
        require_gpu=False,
        ca_cert=ca_file,
        disable_ssl_verify=True,
    )

    assert model is loaded_model
    assert device == "cpu"
    assert fp16 is False
    assert os.environ["SSL_CERT_FILE"] == str(ca_file)
    assert os.environ["REQUESTS_CA_BUNDLE"] == str(ca_file)
    assert context_calls  # insecure download swapped contexts
