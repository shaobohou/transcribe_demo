# Triton CPU Stub

This is a minimal stub package that satisfies the `triton` dependency declared by `openai-whisper`.
It intentionally provides no runtime functionality because the Transcribe Demo project only targets
CPU execution for its automated tests. Installing the official Triton wheels pulls in CUDA runtime
artifacts that significantly slow down fresh environment bootstraps, so this stub keeps CI lean while
still allowing Whisper to run on CPU.

If you need actual Triton kernels for GPU execution, remove the stub override in `pyproject.toml` and
let `uv` resolve the official package from PyPI or the PyTorch wheel indexes instead.
