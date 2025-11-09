"""CPU-only stub replacement for the Triton package.

The real Triton project bundles GPU kernels used by PyTorch on CUDA hardware. Our automated tests run
on CPU-only machines, so `openai-whisper`'s dependency on Triton is satisfied with this stub to avoid
pulling in the massive CUDA runtime wheels.

Any attempt to access Triton's functionality should fail loudly so developers notice and can opt into
installing the genuine package instead.
"""

from __future__ import annotations

from typing import Any

__all__ = ["__getattr__", "__version__"]

__version__ = "2.0.0"


def __getattr__(name: str) -> Any:
    if name.startswith("__"):
        raise AttributeError(name)
    message = (
        "The Triton CPU stub is installed; GPU kernels are unavailable. "
        "Install the official 'triton' package to use CUDA-dependent features."
    )
    raise AttributeError(message)


def __dir__() -> list[str]:
    """Expose a minimal surface area when introspected."""
    return list(__all__)
