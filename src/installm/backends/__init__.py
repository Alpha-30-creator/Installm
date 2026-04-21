"""Backend abstraction layer for InstaLLM."""

import platform
import shutil


def _has_nvidia_gpu() -> bool:
    """Check if an NVIDIA GPU is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _has_mps() -> bool:
    """Check if Apple Metal (MPS) is available."""
    try:
        import torch
        return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    except ImportError:
        return False


def _vllm_available() -> bool:
    """Check if vLLM is installed and usable."""
    try:
        import vllm  # noqa: F401
        return True
    except ImportError:
        return False


def _ollama_available() -> bool:
    """Check if Ollama CLI is installed."""
    return shutil.which("ollama") is not None


def select_backend(model_id: str) -> str:
    """Auto-select the best backend for the current environment.

    Priority: vLLM (GPU) > Transformers (CPU/MPS) > Ollama (fallback).
    """
    sys = platform.system()

    # vLLM: best for GPU on Linux (not supported on Windows/macOS)
    if sys == "Linux" and _has_nvidia_gpu() and _vllm_available():
        return "vllm"

    # Transformers: works everywhere (CPU, MPS, CUDA)
    try:
        import transformers  # noqa: F401
        return "transformers"
    except ImportError:
        pass

    # Ollama: fallback if installed
    if _ollama_available():
        return "ollama"

    raise RuntimeError(
        "No suitable backend found. Install one of:\n"
        "  pip install installm[transformers]   (CPU/GPU)\n"
        "  pip install installm[vllm]           (Linux + NVIDIA GPU)\n"
        "  Install Ollama from https://ollama.com"
    )
