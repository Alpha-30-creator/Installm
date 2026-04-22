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


def _llamacpp_available() -> bool:
    """Check if llama-cpp-python is installed."""
    try:
        import llama_cpp  # noqa: F401
        return True
    except ImportError:
        return False


def _ollama_available() -> bool:
    """Check if Ollama CLI is installed."""
    return shutil.which("ollama") is not None


def select_backend(model_id: str) -> str:
    """Auto-select the best backend for the current environment.

    Priority: vLLM (GPU) > Transformers (CPU/MPS) > llama.cpp (GGUF) > Ollama (fallback).
    GGUF files (.gguf extension) always route to llama.cpp.
    """
    # GGUF files always go to llama.cpp
    if model_id.endswith(".gguf"):
        if _llamacpp_available():
            return "llamacpp"
        raise RuntimeError(
            f"Model '{model_id}' is a GGUF file but llama-cpp-python is not installed.\n"
            "  pip install installm[llamacpp]"
        )

    sys = platform.system()

    # vLLM: best for GPU on Linux
    if sys == "Linux" and _has_nvidia_gpu() and _vllm_available():
        return "vllm"

    # Transformers: works everywhere (CPU, MPS, CUDA)
    try:
        import transformers  # noqa: F401
        return "transformers"
    except ImportError:
        pass

    # llama.cpp: lightweight CPU inference for GGUF models
    if _llamacpp_available():
        return "llamacpp"

    # Ollama: fallback if installed
    if _ollama_available():
        return "ollama"

    raise RuntimeError(
        "No suitable backend found. Install one of:\n"
        "  pip install installm[transformers]   (CPU/GPU)\n"
        "  pip install installm[llamacpp]       (CPU, GGUF models)\n"
        "  pip install installm[vllm]           (Linux + NVIDIA GPU)\n"
        "  Install Ollama from https://ollama.com"
    )
