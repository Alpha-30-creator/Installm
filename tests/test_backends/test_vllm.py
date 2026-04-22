"""Tests for the vLLM backend.

These tests mock vLLM and CUDA entirely so they run on any machine.
Real GPU integration is validated manually on a Linux + NVIDIA machine.
"""

import platform
import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from installm.backends.vllm import VLLMBackend, _check_platform


def test_check_platform_non_linux():
    """_check_platform should raise on non-Linux systems."""
    with patch("platform.system", return_value="Windows"):
        with pytest.raises(RuntimeError, match="not supported on Windows"):
            _check_platform()


def test_check_platform_no_cuda():
    """_check_platform should raise when no CUDA GPU is present."""
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    with patch("platform.system", return_value="Linux"), \
         patch.dict("sys.modules", {"torch": mock_torch}):
        with pytest.raises(RuntimeError, match="NVIDIA GPU"):
            _check_platform()


def test_check_platform_ok():
    """_check_platform should pass on Linux with CUDA."""
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True
    with patch("platform.system", return_value="Linux"), \
         patch.dict("sys.modules", {"torch": mock_torch}):
        _check_platform()  # Should not raise


def test_backend_properties():
    """vLLM backend should declare tool and structured output support."""
    backend = VLLMBackend()
    assert backend.supports_tools is True
    assert backend.supports_structured_output is True


@pytest.mark.asyncio
async def test_load_not_linux():
    """load() should raise on non-Linux platforms."""
    backend = VLLMBackend()
    with patch("platform.system", return_value="Darwin"):
        with pytest.raises(RuntimeError, match="not supported"):
            await backend.load("meta-llama/Llama-3-8B")


@pytest.mark.asyncio
async def test_load_vllm_not_installed():
    """load() should raise with a helpful message if vLLM is not installed."""
    backend = VLLMBackend()
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True

    with patch("platform.system", return_value="Linux"), \
         patch.dict("sys.modules", {"torch": mock_torch, "vllm": None}):
        with pytest.raises(RuntimeError, match="vLLM is not installed"):
            await backend.load("meta-llama/Llama-3-8B")


@pytest.mark.asyncio
async def test_embed_not_supported():
    """embed() should raise NotImplementedError."""
    backend = VLLMBackend()
    with pytest.raises(NotImplementedError, match="not supported"):
        await backend.embed(["hello"])


@pytest.mark.asyncio
async def test_unload():
    """unload() should clear the engine reference."""
    backend = VLLMBackend()
    backend.engine = MagicMock()
    await backend.unload()
    assert backend.engine is None


def test_messages_to_prompt():
    """_messages_to_prompt should format messages into a prompt string."""
    backend = VLLMBackend()
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
    ]
    prompt = backend._messages_to_prompt(messages)
    assert "system: You are helpful." in prompt
    assert "user: Hello" in prompt
    assert prompt.endswith("assistant:")
