"""Tests for the Transformers backend."""

import sys
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from installm.backends.transformers import TransformersBackend, _detect_device


def test_detect_device_no_torch():
    """_detect_device returns cpu when torch is not importable."""
    with patch.dict(sys.modules, {"torch": None}):
        # Force re-import to pick up the patched module
        assert _detect_device() == "cpu"


def test_detect_device_cuda():
    """_detect_device returns cuda when CUDA is available."""
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True
    with patch.dict(sys.modules, {"torch": mock_torch}):
        assert _detect_device() == "cuda"


def test_detect_device_mps():
    """_detect_device returns mps when MPS is available but not CUDA."""
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    mock_torch.backends.mps.is_available.return_value = True
    with patch.dict(sys.modules, {"torch": mock_torch}):
        assert _detect_device() == "mps"


def test_detect_device_cpu_fallback():
    """_detect_device returns cpu when no GPU is available."""
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    mock_torch.backends.mps.is_available.return_value = False
    with patch.dict(sys.modules, {"torch": mock_torch}):
        assert _detect_device() == "cpu"


def test_backend_properties():
    """Transformers backend should not claim native tool/structured output support."""
    backend = TransformersBackend()
    assert backend.supports_tools is False
    assert backend.supports_structured_output is False


@pytest.mark.asyncio
async def test_load_no_transformers():
    """load() should raise if transformers is not installed."""
    backend = TransformersBackend()
    with patch.dict(sys.modules, {"transformers": None}):
        with pytest.raises(RuntimeError, match="transformers is not installed"):
            await backend.load("gpt2", device="cpu")


@pytest.mark.asyncio
async def test_load_success():
    """load() should set model_id and device after successful load."""
    backend = TransformersBackend()

    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token = None
    mock_tokenizer.eos_token = "<eos>"
    mock_model = MagicMock()

    # Patch asyncio.to_thread to return the mock model and tokenizer
    with patch("installm.backends.transformers.asyncio") as mock_asyncio:
        mock_asyncio.to_thread = AsyncMock(return_value=(mock_model, mock_tokenizer))
        await backend.load("gpt2", device="cpu")

    assert backend.model_id == "gpt2"
    assert backend.device == "cpu"
    assert backend.model is mock_model
    assert backend.tokenizer is mock_tokenizer
    assert mock_tokenizer.pad_token == "<eos>"


@pytest.mark.asyncio
async def test_generate():
    """generate() should return an OpenAI-compatible chat completion dict."""
    backend = TransformersBackend()
    backend.model_id = "gpt2"
    backend.device = "cpu"

    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token_id = 0
    mock_tokenizer.chat_template = None  # Force fallback prompt format
    mock_tokenizer.decode.return_value = "World"
    mock_tokenizer.encode.return_value = [1, 2, 3]

    # Mock tokenizer call (inputs)
    mock_input_ids = MagicMock()
    mock_input_ids.shape = [1, 5]
    mock_inputs = {"input_ids": mock_input_ids}
    mock_inputs_obj = MagicMock()
    mock_inputs_obj.__getitem__ = lambda self, key: mock_inputs[key]
    mock_inputs_obj.to.return_value = mock_inputs_obj
    mock_tokenizer.return_value = mock_inputs_obj

    backend.tokenizer = mock_tokenizer
    backend.model = MagicMock()

    with patch("installm.backends.transformers.asyncio") as mock_asyncio:
        mock_asyncio.to_thread = AsyncMock(return_value="World")
        result = await backend.generate([{"role": "user", "content": "Hi"}])

    assert result["object"] == "chat.completion"
    assert result["choices"][0]["message"]["role"] == "assistant"
    assert result["choices"][0]["message"]["content"] == "World"
    assert result["model"] == "gpt2"


@pytest.mark.asyncio
async def test_unload():
    """unload() should clear model and tokenizer references."""
    backend = TransformersBackend()
    backend.model = MagicMock()
    backend.tokenizer = MagicMock()

    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    with patch.dict(sys.modules, {"torch": mock_torch}):
        await backend.unload()

    assert backend.model is None
    assert backend.tokenizer is None


def test_format_prompt_with_chat_template():
    """_format_prompt uses chat template when available."""
    backend = TransformersBackend()
    backend.tokenizer = MagicMock()
    backend.tokenizer.chat_template = "some_template"
    backend.tokenizer.apply_chat_template.return_value = "formatted"

    result = backend._format_prompt([{"role": "user", "content": "Hi"}])
    assert result == "formatted"
    backend.tokenizer.apply_chat_template.assert_called_once()


def test_format_prompt_fallback():
    """_format_prompt falls back to simple format when no chat template."""
    backend = TransformersBackend()
    backend.tokenizer = MagicMock()
    backend.tokenizer.chat_template = None

    result = backend._format_prompt([
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hi"},
    ])
    assert "system: You are helpful." in result
    assert "user: Hi" in result
    assert result.endswith("assistant:")
