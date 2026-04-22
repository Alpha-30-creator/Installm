"""Tests for the Transformers backend."""

import sys
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from installm.backends.transformers import TransformersBackend, _detect_device


def test_detect_device_no_torch():
    """_detect_device should return cpu when torch is not importable."""
    with patch.dict(sys.modules, {"torch": None}):
        device = _detect_device()
    assert device == "cpu"


def test_detect_device_cuda():
    """_detect_device should return cuda when CUDA is available."""
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True
    with patch.dict(sys.modules, {"torch": mock_torch}):
        device = _detect_device()
    assert device == "cuda"


def test_detect_device_cpu_fallback():
    """_detect_device should return cpu when no GPU is available."""
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    mock_torch.backends.mps.is_available.return_value = False
    with patch.dict(sys.modules, {"torch": mock_torch}):
        device = _detect_device()
    assert device == "cpu"


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
    mock_model.to = MagicMock(return_value=mock_model)

    mock_transformers = MagicMock()
    mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer
    mock_transformers.AutoModelForCausalLM.from_pretrained.return_value = mock_model

    with patch.dict(sys.modules, {"transformers": mock_transformers}), \
         patch("asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
        mock_thread.return_value = (mock_model, mock_tokenizer)
        await backend.load("gpt2", device="cpu")

    assert backend.model_id == "gpt2"
    assert backend.device == "cpu"


@pytest.mark.asyncio
async def test_generate():
    """generate() should return an OpenAI-compatible chat completion dict."""
    backend = TransformersBackend()
    backend.model_id = "gpt2"
    backend.device = "cpu"

    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token_id = 0
    mock_tokenizer.apply_chat_template.return_value = "Hello"
    mock_tokenizer.decode.return_value = "World"
    mock_tokenizer.encode.return_value = [1, 2, 3]
    mock_inputs = MagicMock()
    mock_inputs.__getitem__ = lambda self, key: MagicMock(shape=[1, 5])
    mock_tokenizer.return_value = mock_inputs

    backend.tokenizer = mock_tokenizer
    backend.model = MagicMock()

    mock_torch = MagicMock()
    mock_torch.no_grad.return_value.__enter__ = MagicMock(return_value=None)
    mock_torch.no_grad.return_value.__exit__ = MagicMock(return_value=False)

    with patch.dict(sys.modules, {"torch": mock_torch}), \
         patch("asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
        mock_thread.return_value = "World"
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
