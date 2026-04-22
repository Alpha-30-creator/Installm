"""Tests for the Ollama backend."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from installm.backends.ollama import OllamaBackend


@pytest.fixture
def backend():
    return OllamaBackend(base_url="http://localhost:11434")


def test_backend_properties(backend):
    """Ollama backend should declare tool and structured output support."""
    assert backend.supports_tools is True
    assert backend.supports_structured_output is True


@pytest.mark.asyncio
async def test_load_no_ollama(backend):
    """load() should raise if ollama CLI is not installed."""
    with patch("shutil.which", return_value=None):
        with pytest.raises(RuntimeError, match="Ollama is not installed"):
            await backend.load("llama3")


@pytest.mark.asyncio
async def test_load_success(backend):
    """load() should run ollama pull and set model_name."""
    mock_proc = AsyncMock()
    mock_proc.returncode = 0
    mock_proc.communicate = AsyncMock(return_value=(b"", b""))

    with patch("shutil.which", return_value="/usr/bin/ollama"), \
         patch("asyncio.create_subprocess_exec", return_value=mock_proc):
        await backend.load("llama3")

    assert backend.model_name == "llama3"


@pytest.mark.asyncio
async def test_load_pull_failure(backend):
    """load() should raise if ollama pull fails."""
    mock_proc = AsyncMock()
    mock_proc.returncode = 1
    mock_proc.communicate = AsyncMock(return_value=(b"", b"error message"))

    with patch("shutil.which", return_value="/usr/bin/ollama"), \
         patch("asyncio.create_subprocess_exec", return_value=mock_proc):
        with pytest.raises(RuntimeError, match="ollama pull failed"):
            await backend.load("badmodel")


@pytest.mark.asyncio
async def test_generate(backend):
    """generate() should POST to /v1/chat/completions and return JSON."""
    backend.model_name = "llama3"
    expected = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "choices": [{"message": {"role": "assistant", "content": "Hello!"}}],
    }

    mock_response = MagicMock()
    mock_response.json.return_value = expected
    mock_response.raise_for_status = MagicMock()

    backend._client = AsyncMock()
    backend._client.post = AsyncMock(return_value=mock_response)

    result = await backend.generate([{"role": "user", "content": "Hi"}])
    assert result["choices"][0]["message"]["content"] == "Hello!"


@pytest.mark.asyncio
async def test_unload(backend):
    """unload() should close the HTTP client."""
    backend._client = AsyncMock()
    await backend.unload()
    backend._client.aclose.assert_called_once()
