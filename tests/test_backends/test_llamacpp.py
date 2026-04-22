"""Tests for the llama.cpp backend."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from installm.backends.llamacpp import LlamaCppBackend


@pytest.fixture
def backend():
    return LlamaCppBackend()


def test_properties(backend):
    """Verify backend metadata."""
    assert backend.name == "llamacpp"
    assert backend.supports_tools is False
    assert backend.supports_structured_output is False


@pytest.mark.asyncio
async def test_load(backend):
    """load() should call Llama with the resolved path."""
    mock_llama_cls = MagicMock(return_value=MagicMock())

    with patch("installm.backends.llamacpp.asyncio.to_thread", new_callable=AsyncMock) as mock_thread, \
         patch.object(backend, "_resolve_model_path", return_value="/tmp/model.gguf"):
        mock_thread.return_value = mock_llama_cls()
        await backend.load("some-model.gguf")

    assert backend.model is not None
    assert backend.model_id == "some-model.gguf"


@pytest.mark.asyncio
async def test_unload(backend):
    """unload() should clear model and model_id."""
    backend.model = MagicMock()
    backend.model_id = "test"
    await backend.unload()
    assert backend.model is None
    assert backend.model_id is None


@pytest.mark.asyncio
async def test_generate(backend):
    """generate() should call create_chat_completion and return the result."""
    mock_result = {
        "choices": [{"message": {"role": "assistant", "content": "Hello!"}}]
    }
    backend.model = MagicMock()

    with patch("installm.backends.llamacpp.asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
        mock_thread.return_value = mock_result
        result = await backend.generate([{"role": "user", "content": "Hi"}])

    assert result["choices"][0]["message"]["content"] == "Hello!"


@pytest.mark.asyncio
async def test_stream(backend):
    """stream() should yield chunks from the model."""
    chunks = [
        {"choices": [{"delta": {"content": "Hel"}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": "lo"}, "finish_reason": None}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
    ]
    chunk_iter = iter(chunks)

    backend.model = MagicMock()

    async def mock_to_thread(fn, *args, **kwargs):
        # First call is create_chat_completion (returns the generator)
        # Subsequent calls are _next
        if fn == backend.model.create_chat_completion:
            return chunk_iter
        # _next calls
        return fn(*args)

    with patch("installm.backends.llamacpp.asyncio.to_thread", side_effect=mock_to_thread):
        collected = []
        async for c in backend.stream([{"role": "user", "content": "Hi"}]):
            collected.append(c)

    assert len(collected) == 3
    assert collected[0]["choices"][0]["delta"]["content"] == "Hel"


def test_resolve_gguf_path(backend, tmp_path):
    """_resolve_model_path should return direct .gguf paths."""
    gguf = tmp_path / "model.gguf"
    gguf.write_text("fake")
    result = backend._resolve_model_path(str(gguf))
    assert result == str(gguf)


def test_resolve_missing_raises(backend):
    """_resolve_model_path should raise for non-existent models."""
    with pytest.raises(FileNotFoundError, match="No .gguf model found"):
        backend._resolve_model_path("nonexistent/model")
