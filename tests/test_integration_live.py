"""Live integration tests using a real tiny model (sshleifer/tiny-gpt2).

These tests require torch and transformers to be installed.
They are skipped automatically when those packages are not available.
Run with:  pytest tests/test_integration_live.py -v -s

What is tested:
  1. TransformersBackend can load a real tiny model
  2. generate() returns a valid OpenAI-shaped response
  3. stream() yields valid chunk dicts
  4. The full FastAPI gateway serves /v1/chat/completions with a real backend
  5. Streaming endpoint returns proper SSE chunks
"""

import json
import pytest
import asyncio

# Skip the entire module if torch or transformers are not installed
pytest.importorskip("torch")
pytest.importorskip("transformers")

from httpx import AsyncClient, ASGITransport
from installm.backends.transformers import TransformersBackend
from installm.gateway.app import app, _backends, register_backend

MODEL = "sshleifer/tiny-gpt2"  # ~5 MB, runs on CPU in seconds


@pytest.fixture(scope="module")
def event_loop():
    """Use a single event loop for all module-scoped async fixtures."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
async def loaded_backend():
    """Load the tiny model once for all tests in this module."""
    backend = TransformersBackend()
    await backend.load(MODEL, device="cpu")
    yield backend
    await backend.unload()


@pytest.fixture(scope="module", autouse=True)
async def register_live_backend(loaded_backend):
    """Register the live backend in the gateway registry."""
    _backends.clear()
    register_backend(MODEL, loaded_backend)
    yield
    _backends.clear()


# ---------------------------------------------------------------------------
# Backend-level tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_live_generate(loaded_backend):
    """generate() should return a valid OpenAI-shaped dict with real content."""
    result = await loaded_backend.generate(
        [{"role": "user", "content": "Hello"}],
        max_tokens=20,
    )
    assert result["object"] == "chat.completion"
    assert result["model"] == MODEL
    choices = result["choices"]
    assert len(choices) == 1
    assert choices[0]["message"]["role"] == "assistant"
    content = choices[0]["message"]["content"]
    assert isinstance(content, str)
    assert len(content) > 0
    print(f"\n[live] generate output: {content!r}")


@pytest.mark.asyncio
async def test_live_stream(loaded_backend):
    """stream() should yield at least one chunk with content."""
    chunks = []
    async for chunk in loaded_backend.stream(
        [{"role": "user", "content": "Hello"}],
        max_tokens=20,
    ):
        chunks.append(chunk)

    assert len(chunks) >= 1
    # At least one chunk should have content
    content_chunks = [
        c for c in chunks
        if c.get("choices", [{}])[0].get("delta", {}).get("content")
    ]
    assert len(content_chunks) >= 1
    print(f"\n[live] stream chunks: {len(chunks)}, content chunks: {len(content_chunks)}")


# ---------------------------------------------------------------------------
# Gateway-level tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_live_gateway_chat():
    """Gateway /v1/chat/completions should return a valid response with real model."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/v1/chat/completions", json={
            "model": MODEL,
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 20,
        })

    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == MODEL
    content = body["choices"][0]["message"]["content"]
    assert isinstance(content, str) and len(content) > 0
    print(f"\n[live] gateway chat output: {content!r}")


@pytest.mark.asyncio
async def test_live_gateway_streaming():
    """Gateway streaming should return SSE chunks with real content."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        async with client.stream("POST", "/v1/chat/completions", json={
            "model": MODEL,
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
            "max_tokens": 20,
        }) as resp:
            assert resp.status_code == 200
            lines = []
            async for line in resp.aiter_lines():
                lines.append(line)

    data_lines = [l for l in lines if l.startswith("data: ") and l != "data: [DONE]"]
    assert len(data_lines) >= 1
    assert "data: [DONE]" in lines

    # Verify chunk structure
    first = json.loads(data_lines[0][6:])
    assert first["object"] == "chat.completion.chunk"
    print(f"\n[live] gateway stream: {len(data_lines)} data lines")


@pytest.mark.asyncio
async def test_live_models_endpoint():
    """GET /v1/models should list the loaded model."""
    from unittest.mock import patch
    from installm.config import list_models

    mock_models = {MODEL: {"added_at": 1700000000, "backend": "transformers"}}
    transport = ASGITransport(app=app)
    with patch("installm.gateway.routes.models.list_models", return_value=mock_models):
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/v1/models")

    assert resp.status_code == 200
    body = resp.json()
    ids = [m["id"] for m in body["data"]]
    assert MODEL in ids
