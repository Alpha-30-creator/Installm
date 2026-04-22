"""Tests for POST /v1/embeddings endpoint."""

import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import AsyncMock, MagicMock

from installm.gateway.app import app, _backends


@pytest.fixture(autouse=True)
def clean_backends():
    _backends.clear()
    yield
    _backends.clear()


@pytest.mark.asyncio
async def test_embeddings_model_not_found():
    """Should return 404 when model is not loaded."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/v1/embeddings", json={
            "model": "nonexistent",
            "input": "hello",
        })
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_embeddings_not_supported():
    """Should return 400 when model does not support embeddings."""
    backend = MagicMock()
    backend.embed = AsyncMock(side_effect=NotImplementedError("not supported"))
    _backends["gpt2"] = backend

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/v1/embeddings", json={
            "model": "gpt2",
            "input": "hello",
        })
    assert resp.status_code == 400
    assert "does not support embeddings" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_embeddings_single_string():
    """Single string input should return one embedding object."""
    backend = MagicMock()
    backend.embed = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
    _backends["embed-model"] = backend

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/v1/embeddings", json={
            "model": "embed-model",
            "input": "hello world",
        })

    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "list"
    assert len(body["data"]) == 1
    assert body["data"][0]["object"] == "embedding"
    assert body["data"][0]["embedding"] == [0.1, 0.2, 0.3]
    assert body["data"][0]["index"] == 0


@pytest.mark.asyncio
async def test_embeddings_batch():
    """List input should return multiple embedding objects."""
    backend = MagicMock()
    backend.embed = AsyncMock(return_value=[[0.1, 0.2], [0.3, 0.4]])
    _backends["embed-model"] = backend

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/v1/embeddings", json={
            "model": "embed-model",
            "input": ["hello", "world"],
        })

    assert resp.status_code == 200
    body = resp.json()
    assert len(body["data"]) == 2
    assert body["data"][0]["index"] == 0
    assert body["data"][1]["index"] == 1
