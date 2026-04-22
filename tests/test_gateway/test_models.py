"""Tests for GET /v1/models endpoint."""

import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import patch

from installm.gateway.app import app


@pytest.mark.asyncio
async def test_models_empty():
    """GET /v1/models with no models should return an empty list."""
    with patch("installm.gateway.routes.models.list_models", return_value={}):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/v1/models")

    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "list"
    assert body["data"] == []


@pytest.mark.asyncio
async def test_models_with_entries():
    """GET /v1/models should return all registered models in OpenAI shape."""
    mock_models = {
        "gpt2": {"added_at": 1700000000, "backend": "transformers"},
        "llama3": {"added_at": 1700000001, "backend": "ollama"},
    }
    with patch("installm.gateway.routes.models.list_models", return_value=mock_models):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/v1/models")

    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "list"
    ids = [m["id"] for m in body["data"]]
    assert "gpt2" in ids
    assert "llama3" in ids
    for m in body["data"]:
        assert m["object"] == "model"
        assert "created" in m
        assert m["owned_by"] == "installm"
