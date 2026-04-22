"""Tests for the health endpoint."""

import pytest
from httpx import AsyncClient, ASGITransport
from installm.gateway.app import app


@pytest.mark.asyncio
async def test_health():
    """GET /health should return ok status."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "models_loaded" in body
