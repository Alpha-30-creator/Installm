"""Tests for API key authentication."""

import os
import pytest
from installm.auth import create_key, revoke_key, list_keys, validate_key, has_keys, _hash


class TestAuthKeyManagement:
    """Test key creation, listing, and revocation."""

    def test_create_key_returns_full_key_and_id(self, tmp_state):
        full_key, key_id = create_key()
        assert full_key.startswith("sk-installm-")
        assert len(key_id) == 8
        assert len(full_key) == len("sk-installm-") + 48  # 24 bytes = 48 hex

    def test_create_key_with_label(self, tmp_state):
        full_key, key_id = create_key(label="test-laptop")
        keys = list_keys()
        assert key_id in keys
        assert keys[key_id]["label"] == "test-laptop"

    def test_create_key_stores_hash_not_plaintext(self, tmp_state):
        full_key, key_id = create_key()
        keys = list_keys()
        stored = keys[key_id]
        assert stored["hash"] == _hash(full_key)
        assert full_key not in str(stored)  # Full key never stored

    def test_create_key_stores_prefix(self, tmp_state):
        full_key, key_id = create_key()
        keys = list_keys()
        assert keys[key_id]["prefix"].endswith("...")
        assert keys[key_id]["prefix"].startswith("sk-installm-")

    def test_list_keys_empty(self, tmp_state):
        assert list_keys() == {}

    def test_list_keys_multiple(self, tmp_state):
        create_key(label="key-1")
        create_key(label="key-2")
        keys = list_keys()
        assert len(keys) == 2

    def test_revoke_key(self, tmp_state):
        _, key_id = create_key()
        assert revoke_key(key_id) is True
        assert key_id not in list_keys()

    def test_revoke_nonexistent_key(self, tmp_state):
        assert revoke_key("nonexistent") is False

    def test_has_keys_false_when_empty(self, tmp_state):
        assert has_keys() is False

    def test_has_keys_true_after_create(self, tmp_state):
        create_key()
        assert has_keys() is True

    def test_has_keys_false_after_revoke(self, tmp_state):
        _, key_id = create_key()
        revoke_key(key_id)
        assert has_keys() is False


class TestAuthValidation:
    """Test key validation."""

    def test_validate_valid_key(self, tmp_state):
        full_key, _ = create_key()
        assert validate_key(full_key) is True

    def test_validate_invalid_key(self, tmp_state):
        create_key()
        assert validate_key("sk-installm-wrong") is False

    def test_validate_revoked_key(self, tmp_state):
        full_key, key_id = create_key()
        revoke_key(key_id)
        assert validate_key(full_key) is False

    def test_validate_no_keys_configured(self, tmp_state):
        assert validate_key("sk-installm-anything") is False

    def test_validate_multiple_keys(self, tmp_state):
        key1, _ = create_key()
        key2, _ = create_key()
        assert validate_key(key1) is True
        assert validate_key(key2) is True


class TestAuthMiddleware:
    """Test the auth middleware via the gateway."""

    @pytest.fixture(autouse=True)
    def setup_app(self, tmp_state):
        """Provide a test client."""
        from httpx import AsyncClient, ASGITransport
        from installm.gateway.app import app, _backends, register_backend
        from unittest.mock import AsyncMock, PropertyMock

        backend = AsyncMock()
        type(backend).supports_tools = PropertyMock(return_value=False)
        type(backend).supports_structured_output = PropertyMock(return_value=False)
        backend.generate = AsyncMock(return_value={
            "choices": [{"message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}]
        })

        _backends.clear()
        register_backend("test-model", backend)

        self.transport = ASGITransport(app=app)

    @pytest.mark.asyncio
    async def test_no_auth_required_by_default(self):
        """When auth is not enabled, requests pass through without a key."""
        os.environ.pop("INSTALLM_REQUIRE_AUTH", None)
        from httpx import AsyncClient
        async with AsyncClient(transport=self.transport, base_url="http://test") as client:
            resp = await client.get("/health")
            assert resp.status_code == 200

            resp = await client.post("/v1/chat/completions", json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
            })
            assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_health_always_open(self):
        """Health endpoint is always accessible, even with auth enabled."""
        os.environ["INSTALLM_REQUIRE_AUTH"] = "1"
        try:
            from httpx import AsyncClient
            async with AsyncClient(transport=self.transport, base_url="http://test") as client:
                resp = await client.get("/health")
                assert resp.status_code == 200
        finally:
            os.environ.pop("INSTALLM_REQUIRE_AUTH", None)

    @pytest.mark.asyncio
    async def test_reject_missing_key(self):
        """Requests without a key are rejected when auth is enabled."""
        os.environ["INSTALLM_REQUIRE_AUTH"] = "1"
        try:
            from httpx import AsyncClient
            async with AsyncClient(transport=self.transport, base_url="http://test") as client:
                resp = await client.post("/v1/chat/completions", json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "hi"}],
                })
                assert resp.status_code == 401
                body = resp.json()
                assert body["error"]["code"] == "missing_api_key"
        finally:
            os.environ.pop("INSTALLM_REQUIRE_AUTH", None)

    @pytest.mark.asyncio
    async def test_reject_invalid_key(self):
        """Requests with a wrong key are rejected."""
        os.environ["INSTALLM_REQUIRE_AUTH"] = "1"
        try:
            from httpx import AsyncClient
            async with AsyncClient(transport=self.transport, base_url="http://test") as client:
                resp = await client.post(
                    "/v1/chat/completions",
                    json={"model": "test-model", "messages": [{"role": "user", "content": "hi"}]},
                    headers={"Authorization": "Bearer sk-installm-wrong"},
                )
                assert resp.status_code == 401
                body = resp.json()
                assert body["error"]["code"] == "invalid_api_key"
        finally:
            os.environ.pop("INSTALLM_REQUIRE_AUTH", None)

    @pytest.mark.asyncio
    async def test_accept_valid_key(self):
        """Requests with a valid key pass through."""
        full_key, _ = create_key()
        os.environ["INSTALLM_REQUIRE_AUTH"] = "1"
        try:
            from httpx import AsyncClient
            async with AsyncClient(transport=self.transport, base_url="http://test") as client:
                resp = await client.post(
                    "/v1/chat/completions",
                    json={"model": "test-model", "messages": [{"role": "user", "content": "hi"}]},
                    headers={"Authorization": f"Bearer {full_key}"},
                )
                assert resp.status_code == 200
                body = resp.json()
                assert body["choices"][0]["message"]["content"] == "hello"
        finally:
            os.environ.pop("INSTALLM_REQUIRE_AUTH", None)

    @pytest.mark.asyncio
    async def test_reject_malformed_auth_header(self):
        """Auth header without 'Bearer ' prefix is rejected."""
        os.environ["INSTALLM_REQUIRE_AUTH"] = "1"
        try:
            from httpx import AsyncClient
            async with AsyncClient(transport=self.transport, base_url="http://test") as client:
                resp = await client.post(
                    "/v1/chat/completions",
                    json={"model": "test-model", "messages": [{"role": "user", "content": "hi"}]},
                    headers={"Authorization": "Token some-key"},
                )
                assert resp.status_code == 401
        finally:
            os.environ.pop("INSTALLM_REQUIRE_AUTH", None)

    @pytest.mark.asyncio
    async def test_revoked_key_rejected(self):
        """A revoked key should no longer work."""
        full_key, key_id = create_key()
        os.environ["INSTALLM_REQUIRE_AUTH"] = "1"
        try:
            from httpx import AsyncClient
            async with AsyncClient(transport=self.transport, base_url="http://test") as client:
                # First request should work
                resp = await client.post(
                    "/v1/chat/completions",
                    json={"model": "test-model", "messages": [{"role": "user", "content": "hi"}]},
                    headers={"Authorization": f"Bearer {full_key}"},
                )
                assert resp.status_code == 200

                # Revoke and retry
                revoke_key(key_id)
                resp = await client.post(
                    "/v1/chat/completions",
                    json={"model": "test-model", "messages": [{"role": "user", "content": "hi"}]},
                    headers={"Authorization": f"Bearer {full_key}"},
                )
                assert resp.status_code == 401
        finally:
            os.environ.pop("INSTALLM_REQUIRE_AUTH", None)


class TestAuthCLI:
    """Test the auth CLI commands."""

    def test_auth_create(self, tmp_state, cli_runner):
        result = cli_runner.invoke(["auth", "create"])
        assert result.exit_code == 0
        assert "sk-installm-" in result.output
        assert "Save this key" in result.output

    def test_auth_create_with_label(self, tmp_state, cli_runner):
        result = cli_runner.invoke(["auth", "create", "--label", "my-key"])
        assert result.exit_code == 0
        keys = list_keys()
        assert len(keys) == 1
        assert list(keys.values())[0]["label"] == "my-key"

    def test_auth_ls_empty(self, tmp_state, cli_runner):
        result = cli_runner.invoke(["auth", "ls"])
        assert result.exit_code == 0
        assert "No API keys configured" in result.output

    def test_auth_ls_shows_keys(self, tmp_state, cli_runner):
        create_key(label="test-key")
        result = cli_runner.invoke(["auth", "ls"])
        assert result.exit_code == 0
        assert "test-key" in result.output
        assert "sk-installm-" in result.output

    def test_auth_revoke(self, tmp_state, cli_runner):
        _, key_id = create_key()
        result = cli_runner.invoke(["auth", "revoke", key_id])
        assert result.exit_code == 0
        assert "revoked" in result.output
        assert list_keys() == {}

    def test_auth_revoke_nonexistent(self, tmp_state, cli_runner):
        result = cli_runner.invoke(["auth", "revoke", "nope"])
        assert result.exit_code == 0
        assert "not found" in result.output
