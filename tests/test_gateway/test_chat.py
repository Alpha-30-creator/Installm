"""Tests for POST /v1/chat/completions endpoint."""

import json
import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import AsyncMock, MagicMock, patch

from installm.gateway.app import app, _backends


def _make_backend(supports_tools=False, supports_structured=False):
    """Create a mock backend with configurable capabilities."""
    backend = MagicMock()
    type(backend).supports_tools = property(lambda self: supports_tools)
    type(backend).supports_structured_output = property(lambda self: supports_structured)
    backend.generate = AsyncMock(return_value={
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "test-model",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "Hello, world!"},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
    })

    async def _stream(*args, **kwargs):
        yield {
            "choices": [{
                "delta": {"content": "Hello"},
                "finish_reason": None,
            }]
        }
        yield {
            "choices": [{
                "delta": {},
                "finish_reason": "stop",
            }]
        }

    backend.stream = _stream
    return backend


@pytest.fixture(autouse=True)
def clean_backends():
    """Ensure backend registry is clean before each test."""
    _backends.clear()
    yield
    _backends.clear()


@pytest.mark.asyncio
async def test_chat_model_not_found():
    """Should return 404 when model is not loaded."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/v1/chat/completions", json={
            "model": "nonexistent",
            "messages": [{"role": "user", "content": "Hi"}],
        })
    assert resp.status_code == 404
    assert "not loaded" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_chat_basic():
    """Basic non-streaming chat should return OpenAI-shaped response."""
    _backends["gpt2"] = _make_backend()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/v1/chat/completions", json={
            "model": "gpt2",
            "messages": [{"role": "user", "content": "Hello"}],
        })

    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["choices"][0]["message"]["content"] == "Hello, world!"


@pytest.mark.asyncio
async def test_chat_streaming():
    """Streaming chat should return SSE with chat.completion.chunk objects."""
    _backends["gpt2"] = _make_backend()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        async with client.stream("POST", "/v1/chat/completions", json={
            "model": "gpt2",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        }) as resp:
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers["content-type"]
            lines = []
            async for line in resp.aiter_lines():
                lines.append(line)

    data_lines = [l for l in lines if l.startswith("data: ") and l != "data: [DONE]"]
    assert len(data_lines) >= 1

    # First chunk should have role: assistant
    first = json.loads(data_lines[0][6:])
    assert first["object"] == "chat.completion.chunk"
    assert first["choices"][0]["delta"].get("role") == "assistant"

    # Should end with [DONE]
    assert "data: [DONE]" in lines


@pytest.mark.asyncio
async def test_chat_with_tools_native():
    """When backend supports tools natively, tools should be passed through."""
    backend = _make_backend(supports_tools=True)
    backend.generate = AsyncMock(return_value={
        "id": "chatcmpl-tool",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "gpt2",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_abc",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "HK"}'},
                }],
            },
            "finish_reason": "tool_calls",
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    })
    _backends["gpt2"] = backend

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/v1/chat/completions", json={
            "model": "gpt2",
            "messages": [{"role": "user", "content": "What's the weather in HK?"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                },
            }],
        })

    assert resp.status_code == 200
    body = resp.json()
    assert body["choices"][0]["message"]["tool_calls"] is not None
    assert body["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "get_weather"


@pytest.mark.asyncio
async def test_chat_with_tools_fallback():
    """When backend doesn't support tools, fallback prompt-and-parse should be used."""
    backend = _make_backend(supports_tools=False)
    # Simulate model returning a tool call JSON
    backend.generate = AsyncMock(return_value={
        "id": "chatcmpl-fallback",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "gpt2",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": '{"tool_call": {"name": "get_weather", "arguments": {"city": "HK"}}}',
            },
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    })
    _backends["gpt2"] = backend

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/v1/chat/completions", json={
            "model": "gpt2",
            "messages": [{"role": "user", "content": "What's the weather?"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                },
            }],
        })

    assert resp.status_code == 200
    body = resp.json()
    # Should have parsed the tool call from the text
    assert body["choices"][0]["message"]["tool_calls"] is not None
    assert body["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "get_weather"


@pytest.mark.asyncio
async def test_chat_json_object_format():
    """response_format json_object should return valid JSON content."""
    backend = _make_backend(supports_structured=False)
    backend.generate = AsyncMock(return_value={
        "id": "chatcmpl-json",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "gpt2",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": '{"name": "Alice", "age": 30}'},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
    })
    _backends["gpt2"] = backend

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/v1/chat/completions", json={
            "model": "gpt2",
            "messages": [{"role": "user", "content": "Give me a person object"}],
            "response_format": {"type": "json_object"},
        })

    assert resp.status_code == 200
    body = resp.json()
    content = body["choices"][0]["message"]["content"]
    parsed = json.loads(content)
    assert "name" in parsed
