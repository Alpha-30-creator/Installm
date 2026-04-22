"""Tests for POST /v1/responses endpoint."""

import json
import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import AsyncMock, MagicMock

from installm.gateway.app import app, _backends


def _make_backend():
    backend = MagicMock()
    backend.generate = AsyncMock(return_value={
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "test-model",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "The answer is 42."},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
    })

    async def _stream(*args, **kwargs):
        yield {"choices": [{"delta": {"content": "The "}, "finish_reason": None}]}
        yield {"choices": [{"delta": {"content": "answer"}, "finish_reason": None}]}
        yield {"choices": [{"delta": {}, "finish_reason": "stop"}]}

    backend.stream = _stream
    return backend


@pytest.fixture(autouse=True)
def clean_backends():
    _backends.clear()
    yield
    _backends.clear()


@pytest.mark.asyncio
async def test_responses_model_not_found():
    """Should return 404 when model is not loaded."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/v1/responses", json={
            "model": "nonexistent",
            "input": "Hello",
        })
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_responses_non_streaming():
    """Non-streaming Responses API should return a response object."""
    _backends["gpt2"] = _make_backend()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/v1/responses", json={
            "model": "gpt2",
            "input": "What is the answer?",
        })

    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "response"
    assert body["status"] == "completed"
    assert len(body["output"]) == 1
    assert body["output"][0]["type"] == "message"
    assert body["output"][0]["role"] == "assistant"
    content = body["output"][0]["content"][0]["text"]
    assert "42" in content


@pytest.mark.asyncio
async def test_responses_with_instructions():
    """instructions field should be passed as a system message."""
    backend = _make_backend()
    _backends["gpt2"] = backend

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post("/v1/responses", json={
            "model": "gpt2",
            "input": "Hello",
            "instructions": "You are a helpful assistant.",
        })

    # Verify the system message was included in the generate call
    call_args = backend.generate.call_args
    messages = call_args[0][0]
    assert any(m["role"] == "system" for m in messages)
    system_msg = next(m for m in messages if m["role"] == "system")
    assert "helpful assistant" in system_msg["content"]


@pytest.mark.asyncio
async def test_responses_streaming_event_order():
    """Streaming Responses API should emit events in the correct order."""
    _backends["gpt2"] = _make_backend()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        async with client.stream("POST", "/v1/responses", json={
            "model": "gpt2",
            "input": "Hello",
            "stream": True,
        }) as resp:
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers["content-type"]

            events = []
            current_event = None
            async for line in resp.aiter_lines():
                if line.startswith("event: "):
                    current_event = line[7:]
                elif line.startswith("data: ") and line != "data: [DONE]":
                    events.append(current_event)

    # Verify required event types are present and in order
    assert events[0] == "response.created"
    assert events[1] == "response.in_progress"
    assert "response.output_item.added" in events
    assert "response.output_text.delta" in events
    assert "response.output_text.done" in events
    assert "response.output_item.done" in events
    assert events[-1] == "response.completed"


@pytest.mark.asyncio
async def test_responses_streaming_sequence_numbers():
    """Each streaming event should have an incrementing sequence_number."""
    _backends["gpt2"] = _make_backend()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        async with client.stream("POST", "/v1/responses", json={
            "model": "gpt2",
            "input": "Hello",
            "stream": True,
        }) as resp:
            seq_numbers = []
            async for line in resp.aiter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    data = json.loads(line[6:])
                    if "sequence_number" in data:
                        seq_numbers.append(data["sequence_number"])

    assert seq_numbers == list(range(len(seq_numbers)))
