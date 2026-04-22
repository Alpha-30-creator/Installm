"""POST /v1/responses - Responses API with semantic streaming events."""

import json
import time
import uuid
from typing import AsyncIterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from installm.gateway.schemas import ResponsesRequest
from installm.gateway.app import get_backends

router = APIRouter()


def _build_messages(req: ResponsesRequest) -> list[dict]:
    """Convert Responses API input to chat-style messages list."""
    messages = []

    if req.instructions:
        messages.append({"role": "system", "content": req.instructions})

    if isinstance(req.input, str):
        messages.append({"role": "user", "content": req.input})
    elif isinstance(req.input, list):
        for item in req.input:
            if isinstance(item, str):
                messages.append({"role": "user", "content": item})
            elif isinstance(item, dict):
                messages.append(item)

    return messages


def _gen_kwargs(req: ResponsesRequest) -> dict:
    return {k: v for k, v in {
        "temperature": req.temperature,
        "top_p": req.top_p,
        "max_tokens": req.max_output_tokens,
    }.items() if v is not None}


@router.post("/v1/responses")
async def create_response(req: ResponsesRequest):
    """Handle Responses API requests with semantic streaming events."""
    backends = get_backends()
    backend = backends.get(req.model)
    if backend is None:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{req.model}' is not loaded. Run: installm up --model {req.model}",
        )

    messages = _build_messages(req)
    gen_kwargs = _gen_kwargs(req)

    if req.stream:
        return StreamingResponse(
            _stream_events(backend, req, messages, gen_kwargs),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # Non-streaming: collect all events and return final response object
    response_id = f"resp_{uuid.uuid4().hex[:12]}"
    result = await backend.generate(messages, **gen_kwargs)
    raw_msg = result["choices"][0]["message"]
    content = raw_msg.get("content") or ""

    return {
        "id": response_id,
        "object": "response",
        "created_at": int(time.time()),
        "model": req.model,
        "status": "completed",
        "output": [
            {
                "type": "message",
                "id": f"msg_{uuid.uuid4().hex[:8]}",
                "role": "assistant",
                "content": [{"type": "output_text", "text": content}],
            }
        ],
        "usage": result.get("usage"),
    }


async def _stream_events(
    backend, req: ResponsesRequest, messages: list[dict], gen_kwargs: dict
) -> AsyncIterator[str]:
    """Yield semantic SSE events following the Open Responses spec."""
    response_id = f"resp_{uuid.uuid4().hex[:12]}"
    item_id = f"msg_{uuid.uuid4().hex[:8]}"
    seq = 0

    def _event(event_type: str, data: dict) -> str:
        nonlocal seq
        data["sequence_number"] = seq
        seq += 1
        return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"

    # response.created
    yield _event("response.created", {
        "response": {
            "id": response_id,
            "object": "response",
            "created_at": int(time.time()),
            "model": req.model,
            "status": "in_progress",
        }
    })

    # response.in_progress
    yield _event("response.in_progress", {"response_id": response_id})

    # response.output_item.added
    yield _event("response.output_item.added", {
        "response_id": response_id,
        "item": {
            "id": item_id,
            "type": "message",
            "role": "assistant",
            "status": "in_progress",
        },
    })

    # Stream content deltas
    full_text = ""
    async for chunk in backend.stream(messages, **gen_kwargs):
        choices = chunk.get("choices", [])
        if not choices:
            continue
        delta = choices[0].get("delta", {})
        content = delta.get("content", "")
        tool_calls = delta.get("tool_calls")

        if content:
            full_text += content
            yield _event("response.output_text.delta", {
                "response_id": response_id,
                "item_id": item_id,
                "delta": content,
            })

        if tool_calls:
            for tc_delta in tool_calls:
                yield _event("response.function_call_arguments.delta", {
                    "response_id": response_id,
                    "item_id": item_id,
                    "delta": tc_delta.get("function", {}).get("arguments", ""),
                })

    # response.output_text.done
    yield _event("response.output_text.done", {
        "response_id": response_id,
        "item_id": item_id,
        "text": full_text,
    })

    # response.output_item.done
    yield _event("response.output_item.done", {
        "response_id": response_id,
        "item": {
            "id": item_id,
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": full_text}],
        },
    })

    # response.completed
    yield _event("response.completed", {
        "response": {
            "id": response_id,
            "object": "response",
            "model": req.model,
            "status": "completed",
        }
    })

    # Terminal marker (some clients expect this)
    yield "data: [DONE]\n\n"
