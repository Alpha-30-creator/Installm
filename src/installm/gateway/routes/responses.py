"""POST /v1/responses - Responses API with semantic streaming events."""

import json
import time
import uuid
from typing import AsyncIterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from installm.gateway.schemas import ResponsesRequest
from installm.gateway.tools import build_tool_prompt, parse_tool_call
from installm.gateway.app import resolve_model

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
    backend = resolve_model(req.model)
    if backend is None:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{req.model}' is not loaded. Run: installm up --model {req.model}",
        )

    messages = _build_messages(req)
    gen_kwargs = _gen_kwargs(req)

    # Inject tool prompt for backends without native tool support
    if req.tools and not backend.supports_tools:
        tool_system = build_tool_prompt(req.tools)
        messages = [{"role": "system", "content": tool_system}] + messages

    if req.stream:
        return StreamingResponse(
            _stream_events(backend, req, messages, gen_kwargs),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # Non-streaming: generate and build response object
    response_id = f"resp_{uuid.uuid4().hex[:12]}"

    if req.tools and backend.supports_tools:
        result = await backend.generate(
            messages,
            tools=[t.model_dump() for t in req.tools],
            tool_choice=req.tool_choice if isinstance(req.tool_choice, str)
                        else req.tool_choice.model_dump() if req.tool_choice else None,
            **gen_kwargs,
        )
    else:
        result = await backend.generate(messages, **gen_kwargs)

    raw_msg = result["choices"][0]["message"]
    content = raw_msg.get("content") or ""
    output_items = []

    # Check for native tool calls from backend
    tool_calls = raw_msg.get("tool_calls")
    if tool_calls:
        for tc in tool_calls:
            output_items.append({
                "type": "function_call",
                "id": tc.get("id", f"fc_{uuid.uuid4().hex[:8]}"),
                "call_id": tc.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                "name": tc["function"]["name"],
                "arguments": tc["function"]["arguments"],
                "status": "completed",
            })
    elif req.tools and not backend.supports_tools:
        # Try to parse tool call from content (fallback)
        parsed = parse_tool_call(content)
        if parsed:
            output_items.append({
                "type": "function_call",
                "id": f"fc_{uuid.uuid4().hex[:8]}",
                "call_id": parsed.id,
                "name": parsed.function.name,
                "arguments": parsed.function.arguments,
                "status": "completed",
            })
            content = ""  # Clear content since it was a tool call

    # Add message item if there is text content
    if content or not output_items:
        output_items.insert(0, {
            "type": "message",
            "id": f"msg_{uuid.uuid4().hex[:8]}",
            "role": "assistant",
            "content": [{"type": "output_text", "text": content}],
        })

    return {
        "id": response_id,
        "object": "response",
        "created_at": int(time.time()),
        "model": req.model,
        "status": "completed",
        "output": output_items,
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
    tool_args_buffer = ""
    has_tool_calls = False

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
            has_tool_calls = True
            for tc_delta in tool_calls:
                args_chunk = tc_delta.get("function", {}).get("arguments", "")
                tool_args_buffer += args_chunk
                yield _event("response.function_call_arguments.delta", {
                    "response_id": response_id,
                    "item_id": item_id,
                    "delta": args_chunk,
                })

    # Emit done events for tool calls if any
    if has_tool_calls:
        yield _event("response.function_call_arguments.done", {
            "response_id": response_id,
            "item_id": item_id,
            "arguments": tool_args_buffer,
        })

    # response.output_text.done
    if full_text:
        yield _event("response.output_text.done", {
            "response_id": response_id,
            "item_id": item_id,
            "text": full_text,
        })

    # response.output_item.done
    output_content = [{"type": "output_text", "text": full_text}] if full_text else []
    yield _event("response.output_item.done", {
        "response_id": response_id,
        "item": {
            "id": item_id,
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": output_content,
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

    # Terminal marker
    yield "data: [DONE]\n\n"
