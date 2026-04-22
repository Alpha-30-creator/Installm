"""POST /v1/chat/completions - chat with a model."""

import json
import uuid
import time
from typing import AsyncIterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from installm.gateway.schemas import (
    ChatRequest, ChatResponse, ChatChunk, Choice, ChunkChoice,
    ChoiceDelta, Message, ToolCall, ToolCallFunction, Usage,
    ResponseFormatJSONSchema, ResponseFormatJSON,
)
from installm.gateway.tools import build_tool_prompt, parse_tool_call
from installm.gateway.structured import generate_with_retry
from installm.gateway.app import resolve_model

router = APIRouter()


def _resolve_backend(model_id: str):
    """Look up the backend for a model (supports aliases), raising 404 if not found."""
    backend = resolve_model(model_id)
    if backend is None:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_id}' is not loaded. Run: installm up --model {model_id}",
        )
    return backend


def _messages_to_dicts(messages: list[Message]) -> list[dict]:
    """Convert Pydantic Message objects to plain dicts for backends."""
    result = []
    for m in messages:
        d = {"role": m.role, "content": m.content or ""}
        if m.tool_calls:
            d["tool_calls"] = [tc.model_dump() for tc in m.tool_calls]
        if m.tool_call_id:
            d["tool_call_id"] = m.tool_call_id
        if m.name:
            d["name"] = m.name
        result.append(d)
    return result


def _build_gen_kwargs(req: ChatRequest) -> dict:
    """Extract generation parameters from the request."""
    return {
        k: v for k, v in {
            "temperature": req.temperature,
            "top_p": req.top_p,
            "max_tokens": req.max_tokens,
        }.items() if v is not None
    }


async def _handle_tools_fallback(
    backend, messages: list[dict], req: ChatRequest, gen_kwargs: dict
) -> Message:
    """Inject tool prompt and parse tool call from response for backends
    that do not natively support tool calling."""
    tool_system = build_tool_prompt(req.tools)
    augmented = [{"role": "system", "content": tool_system}] + messages

    result = await backend.generate(augmented, **gen_kwargs)
    raw = result["choices"][0]["message"]["content"] or ""

    tool_call = parse_tool_call(raw)
    if tool_call:
        return Message(role="assistant", content=None, tool_calls=[tool_call])
    return Message(role="assistant", content=raw)


async def _handle_structured_output(
    backend, messages: list[dict], req: ChatRequest, gen_kwargs: dict
) -> tuple[str, Message]:
    """Handle json_object and json_schema response formats."""
    schema = None
    if isinstance(req.response_format, ResponseFormatJSONSchema):
        schema = req.response_format.json_schema.schema_

    if backend.supports_structured_output:
        # Pass response_format directly to the backend
        result = await backend.generate(
            messages,
            response_format=req.response_format.model_dump(),
            **gen_kwargs,
        )
        raw = result["choices"][0]["message"]["content"] or ""
        return raw, Message(role="assistant", content=raw)

    # Fallback: validate-and-retry loop
    raw, parsed = await generate_with_retry(
        backend.generate, messages, schema, **gen_kwargs
    )
    return raw, Message(role="assistant", content=json.dumps(parsed))


@router.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    """Handle chat completion requests (streaming and non-streaming)."""
    backend = _resolve_backend(req.model)
    messages = _messages_to_dicts(req.messages)
    gen_kwargs = _build_gen_kwargs(req)

    # --- Streaming path ---
    if req.stream:
        return StreamingResponse(
            _stream_response(backend, req, messages, gen_kwargs),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # --- Non-streaming path ---

    # Structured output
    if req.response_format and not isinstance(req.response_format, type(None)):
        if isinstance(req.response_format, (ResponseFormatJSON, ResponseFormatJSONSchema)):
            raw, msg = await _handle_structured_output(backend, messages, req, gen_kwargs)
            return _build_response(req.model, msg)

    # Tool calling
    if req.tools:
        if backend.supports_tools:
            # Native tool calling: pass tools directly to backend
            result = await backend.generate(
                messages,
                tools=[t.model_dump() for t in req.tools],
                tool_choice=req.tool_choice if isinstance(req.tool_choice, str)
                            else req.tool_choice.model_dump() if req.tool_choice else None,
                **gen_kwargs,
            )
            return _adapt_backend_response(result, req.model)
        else:
            msg = await _handle_tools_fallback(backend, messages, req, gen_kwargs)
            finish = "tool_calls" if msg.tool_calls else "stop"
            return _build_response(req.model, msg, finish_reason=finish)

    # Plain generation
    result = await backend.generate(messages, **gen_kwargs)
    return _adapt_backend_response(result, req.model)


async def _stream_response(
    backend, req: ChatRequest, messages: list[dict], gen_kwargs: dict
) -> AsyncIterator[str]:
    """Async generator that yields SSE lines for a streaming chat response."""
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())

    # First chunk: role announcement
    first = ChatChunk(
        id=chunk_id,
        created=created,
        model=req.model,
        choices=[ChunkChoice(
            index=0,
            delta=ChoiceDelta(role="assistant"),
            finish_reason=None,
        )],
    )
    yield f"data: {first.model_dump_json()}\n\n"

    # Content chunks
    async for chunk in backend.stream(messages, **gen_kwargs):
        # Normalise backend output to our schema
        choices = chunk.get("choices", [])
        if choices:
            delta = choices[0].get("delta", {})
            finish = choices[0].get("finish_reason")
            out = ChatChunk(
                id=chunk_id,
                created=created,
                model=req.model,
                choices=[ChunkChoice(
                    index=0,
                    delta=ChoiceDelta(
                        content=delta.get("content"),
                        tool_calls=delta.get("tool_calls"),
                    ),
                    finish_reason=finish,
                )],
            )
            yield f"data: {out.model_dump_json(exclude_none=True)}\n\n"

    yield "data: [DONE]\n\n"


def _build_response(model: str, msg: Message, finish_reason: str = "stop") -> ChatResponse:
    """Wrap a Message in a ChatResponse."""
    return ChatResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
        model=model,
        choices=[Choice(index=0, message=msg, finish_reason=finish_reason)],
    )


def _adapt_backend_response(result: dict, model: str) -> ChatResponse:
    """Convert a raw backend response dict to a ChatResponse."""
    choices = result.get("choices", [])
    if not choices:
        raise HTTPException(status_code=502, detail="Backend returned no choices.")

    raw_msg = choices[0].get("message", {})
    tool_calls = None
    if raw_msg.get("tool_calls"):
        tool_calls = [
            ToolCall(
                id=tc.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                function=ToolCallFunction(
                    name=tc["function"]["name"],
                    arguments=tc["function"]["arguments"],
                ),
            )
            for tc in raw_msg["tool_calls"]
        ]

    msg = Message(
        role=raw_msg.get("role", "assistant"),
        content=raw_msg.get("content"),
        tool_calls=tool_calls,
    )

    usage = result.get("usage")
    usage_obj = None
    if usage:
        usage_obj = Usage(
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
        )

    return ChatResponse(
        id=result.get("id", f"chatcmpl-{uuid.uuid4().hex[:8]}"),
        model=model,
        choices=[Choice(
            index=0,
            message=msg,
            finish_reason=choices[0].get("finish_reason", "stop"),
        )],
        usage=usage_obj,
    )
