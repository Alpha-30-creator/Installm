"""Tests for tool calling and structured output helper modules."""

import json
import pytest
from unittest.mock import AsyncMock

from installm.gateway.tools import build_tool_prompt, parse_tool_call
from installm.gateway.structured import (
    build_json_prompt, extract_json, validate_against_schema, generate_with_retry,
)
from installm.gateway.schemas import ToolDefinition, FunctionDefinition


# ---------------------------------------------------------------------------
# Tool calling helpers
# ---------------------------------------------------------------------------

def _make_tool(name="get_weather", description="Get weather"):
    return ToolDefinition(
        type="function",
        function=FunctionDefinition(
            name=name,
            description=description,
            parameters={"type": "object", "properties": {"city": {"type": "string"}}},
        ),
    )


def test_build_tool_prompt_contains_tool_name():
    """Tool prompt should include the tool name and description."""
    tools = [_make_tool()]
    prompt = build_tool_prompt(tools)
    assert "get_weather" in prompt
    assert "Get weather" in prompt
    assert "tool_call" in prompt


def test_parse_tool_call_valid():
    """parse_tool_call should extract a valid tool call from model output."""
    text = '{"tool_call": {"name": "get_weather", "arguments": {"city": "HK"}}}'
    result = parse_tool_call(text)
    assert result is not None
    assert result.function.name == "get_weather"
    args = json.loads(result.function.arguments)
    assert args["city"] == "HK"


def test_parse_tool_call_plain_text():
    """parse_tool_call should return None for plain text responses."""
    result = parse_tool_call("The weather in Hong Kong is sunny today.")
    assert result is None


def test_parse_tool_call_truncation():
    """parse_tool_call should handle very long text without hanging."""
    long_text = "x" * 10000 + '{"tool_call": {"name": "fn", "arguments": {}}}'
    # Should not hang; may or may not find the tool call depending on truncation
    result = parse_tool_call(long_text)
    # Just verify it returns without error
    assert result is None or result.function.name == "fn"


def test_parse_tool_call_malformed_json():
    """parse_tool_call should return None for malformed JSON."""
    result = parse_tool_call('{"tool_call": {"name": "fn", "arguments": {broken}}}')
    assert result is None


# ---------------------------------------------------------------------------
# Structured output helpers
# ---------------------------------------------------------------------------

def test_build_json_prompt_no_schema():
    """JSON prompt without schema should instruct plain JSON output."""
    prompt = build_json_prompt()
    assert "JSON" in prompt
    assert "markdown" in prompt.lower() or "code" in prompt.lower()


def test_build_json_prompt_with_schema():
    """JSON prompt with schema should include the schema."""
    schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    prompt = build_json_prompt(schema)
    assert "name" in prompt
    assert "Schema" in prompt


def test_extract_json_plain():
    """extract_json should parse a plain JSON string."""
    result = extract_json('{"name": "Alice"}')
    assert result == {"name": "Alice"}


def test_extract_json_with_fence():
    """extract_json should strip markdown code fences."""
    result = extract_json('```json\n{"name": "Alice"}\n```')
    assert result == {"name": "Alice"}


def test_extract_json_embedded():
    """extract_json should find JSON embedded in prose."""
    result = extract_json('Here is the result: {"name": "Bob"} done.')
    assert result == {"name": "Bob"}


def test_extract_json_invalid():
    """extract_json should return None for non-JSON text."""
    result = extract_json("This is just plain text with no JSON.")
    assert result is None


def test_validate_against_schema_valid():
    """validate_against_schema should return no errors for valid data."""
    schema = {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}
    errors = validate_against_schema({"name": "Alice"}, schema)
    assert errors == []


def test_validate_against_schema_invalid():
    """validate_against_schema should return errors for invalid data."""
    schema = {"type": "object", "properties": {"age": {"type": "integer"}}, "required": ["age"]}
    errors = validate_against_schema({"age": "not-a-number"}, schema)
    assert len(errors) > 0


@pytest.mark.asyncio
async def test_generate_with_retry_success_first_try():
    """generate_with_retry should succeed on first attempt with valid JSON."""
    async def mock_generate(messages, **kwargs):
        return {
            "choices": [{"message": {"content": '{"result": 42}'}}]
        }

    raw, parsed = await generate_with_retry(mock_generate, [], schema=None)
    assert parsed == {"result": 42}


@pytest.mark.asyncio
async def test_generate_with_retry_retries_on_invalid():
    """generate_with_retry should retry when JSON is invalid."""
    call_count = 0

    async def mock_generate(messages, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            return {"choices": [{"message": {"content": "not json"}}]}
        return {"choices": [{"message": {"content": '{"ok": true}'}}]}

    raw, parsed = await generate_with_retry(mock_generate, [], schema=None, max_retries=3)
    assert parsed == {"ok": True}
    assert call_count == 2


@pytest.mark.asyncio
async def test_generate_with_retry_exhausted():
    """generate_with_retry should raise ValueError after max retries."""
    async def mock_generate(messages, **kwargs):
        return {"choices": [{"message": {"content": "still not json"}}]}

    with pytest.raises(ValueError, match="failed to produce valid JSON"):
        await generate_with_retry(mock_generate, [], schema=None, max_retries=2)


@pytest.mark.asyncio
async def test_generate_with_retry_schema_validation():
    """generate_with_retry should retry when schema validation fails."""
    call_count = 0
    schema = {"type": "object", "properties": {"age": {"type": "integer"}}, "required": ["age"]}

    async def mock_generate(messages, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            return {"choices": [{"message": {"content": '{"age": "wrong-type"}'}}]}
        return {"choices": [{"message": {"content": '{"age": 25}'}}]}

    raw, parsed = await generate_with_retry(mock_generate, [], schema=schema, max_retries=3)
    assert parsed["age"] == 25
    assert call_count == 2
