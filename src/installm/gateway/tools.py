"""Tool calling helpers: prompt injection and response parsing fallback."""

import json
import uuid
from typing import Optional

from installm.gateway.schemas import ToolCall, ToolCallFunction, ToolDefinition


def build_tool_prompt(tools: list[ToolDefinition]) -> str:
    """Build a system prompt fragment that instructs the model to use tools.

    Used as a fallback when the backend does not natively support tool calling.
    """
    tool_specs = [
        {
            "name": t.function.name,
            "description": t.function.description or "",
            "parameters": t.function.parameters or {},
        }
        for t in tools
    ]
    return (
        "You have access to the following tools. When you need to call a tool, "
        "respond ONLY with a JSON object in this exact format and nothing else:\n"
        '{"tool_call": {"name": "<tool_name>", "arguments": {<args>}}}\n\n'
        f"Available tools:\n{json.dumps(tool_specs, indent=2)}"
    )


def _extract_balanced_json(text: str, start: int) -> Optional[str]:
    """Extract a balanced JSON object starting at `start` index."""
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def parse_tool_call(text: str, timeout_chars: int = 4096) -> Optional[ToolCall]:
    """Attempt to extract a tool call from model output text.

    Returns a ToolCall if found, or None if the output is plain text.
    Applies a character limit to prevent processing excessively long strings.
    """
    if len(text) > timeout_chars:
        text = text[:timeout_chars]

    # Find the start of a {"tool_call": ...} object
    start = text.find('{"tool_call"')
    if start == -1:
        return None

    json_str = _extract_balanced_json(text, start)
    if json_str is None:
        return None

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return None

    tc = data.get("tool_call", {})
    name = tc.get("name")
    args = tc.get("arguments", {})

    if not name:
        return None

    return ToolCall(
        id=f"call_{uuid.uuid4().hex[:8]}",
        type="function",
        function=ToolCallFunction(
            name=name,
            arguments=json.dumps(args),
        ),
    )
