"""Structured output helpers: JSON enforcement and validate-and-retry loop."""

import json
import re
from typing import Any, Optional


def build_json_prompt(schema: Optional[dict] = None) -> str:
    """Build a system prompt fragment that instructs the model to output JSON.

    If a schema is provided, it is included so the model can target it.
    """
    if schema:
        return (
            "You must respond with ONLY valid JSON that matches this schema. "
            "No markdown, no explanation, no code fences. Just the raw JSON object.\n"
            f"Schema: {json.dumps(schema)}"
        )
    return (
        "You must respond with ONLY valid JSON. "
        "No markdown, no explanation, no code fences. Just the raw JSON object."
    )


def extract_json(text: str) -> Optional[Any]:
    """Extract JSON from model output, stripping markdown code fences if present."""
    # Strip ```json ... ``` or ``` ... ``` wrappers
    text = text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find a JSON object or array within the text
        for pattern in (r"\{.*\}", r"\[.*\]"):
            match = re.search(pattern, text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
    return None


def validate_against_schema(data: Any, schema: dict) -> list[str]:
    """Validate data against a JSON schema. Returns a list of error messages."""
    try:
        import jsonschema
        validator = jsonschema.Draft7Validator(schema)
        errors = [e.message for e in validator.iter_errors(data)]
        return errors
    except ImportError:
        # jsonschema not installed; skip validation
        return []


async def generate_with_retry(
    generate_fn,
    messages: list,
    schema: Optional[dict],
    max_retries: int = 3,
    **kwargs,
) -> tuple[str, Any]:
    """Run generate_fn with JSON instruction, retrying on parse/validation failure.

    Returns (raw_text, parsed_json). Raises ValueError after max_retries.
    """
    json_instruction = build_json_prompt(schema)

    # Inject JSON instruction as a system message
    augmented = [{"role": "system", "content": json_instruction}] + list(messages)

    last_error = None
    for attempt in range(max_retries):
        if attempt > 0 and last_error:
            # Tell the model what went wrong so it can self-correct
            augmented = augmented + [{
                "role": "user",
                "content": f"Your previous response was invalid. Error: {last_error}. "
                           "Please respond with valid JSON only.",
            }]

        result = await generate_fn(augmented, **kwargs)
        raw_text = result["choices"][0]["message"]["content"] or ""

        parsed = extract_json(raw_text)
        if parsed is None:
            last_error = "Response was not valid JSON."
            continue

        if schema:
            errors = validate_against_schema(parsed, schema)
            if errors:
                last_error = "; ".join(errors[:3])
                continue

        return raw_text, parsed

    raise ValueError(
        f"Model failed to produce valid JSON after {max_retries} attempts. "
        f"Last error: {last_error}"
    )
