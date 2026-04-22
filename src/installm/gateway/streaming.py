"""SSE streaming helpers for InstaLLM gateway."""

import json
from typing import AsyncIterator


async def sse_stream(chunks: AsyncIterator[dict]):
    """Yield SSE-formatted lines from an async iterator of chunk dicts.

    Emits:  data: <json>\\n\\n  for each chunk, then  data: [DONE]\\n\\n
    """
    async for chunk in chunks:
        yield f"data: {json.dumps(chunk)}\n\n"
    yield "data: [DONE]\n\n"
