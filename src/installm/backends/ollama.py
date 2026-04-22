"""Ollama backend - proxies requests to a local Ollama server."""

import asyncio
import shutil
import json
from typing import AsyncIterator

import httpx

from installm.backends.base import Backend


class OllamaBackend(Backend):
    """Backend that delegates inference to a running Ollama instance."""

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.model_name = None
        self._client = httpx.AsyncClient(base_url=base_url, timeout=300)

    async def load(self, model_id: str, **kwargs):
        """Pull the model into Ollama and verify it's available."""
        if not shutil.which("ollama"):
            raise RuntimeError("Ollama is not installed. Get it from https://ollama.com")

        self.model_name = model_id
        # Pull model (Ollama handles caching)
        proc = await asyncio.create_subprocess_exec(
            "ollama", "pull", model_id,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"ollama pull failed: {stderr.decode()}")

    async def generate(self, messages: list, **kwargs) -> dict:
        """Non-streaming chat completion via Ollama's OpenAI-compat endpoint."""
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            **{k: v for k, v in kwargs.items() if v is not None},
        }
        resp = await self._client.post("/v1/chat/completions", json=payload)
        resp.raise_for_status()
        return resp.json()

    async def stream(self, messages: list, **kwargs) -> AsyncIterator[dict]:
        """Streaming chat completion via Ollama's OpenAI-compat endpoint."""
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": True,
            **{k: v for k, v in kwargs.items() if v is not None},
        }
        async with self._client.stream(
            "POST", "/v1/chat/completions", json=payload
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data.strip() == "[DONE]":
                    break
                yield json.loads(data)

    async def embed(self, texts: list[str], **kwargs) -> list[list[float]]:
        """Generate embeddings via Ollama's embeddings endpoint."""
        results = []
        for text in texts:
            resp = await self._client.post(
                "/v1/embeddings",
                json={"model": self.model_name, "input": text},
            )
            resp.raise_for_status()
            data = resp.json()
            results.append(data["data"][0]["embedding"])
        return results

    @property
    def supports_tools(self) -> bool:
        return True  # Ollama supports tool calling

    @property
    def supports_structured_output(self) -> bool:
        return True  # Ollama supports JSON mode

    async def unload(self):
        """Close the HTTP client."""
        await self._client.aclose()
