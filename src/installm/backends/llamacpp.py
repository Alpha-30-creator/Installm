"""llama.cpp backend via llama-cpp-python.

Uses GGUF quantised models for efficient CPU/GPU inference.
Replaces Ollama as the lightweight backend — no external service needed.
"""

import asyncio
from typing import AsyncIterator

from installm.backends.base import Backend


class LlamaCppBackend(Backend):
    """Backend powered by llama-cpp-python (llama.cpp bindings)."""

    name = "llamacpp"
    supports_tools = False
    supports_structured_output = False

    def __init__(self):
        self.model = None
        self.model_id = None

    async def load(self, model_id: str, **kwargs) -> None:
        """Load a GGUF model file.

        Args:
            model_id: Path to a .gguf file, or a HuggingFace repo ID.
                      If a repo ID is given, we attempt to find a .gguf file
                      in the HF cache.
            **kwargs: Passed to llama_cpp.Llama (n_ctx, n_gpu_layers, etc.)
        """
        from llama_cpp import Llama

        gguf_path = self._resolve_model_path(model_id)
        n_ctx = kwargs.pop("n_ctx", 2048)
        n_gpu_layers = kwargs.pop("n_gpu_layers", 0)

        # Load in a thread to avoid blocking the event loop
        self.model = await asyncio.to_thread(
            Llama,
            model_path=gguf_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
            **kwargs,
        )
        self.model_id = model_id

    async def unload(self) -> None:
        """Release model resources."""
        if self.model is not None:
            del self.model
            self.model = None
        self.model_id = None

    async def generate(self, messages: list, **kwargs) -> dict:
        """Generate a chat completion (non-streaming)."""
        max_tokens = kwargs.get("max_tokens", 256)
        temperature = kwargs.get("temperature", 0.7)
        top_p = kwargs.get("top_p", 1.0)

        result = await asyncio.to_thread(
            self.model.create_chat_completion,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=False,
        )
        return result

    async def stream(self, messages: list, **kwargs) -> AsyncIterator[dict]:
        """Stream chat completion chunks."""
        max_tokens = kwargs.get("max_tokens", 256)
        temperature = kwargs.get("temperature", 0.7)
        top_p = kwargs.get("top_p", 1.0)

        # llama-cpp-python returns a generator when stream=True
        gen = await asyncio.to_thread(
            self.model.create_chat_completion,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=True,
        )

        # Iterate the synchronous generator in a thread-safe way
        def _next(it):
            try:
                return next(it)
            except StopIteration:
                return None

        while True:
            chunk = await asyncio.to_thread(_next, gen)
            if chunk is None:
                break
            yield chunk

    async def embed(self, texts: list[str], **kwargs) -> list[list[float]]:
        """Generate embeddings. Not supported by default llama.cpp setup."""
        # llama-cpp-python supports embeddings if the model was loaded with
        # embedding=True, but most chat models don't support it well.
        results = []
        for text in texts:
            emb = await asyncio.to_thread(self.model.embed, text)
            results.append(emb)
        return results

    def _resolve_model_path(self, model_id: str) -> str:
        """Resolve a model identifier to a local .gguf file path.

        Supports:
        1. Direct path to a .gguf file
        2. HuggingFace repo ID — searches the HF cache for .gguf files
        """
        import os

        # Direct .gguf path
        if model_id.endswith(".gguf") and os.path.isfile(model_id):
            return model_id

        # Search HF cache for downloaded .gguf files
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        repo_dir = os.path.join(cache_dir, f"models--{model_id.replace('/', '--')}")

        if os.path.isdir(repo_dir):
            # Walk snapshots to find .gguf files
            for root, _, files in os.walk(repo_dir):
                for f in sorted(files):
                    if f.endswith(".gguf"):
                        return os.path.join(root, f)

        raise FileNotFoundError(
            f"No .gguf model found for '{model_id}'. "
            "Provide a direct path to a .gguf file or download a GGUF model: "
            "installm pull <repo-with-gguf-files>"
        )
