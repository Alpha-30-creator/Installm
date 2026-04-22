"""vLLM backend - high-performance inference for NVIDIA GPUs on Linux."""

import platform
import time
import uuid
from typing import AsyncIterator

from installm.backends.base import Backend


def _check_platform():
    """vLLM only runs on Linux with an NVIDIA GPU."""
    if platform.system() != "Linux":
        raise RuntimeError(
            f"vLLM is not supported on {platform.system()}. "
            "Use the 'transformers' backend instead."
        )
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError(
                "vLLM requires an NVIDIA GPU with CUDA. "
                "No CUDA device detected. Use the 'transformers' backend instead."
            )
    except ImportError:
        raise RuntimeError("torch is required for vLLM. Install: pip install torch")


class VLLMBackend(Backend):
    """Backend using vLLM for high-throughput GPU inference (Linux + NVIDIA only)."""

    def __init__(self):
        self.engine = None
        self.model_id = None

    async def load(self, model_id: str, **kwargs):
        """Initialise the vLLM AsyncLLMEngine."""
        _check_platform()

        try:
            from vllm import AsyncLLMEngine, AsyncEngineArgs
        except ImportError:
            raise RuntimeError(
                "vLLM is not installed. Install: pip install installm[vllm]"
            )

        self.model_id = model_id
        engine_args = AsyncEngineArgs(
            model=model_id,
            tensor_parallel_size=kwargs.get("tensor_parallel_size", 1),
            dtype=kwargs.get("dtype", "auto"),
            max_model_len=kwargs.get("max_model_len", None),
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

    async def generate(self, messages: list, **kwargs) -> dict:
        """Non-streaming generation via vLLM engine."""

        prompt = self._messages_to_prompt(messages)
        params = self._build_sampling_params(kwargs)
        request_id = uuid.uuid4().hex

        output_text = ""
        async for output in self.engine.generate(prompt, params, request_id):
            if output.finished:
                output_text = output.outputs[0].text
                prompt_tokens = len(output.prompt_token_ids)
                completion_tokens = len(output.outputs[0].token_ids)

        return {
            "id": f"chatcmpl-{request_id[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_id,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": output_text},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    async def stream(self, messages: list, **kwargs) -> AsyncIterator[dict]:
        """Streaming generation via vLLM engine."""

        prompt = self._messages_to_prompt(messages)
        params = self._build_sampling_params(kwargs)
        request_id = uuid.uuid4().hex
        chunk_id = f"chatcmpl-{request_id[:8]}"
        created = int(time.time())

        prev_text = ""
        async for output in self.engine.generate(prompt, params, request_id):
            new_text = output.outputs[0].text[len(prev_text):]
            prev_text = output.outputs[0].text

            if new_text:
                yield {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": self.model_id,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": new_text},
                        "finish_reason": None,
                    }],
                }

            if output.finished:
                yield {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": self.model_id,
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }],
                }

    async def embed(self, texts: list[str], **kwargs) -> list[list[float]]:
        """vLLM does not natively expose embeddings via AsyncLLMEngine."""
        raise NotImplementedError(
            "Embeddings are not supported by the vLLM backend. "
            "Use a dedicated embedding model with the transformers backend."
        )

    @property
    def supports_tools(self) -> bool:
        return True  # vLLM supports constrained tool calling

    @property
    def supports_structured_output(self) -> bool:
        return True  # vLLM supports JSON schema constrained decoding

    async def unload(self):
        """Shut down the vLLM engine."""
        if self.engine is not None:
            # vLLM engine cleanup
            del self.engine
            self.engine = None

    def _messages_to_prompt(self, messages: list) -> str:
        """Convert OpenAI-style messages to a plain prompt string."""
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"{role}: {content}")
        parts.append("assistant:")
        return "\n".join(parts)

    def _build_sampling_params(self, kwargs: dict):
        """Build vLLM SamplingParams from request kwargs."""
        from vllm import SamplingParams
        return SamplingParams(
            max_tokens=kwargs.get("max_tokens", 512),
            temperature=kwargs.get("temperature", 0.7),
            top_p=kwargs.get("top_p", 1.0),
        )
