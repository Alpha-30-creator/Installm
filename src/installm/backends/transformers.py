"""Hugging Face Transformers backend - runs models directly via pipeline."""

import asyncio
import time
import uuid
from typing import AsyncIterator

from installm.backends.base import Backend


def _detect_device() -> str:
    """Pick the best available device: cuda > mps > cpu."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


class TransformersBackend(Backend):
    """Backend using HF Transformers for local inference (CPU/MPS/CUDA)."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = None
        self.model_id = None

    async def load(self, model_id: str, **kwargs):
        """Load model and tokenizer from HF Hub or local cache."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise RuntimeError(
                "transformers is not installed. Install: pip install installm[transformers]"
            )

        self.model_id = model_id
        self.device = kwargs.get("device") or _detect_device()

        # Run blocking load in a thread to keep async loop free
        def _load():
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map=self.device if self.device != "cpu" else None,
                torch_dtype="auto",
            )
            if self.device == "cpu" or self.device == "mps":
                model = model.to(self.device)
            return model, tokenizer

        self.model, self.tokenizer = await asyncio.to_thread(_load)

        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _format_prompt(self, messages: list) -> str:
        """Format messages into a prompt string.

        Uses the tokenizer's chat template when available, otherwise falls back
        to a simple role: content format that works with any model.
        """
        if self.tokenizer.chat_template is not None:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        # Fallback for models without a chat template (e.g. base GPT-2)
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content") or ""
            parts.append(f"{role}: {content}")
        parts.append("assistant:")
        return "\n".join(parts)

    async def generate(self, messages: list, **kwargs) -> dict:
        """Generate a complete response (non-streaming)."""
        import torch

        prompt = self._format_prompt(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        max_tokens = kwargs.get("max_tokens", 512)
        temperature = kwargs.get("temperature", 0.7)
        top_p = kwargs.get("top_p", 1.0)

        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p

        def _gen():
            with torch.no_grad():
                output = self.model.generate(**inputs, **gen_kwargs)
            # Decode only the new tokens
            new_tokens = output[0][inputs["input_ids"].shape[1]:]
            return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        text = await asyncio.to_thread(_gen)

        # Build OpenAI-compatible response
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_id,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": inputs["input_ids"].shape[1],
                "completion_tokens": len(self.tokenizer.encode(text)),
                "total_tokens": inputs["input_ids"].shape[1] + len(self.tokenizer.encode(text)),
            },
        }

    async def stream(self, messages: list, **kwargs) -> AsyncIterator[dict]:
        """Yield response chunks using TextIteratorStreamer."""
        from threading import Thread
        from transformers import TextIteratorStreamer

        prompt = self._format_prompt(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        max_tokens = kwargs.get("max_tokens", 512)
        temperature = kwargs.get("temperature", 0.7)
        top_p = kwargs.get("top_p", 1.0)

        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p

        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True,
        )
        gen_kwargs["streamer"] = streamer

        # Run generation in a background thread
        thread = Thread(
            target=lambda: self.model.generate(**inputs, **gen_kwargs),
            daemon=True,
        )
        thread.start()

        chunk_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        created = int(time.time())

        for token_text in streamer:
            if not token_text:
                continue
            yield {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": self.model_id,
                "choices": [{
                    "index": 0,
                    "delta": {"content": token_text},
                    "finish_reason": None,
                }],
            }

        # Final chunk with finish_reason
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
        """Generate embeddings using the model's hidden states.

        Note: This is a basic implementation. For production embeddings,
        use a dedicated embedding model.
        """
        import torch

        results = []
        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True,
                                     max_length=512).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
            # Use mean of last hidden state as embedding
            hidden = outputs.hidden_states[-1]
            embedding = hidden.mean(dim=1).squeeze().cpu().tolist()
            if isinstance(embedding, float):
                embedding = [embedding]
            results.append(embedding)
        return results

    @property
    def supports_tools(self) -> bool:
        return False  # Handled by gateway's prompt-and-parse fallback

    @property
    def supports_structured_output(self) -> bool:
        return False  # Handled by gateway's validate-and-retry fallback

    async def unload(self):
        """Release model from memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
