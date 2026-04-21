"""Abstract base class for inference backends."""

from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional


class Backend(ABC):
    """Base interface that all backends must implement."""

    @abstractmethod
    async def load(self, model_id: str, **kwargs):
        """Load a model and prepare it for inference."""

    @abstractmethod
    async def generate(self, messages: list, **kwargs) -> dict:
        """Generate a complete response (non-streaming)."""

    @abstractmethod
    async def stream(self, messages: list, **kwargs) -> AsyncIterator[dict]:
        """Yield response chunks for streaming."""

    @abstractmethod
    async def embed(self, texts: list[str], **kwargs) -> list[list[float]]:
        """Generate embeddings for a list of texts."""

    @property
    @abstractmethod
    def supports_tools(self) -> bool:
        """Whether this backend natively supports tool calling."""

    @property
    @abstractmethod
    def supports_structured_output(self) -> bool:
        """Whether this backend natively supports constrained JSON output."""

    @abstractmethod
    async def unload(self):
        """Release model resources."""
