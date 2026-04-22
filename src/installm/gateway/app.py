"""FastAPI application and server launcher for InstaLLM."""

import os
import logging
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger("installm")

app = FastAPI(
    title="InstaLLM",
    version="0.1.0",
    description="OpenAI-compatible API gateway for open-source LLMs",
)

# Allow all origins so the API can be used from any client
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Registry: model_id -> backend instance (populated at startup)
_backends: dict = {}


def get_backends() -> dict:
    """Return the active backend registry."""
    return _backends


def resolve_model(model_name: str):
    """Resolve a model name (possibly an alias) and return (canonical_id, backend).

    Returns the backend instance or None if not found.
    """
    # Direct match first
    if model_name in _backends:
        return _backends[model_name]
    # Try alias resolution
    from installm.config import resolve_alias
    canonical = resolve_alias(model_name)
    return _backends.get(canonical)


def register_backend(model_id: str, backend):
    """Register a backend instance for a model."""
    _backends[model_id] = backend
    logger.info("Registered backend for model: %s", model_id)


# Register all routes
from installm.gateway.routes import models, chat, embeddings, responses  # noqa: E402

app.include_router(models.router)
app.include_router(chat.router)
app.include_router(embeddings.router)
app.include_router(responses.router)


@app.get("/health")
async def health():
    """Simple liveness check."""
    return {"status": "ok", "models_loaded": len(_backends)}


def start_server(host: str, port: int, models: dict):
    """Start the uvicorn server. Called from CLI `up` command.

    `models` is a dict of {model_id: backend_name} from the CLI.
    Backends are loaded asynchronously before the server starts.
    """
    import asyncio
    from installm.config import set_server_info, INSTALLM_DIR

    pid = os.getpid()
    set_server_info(host, port, pid)

    log_file = INSTALLM_DIR / "server.log"
    INSTALLM_DIR.mkdir(parents=True, exist_ok=True)

    async def _load_backends():
        for model_id, backend_name in models.items():
            backend = _create_backend(backend_name)
            await backend.load(model_id)
            register_backend(model_id, backend)

    asyncio.run(_load_backends())

    uvicorn.run(
        "installm.gateway.app:app",
        host=host,
        port=port,
        log_level="info",
        log_config=None,
    )


def _create_backend(backend_name: str):
    """Instantiate a backend by name."""
    if backend_name == "ollama":
        from installm.backends.ollama import OllamaBackend
        return OllamaBackend()
    elif backend_name == "vllm":
        from installm.backends.vllm import VLLMBackend
        return VLLMBackend()
    elif backend_name == "llamacpp":
        from installm.backends.llamacpp import LlamaCppBackend
        return LlamaCppBackend()
    else:
        from installm.backends.transformers import TransformersBackend
        return TransformersBackend()
