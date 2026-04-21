"""FastAPI application and server launcher."""

import os
import uvicorn
from fastapi import FastAPI

app = FastAPI(title="InstaLLM", version="0.1.0")

# Registry: model_id -> backend instance (populated at startup)
_backends: dict = {}


def get_backends() -> dict:
    """Return the active backend registry."""
    return _backends


def register_backend(model_id: str, backend):
    """Register a backend instance for a model."""
    _backends[model_id] = backend


@app.get("/health")
async def health():
    return {"status": "ok"}


def start_server(host: str, port: int, models: dict):
    """Start the uvicorn server. Called from CLI `up` command."""
    from installm.config import set_server_info

    pid = os.getpid()
    set_server_info(host, port, pid)

    uvicorn.run(
        "installm.gateway.app:app",
        host=host,
        port=port,
        log_level="info",
    )
