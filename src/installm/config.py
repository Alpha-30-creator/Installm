"""Configuration and local state management for InstaLLM."""

import json
import time
from pathlib import Path
from typing import Optional

# Default paths
INSTALLM_DIR = Path.home() / ".installm"
STATE_FILE = INSTALLM_DIR / "state.json"
DEFAULT_PORT = 8000
DEFAULT_HOST = "0.0.0.0"


def _ensure_dir():
    """Create the .installm directory if it doesn't exist."""
    INSTALLM_DIR.mkdir(parents=True, exist_ok=True)


def load_state() -> dict:
    """Load the state manifest from disk. Returns empty state if none exists."""
    if not STATE_FILE.exists():
        return {"models": {}, "server": None}
    with open(STATE_FILE, "r") as f:
        return json.load(f)


def save_state(state: dict):
    """Persist the state manifest to disk."""
    _ensure_dir()
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def add_model(model_id: str, backend: Optional[str] = None,
              revision: Optional[str] = None) -> dict:
    """Register a model in the manifest. Returns the model entry."""
    state = load_state()
    entry = {
        "model_id": model_id,
        "backend": backend,
        "revision": revision,
        "added_at": int(time.time()),
        "status": "downloaded",
    }
    state["models"][model_id] = entry
    save_state(state)
    return entry


def remove_model(model_id: str) -> bool:
    """Remove a model from the manifest. Returns True if it existed."""
    state = load_state()
    if model_id in state["models"]:
        del state["models"][model_id]
        save_state(state)
        return True
    return False


def list_models() -> dict:
    """Return all registered models from the manifest."""
    state = load_state()
    return state.get("models", {})


def set_server_info(host: str, port: int, pid: Optional[int] = None):
    """Record the running server's connection info."""
    state = load_state()
    state["server"] = {
        "host": host,
        "port": port,
        "pid": pid,
        "started_at": int(time.time()),
    }
    save_state(state)


def clear_server_info():
    """Clear server info from state (used on shutdown)."""
    state = load_state()
    state["server"] = None
    save_state(state)


def get_server_info() -> Optional[dict]:
    """Return current server info, or None if not running."""
    state = load_state()
    return state.get("server")
