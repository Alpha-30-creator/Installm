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
        return {"models": {}, "aliases": {}, "server": None}
    state = {}
    with open(STATE_FILE, "r") as f:
        state = json.load(f)
    # Ensure aliases key exists (backward compat)
    if "aliases" not in state:
        state["aliases"] = {}
    return state


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
        # Also remove any aliases pointing to this model
        state["aliases"] = {
            a: m for a, m in state["aliases"].items() if m != model_id
        }
        save_state(state)
        return True
    return False


def list_models() -> dict:
    """Return all registered models from the manifest."""
    state = load_state()
    return state.get("models", {})


# --- Alias management ---

def set_alias(alias: str, model_id: str):
    """Map a short alias to a canonical model ID.

    Example: set_alias("llama", "meta-llama/Llama-3.1-8B-Instruct")
    """
    state = load_state()
    state["aliases"][alias] = model_id
    save_state(state)


def remove_alias(alias: str) -> bool:
    """Remove an alias. Returns True if it existed."""
    state = load_state()
    if alias in state["aliases"]:
        del state["aliases"][alias]
        save_state(state)
        return True
    return False


def resolve_alias(name: str) -> str:
    """Resolve a model name or alias to a canonical model ID.

    If `name` is an alias, returns the canonical model ID.
    Otherwise returns `name` unchanged.
    """
    state = load_state()
    return state.get("aliases", {}).get(name, name)


def list_aliases() -> dict:
    """Return all aliases as {alias: model_id}."""
    state = load_state()
    return state.get("aliases", {})


# --- HuggingFace token management ---

def save_hf_token(token: str):
    """Save a HuggingFace API token to the local state file.

    The token is stored in ~/.installm/state.json. This is a local
    developer machine store — treat it with the same care as an SSH key.
    """
    state = load_state()
    state["hf_token"] = token
    save_state(state)


def load_hf_token() -> Optional[str]:
    """Return the saved HuggingFace token, or None if not set.

    Resolution order:
      1. HF_TOKEN environment variable (takes priority)
      2. HUGGING_FACE_HUB_TOKEN environment variable
      3. Token saved via `installm token set`
    """
    import os
    env = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if env:
        return env
    state = load_state()
    return state.get("hf_token")


def clear_hf_token() -> bool:
    """Remove the saved HuggingFace token. Returns True if one existed."""
    state = load_state()
    if "hf_token" in state:
        del state["hf_token"]
        save_state(state)
        return True
    return False


# --- Server info ---

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
