"""API key authentication for InstaLLM.

Keys are generated locally, shown once to the user, and stored as SHA-256
hashes in ~/.installm/state.json. Authentication is optional — if no keys
exist or auth is not enabled, all requests pass through.
"""

import hashlib
import secrets
import time
from typing import Optional

from installm.config import load_state, save_state

PREFIX = "sk-installm-"


def _hash(key: str) -> str:
    """SHA-256 hash of a key."""
    return hashlib.sha256(key.encode()).hexdigest()


def create_key(label: Optional[str] = None) -> tuple[str, str]:
    """Generate a new API key.

    Returns (full_key, key_id). The full key is shown once; only the
    hash is persisted.
    """
    raw = secrets.token_hex(24)  # 48 hex chars
    full_key = f"{PREFIX}{raw}"
    key_id = raw[:8]  # Short ID for display and revocation

    state = load_state()
    keys = state.setdefault("api_keys", {})
    keys[key_id] = {
        "hash": _hash(full_key),
        "label": label or "",
        "prefix": full_key[:20] + "...",
        "created_at": int(time.time()),
    }
    save_state(state)
    return full_key, key_id


def revoke_key(key_id: str) -> bool:
    """Revoke a key by its short ID. Returns True if found."""
    state = load_state()
    keys = state.get("api_keys", {})
    if key_id in keys:
        del keys[key_id]
        save_state(state)
        return True
    return False


def list_keys() -> dict:
    """Return all registered keys as {key_id: info} (no full keys)."""
    state = load_state()
    return state.get("api_keys", {})


def validate_key(bearer_token: str) -> bool:
    """Check if a bearer token matches any stored key hash."""
    token_hash = _hash(bearer_token)
    state = load_state()
    for info in state.get("api_keys", {}).values():
        if info["hash"] == token_hash:
            return True
    return False


def has_keys() -> bool:
    """Return True if at least one API key is registered."""
    state = load_state()
    return len(state.get("api_keys", {})) > 0
