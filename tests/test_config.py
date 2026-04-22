"""Tests for config and manifest management."""

import json
from installm.config import (
    load_state, save_state, add_model, remove_model,
    list_models, set_server_info, clear_server_info, get_server_info,
)


def test_load_empty_state(isolated_state):
    """Fresh state should have empty models and no server."""
    state = load_state()
    assert state == {"models": {}, "aliases": {}, "server": None}


def test_save_and_load(isolated_state):
    """State should round-trip through save/load."""
    state = {"models": {"gpt2": {"model_id": "gpt2"}}, "server": None}
    save_state(state)
    loaded = load_state()
    # load_state adds aliases key for backward compat
    assert loaded["models"] == state["models"]
    assert loaded["server"] == state["server"]
    assert loaded["aliases"] == {}


def test_add_model(isolated_state):
    """Adding a model should persist it in the manifest."""
    entry = add_model("gpt2", backend="transformers")
    assert entry["model_id"] == "gpt2"
    assert entry["backend"] == "transformers"
    assert entry["status"] == "downloaded"

    models = list_models()
    assert "gpt2" in models


def test_remove_model(isolated_state):
    """Removing a model should delete it from the manifest."""
    add_model("gpt2")
    assert remove_model("gpt2") is True
    assert remove_model("gpt2") is False  # already gone
    assert "gpt2" not in list_models()


def test_list_models_empty(isolated_state):
    """Listing with no models should return empty dict."""
    assert list_models() == {}


def test_server_info_lifecycle(isolated_state):
    """Server info should be settable and clearable."""
    assert get_server_info() is None

    set_server_info("0.0.0.0", 8000, pid=1234)
    info = get_server_info()
    assert info["host"] == "0.0.0.0"
    assert info["port"] == 8000
    assert info["pid"] == 1234

    clear_server_info()
    assert get_server_info() is None
