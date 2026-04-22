"""Tests for model alias functionality."""

import pytest
from installm.config import (
    load_state, save_state, set_alias, remove_alias,
    resolve_alias, list_aliases, add_model, remove_model,
)


@pytest.fixture(autouse=True)
def clean_state(tmp_path, monkeypatch):
    """Use a temporary state file for each test."""
    import installm.config as cfg
    monkeypatch.setattr(cfg, "INSTALLM_DIR", tmp_path)
    monkeypatch.setattr(cfg, "STATE_FILE", tmp_path / "state.json")


def test_set_and_resolve_alias():
    """Setting an alias should make it resolvable."""
    set_alias("llama", "meta-llama/Llama-3.1-8B-Instruct")
    assert resolve_alias("llama") == "meta-llama/Llama-3.1-8B-Instruct"


def test_resolve_unknown_alias():
    """Resolving an unknown alias returns the name unchanged."""
    assert resolve_alias("unknown-model") == "unknown-model"


def test_list_aliases():
    """list_aliases returns all set aliases."""
    set_alias("llama", "meta-llama/Llama-3.1-8B-Instruct")
    set_alias("qwen", "Qwen/Qwen2.5-0.5B-Instruct")
    aliases = list_aliases()
    assert aliases["llama"] == "meta-llama/Llama-3.1-8B-Instruct"
    assert aliases["qwen"] == "Qwen/Qwen2.5-0.5B-Instruct"


def test_remove_alias():
    """Removing an alias should make it no longer resolvable."""
    set_alias("llama", "meta-llama/Llama-3.1-8B-Instruct")
    assert remove_alias("llama") is True
    assert resolve_alias("llama") == "llama"


def test_remove_nonexistent_alias():
    """Removing a non-existent alias returns False."""
    assert remove_alias("nope") is False


def test_remove_model_cleans_aliases():
    """Removing a model should also remove any aliases pointing to it."""
    add_model("meta-llama/Llama-3.1-8B-Instruct", backend="transformers")
    set_alias("llama", "meta-llama/Llama-3.1-8B-Instruct")
    set_alias("llama2", "meta-llama/Llama-3.1-8B-Instruct")
    set_alias("qwen", "Qwen/Qwen2.5-0.5B-Instruct")

    remove_model("meta-llama/Llama-3.1-8B-Instruct")

    aliases = list_aliases()
    assert "llama" not in aliases
    assert "llama2" not in aliases
    assert "qwen" in aliases


def test_overwrite_alias():
    """Setting an alias that already exists should overwrite it."""
    set_alias("llama", "meta-llama/Llama-3.1-8B-Instruct")
    set_alias("llama", "meta-llama/Llama-3.2-1B-Instruct")
    assert resolve_alias("llama") == "meta-llama/Llama-3.2-1B-Instruct"


def test_backward_compat_no_aliases_key():
    """State files without an aliases key should still work."""
    save_state({"models": {}, "server": None})
    state = load_state()
    assert "aliases" in state
    assert state["aliases"] == {}
