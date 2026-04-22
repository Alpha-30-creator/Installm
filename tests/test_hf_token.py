"""Tests for HuggingFace token save/load/clear and CLI token commands."""

import os
import pytest
from click.testing import CliRunner
from unittest.mock import patch

from installm.cli import cli
from installm.config import save_hf_token, load_hf_token, clear_hf_token


# --- Config layer tests ---

def test_save_and_load_token(tmp_state):
    """Saved token is returned by load_hf_token."""
    save_hf_token("hf_testtoken123")
    assert load_hf_token() == "hf_testtoken123"


def test_clear_token_returns_true(tmp_state):
    """clear_hf_token returns True when a token existed."""
    save_hf_token("hf_testtoken123")
    assert clear_hf_token() is True
    assert load_hf_token() is None


def test_clear_token_no_token(tmp_state):
    """clear_hf_token returns False when no token is saved."""
    assert clear_hf_token() is False


def test_env_var_takes_priority(tmp_state):
    """HF_TOKEN env var takes priority over saved token."""
    save_hf_token("hf_saved")
    with patch.dict(os.environ, {"HF_TOKEN": "hf_from_env"}):
        assert load_hf_token() == "hf_from_env"


def test_hugging_face_hub_token_env_var(tmp_state):
    """HUGGING_FACE_HUB_TOKEN env var is also respected."""
    with patch.dict(os.environ, {"HUGGING_FACE_HUB_TOKEN": "hf_hub_env"},
                    clear=False):
        # Remove HF_TOKEN if present to test fallback
        env = {k: v for k, v in os.environ.items() if k != "HF_TOKEN"}
        env["HUGGING_FACE_HUB_TOKEN"] = "hf_hub_env"
        with patch.dict(os.environ, env, clear=True):
            assert load_hf_token() == "hf_hub_env"


def test_no_token_returns_none(tmp_state):
    """load_hf_token returns None when nothing is configured."""
    env = {k: v for k, v in os.environ.items()
           if k not in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")}
    with patch.dict(os.environ, env, clear=True):
        assert load_hf_token() is None


# --- CLI token command tests ---

def test_cli_token_set(tmp_state):
    """installm token set saves the token and confirms."""
    runner = CliRunner()
    result = runner.invoke(cli, ["token", "set", "hf_mytoken1234"])
    assert result.exit_code == 0
    assert "saved" in result.output
    assert "hf_mytok..." in result.output  # masked (8 chars + ...)
    assert load_hf_token() == "hf_mytoken1234"


def test_cli_token_clear_with_token(tmp_state):
    """installm token clear removes a saved token."""
    save_hf_token("hf_mytoken1234")
    runner = CliRunner()
    result = runner.invoke(cli, ["token", "clear"])
    assert result.exit_code == 0
    assert "removed" in result.output
    assert load_hf_token() is None


def test_cli_token_clear_no_token(tmp_state):
    """installm token clear reports when no token is saved."""
    runner = CliRunner()
    result = runner.invoke(cli, ["token", "clear"])
    assert result.exit_code == 0
    assert "No token" in result.output


def test_cli_token_status_saved(tmp_state):
    """installm token status shows masked saved token."""
    save_hf_token("hf_mytoken1234")
    runner = CliRunner()
    env = {k: v for k, v in os.environ.items()
           if k not in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")}
    result = runner.invoke(cli, ["token", "status"], env=env)
    assert result.exit_code == 0
    assert "state.json" in result.output


def test_cli_token_status_none(tmp_state):
    """installm token status shows instructions when no token is set."""
    runner = CliRunner()
    env = {k: v for k, v in os.environ.items()
           if k not in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")}
    result = runner.invoke(cli, ["token", "status"], env=env)
    assert result.exit_code == 0
    assert "No HuggingFace token" in result.output


def test_cli_token_status_env_var(tmp_state):
    """installm token status reports env var token."""
    runner = CliRunner()
    env = dict(os.environ)
    env["HF_TOKEN"] = "hf_envtoken5678"
    result = runner.invoke(cli, ["token", "status"], env=env)
    assert result.exit_code == 0
    assert "environment variable" in result.output


# --- download.py integration ---

def test_pull_model_passes_token(tmp_state):
    """pull_model forwards the token to snapshot_download."""
    from unittest.mock import patch as _patch
    from installm.download import pull_model

    with _patch("installm.download.snapshot_download", return_value="/cache/model") as mock_dl:
        pull_model("some/model", token="hf_tok")
        mock_dl.assert_called_once_with(
            repo_id="some/model",
            revision=None,
            token="hf_tok",
        )


def test_pull_model_no_token(tmp_state):
    """pull_model passes token=None when no token is provided."""
    from unittest.mock import patch as _patch
    from installm.download import pull_model

    with _patch("installm.download.snapshot_download", return_value="/cache/model") as mock_dl:
        pull_model("some/model")
        mock_dl.assert_called_once_with(
            repo_id="some/model",
            revision=None,
            token=None,
        )
