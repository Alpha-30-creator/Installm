"""Tests for model download functionality."""

from unittest.mock import patch
from installm.download import pull_model


def test_pull_model_calls_snapshot_download():
    """pull_model should delegate to snapshot_download and return the path."""
    with patch("installm.download.snapshot_download", return_value="/cache/gpt2") as mock_dl:
        path = pull_model("gpt2")

    mock_dl.assert_called_once_with(repo_id="gpt2", revision=None)
    assert path == "/cache/gpt2"


def test_pull_model_with_revision():
    """pull_model should pass revision to snapshot_download."""
    with patch("installm.download.snapshot_download", return_value="/cache/gpt2") as mock_dl:
        pull_model("gpt2", revision="abc123")

    mock_dl.assert_called_once_with(repo_id="gpt2", revision="abc123")
