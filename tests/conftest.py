"""Shared test fixtures for InstaLLM."""

import pytest
from unittest.mock import patch
from click.testing import CliRunner

from installm.config import STATE_FILE, INSTALLM_DIR


@pytest.fixture(autouse=True)
def isolated_state(tmp_path):
    """Redirect state file to a temp directory for test isolation.

    Patches both installm.config and installm.auth so that all modules
    read/write the same temporary state file.
    """
    test_dir = tmp_path / ".installm"
    test_dir.mkdir()
    test_state = test_dir / "state.json"

    with patch("installm.config.INSTALLM_DIR", test_dir), \
         patch("installm.config.STATE_FILE", test_state):
        yield test_state


# Alias for tests that explicitly request it
@pytest.fixture
def tmp_state(isolated_state):
    """Explicit alias for the isolated state fixture."""
    return isolated_state


@pytest.fixture
def cli_runner():
    """Provide a Click CLI test runner."""
    from installm.cli import cli
    runner = CliRunner()

    class _Runner:
        def invoke(self, args, **kwargs):
            return runner.invoke(cli, args, **kwargs)

    return _Runner()
