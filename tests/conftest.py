"""Shared test fixtures for InstaLLM."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch

from installm.config import STATE_FILE, INSTALLM_DIR


@pytest.fixture(autouse=True)
def isolated_state(tmp_path):
    """Redirect state file to a temp directory for test isolation."""
    test_dir = tmp_path / ".installm"
    test_dir.mkdir()
    test_state = test_dir / "state.json"

    with patch("installm.config.INSTALLM_DIR", test_dir), \
         patch("installm.config.STATE_FILE", test_state):
        yield test_state
