"""Tests for the CLI entry point."""

from click.testing import CliRunner
from installm.cli import cli


def test_help():
    """installm --help should succeed and show usage."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "InstaLLM" in result.output


def test_version():
    """installm --version should print the version."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_ls_empty(isolated_state):
    """installm ls with no models should show a helpful message."""
    runner = CliRunner()
    result = runner.invoke(cli, ["ls"])
    assert result.exit_code == 0
    assert "No models installed" in result.output


def test_down_no_server(isolated_state):
    """installm down with no running server should say so."""
    runner = CliRunner()
    result = runner.invoke(cli, ["down"])
    assert result.exit_code == 0
    assert "No running server" in result.output


def test_logs_no_file(isolated_state):
    """installm logs with no log file should say so."""
    runner = CliRunner()
    result = runner.invoke(cli, ["logs"])
    assert result.exit_code == 0
    assert "No logs found" in result.output
