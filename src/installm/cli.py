"""CLI entry point for InstaLLM."""

import click

from installm import __version__
from installm.config import (
    list_models, get_server_info, clear_server_info,
    DEFAULT_HOST, DEFAULT_PORT,
)


@click.group()
@click.version_option(version=__version__, prog_name="installm")
def cli():
    """InstaLLM - One-command deployment of OpenAI-compatible APIs for open-source LLMs."""


@cli.command()
@click.option("--model", "-m", multiple=True, required=True,
              help="Hugging Face model ID to deploy (can be repeated).")
@click.option("--host", default=DEFAULT_HOST, show_default=True,
              help="Bind address for the API server.")
@click.option("--port", default=DEFAULT_PORT, type=int, show_default=True,
              help="Port for the API server.")
@click.option("--backend", default=None, type=click.Choice(["ollama", "transformers", "vllm"]),
              help="Force a specific backend (auto-detected if omitted).")
def up(model, host, port, backend):
    """Download model(s) and start the API server.

    Example:
        installm up --model meta-llama/Llama-3-8B
    """
    # Lazy imports to keep CLI startup fast
    from installm.download import pull_model
    from installm.backends import select_backend
    from installm.gateway.app import start_server
    from installm.config import add_model, set_server_info

    loaded = {}
    for m in model:
        click.echo(f">> Pulling {m}...")
        path = pull_model(m)
        click.echo(f"   Downloaded to {path}")

        be_name = backend or select_backend(m)
        click.echo(f">> Selected backend: {be_name}")
        add_model(m, backend=be_name)
        loaded[m] = be_name

    click.echo(f"\n>> Starting API server on {host}:{port}...")
    set_server_info(host, port)
    click.echo(f">> Your API is live at http://{host}:{port}")
    click.echo(f"   Export: export OPENAI_BASE_URL=http://{host}:{port}/v1\n")

    start_server(host, port, loaded)


@cli.command()
@click.option("--model", "-m", required=True,
              help="Hugging Face model ID to download.")
def pull(model):
    """Download a model without starting the server.

    Example:
        installm pull --model gpt2
    """
    from installm.download import pull_model
    from installm.config import add_model

    click.echo(f">> Pulling {model}...")
    path = pull_model(model)
    add_model(model)
    click.echo(f">> Done. Cached at {path}")


@cli.command(name="ls")
def list_cmd():
    """List all downloaded models."""
    models = list_models()
    if not models:
        click.echo("No models installed. Run: installm pull --model <model_id>")
        return

    click.echo(f"{'Model ID':<40} {'Backend':<15} {'Status':<12}")
    click.echo("-" * 67)
    for mid, info in models.items():
        be = info.get("backend") or "auto"
        status = info.get("status", "unknown")
        click.echo(f"{mid:<40} {be:<15} {status:<12}")


@cli.command()
def down():
    """Stop the running InstaLLM server."""
    import os
    import signal

    server = get_server_info()
    if not server or not server.get("pid"):
        click.echo("No running server found.")
        return

    pid = server["pid"]
    try:
        os.kill(pid, signal.SIGTERM)
        click.echo(f">> Stopped server (PID {pid}).")
    except ProcessLookupError:
        click.echo(f">> Server (PID {pid}) was already stopped.")
    clear_server_info()


@cli.command()
def logs():
    """Show recent server logs."""
    from installm.config import INSTALLM_DIR

    log_file = INSTALLM_DIR / "server.log"
    if not log_file.exists():
        click.echo("No logs found.")
        return

    with open(log_file) as f:
        # Show last 50 lines
        lines = f.readlines()
        for line in lines[-50:]:
            click.echo(line, nl=False)


if __name__ == "__main__":
    cli()
