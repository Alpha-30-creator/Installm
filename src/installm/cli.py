"""CLI entry point for InstaLLM."""

import click

from installm import __version__
from installm.config import (
    list_models, get_server_info, clear_server_info,
    DEFAULT_HOST, DEFAULT_PORT,
    set_alias, remove_alias, list_aliases,
    save_hf_token, load_hf_token, clear_hf_token,
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
@click.option("--backend", default=None, type=click.Choice(["ollama", "transformers", "vllm", "llamacpp"]),
              help="Force a specific backend (auto-detected if omitted).")
@click.option("--require-auth", is_flag=True, default=False,
              help="Require API key authentication for all requests.")
@click.option("--token", default=None, envvar="HF_TOKEN",
              help="HuggingFace token for gated models (one-off; use 'installm token set' to save permanently).")
def up(model, host, port, backend, require_auth, token):
    """Download model(s) and start the API server.

    Example:
        installm up --model meta-llama/Llama-3-8B
    """
    # Lazy imports to keep CLI startup fast
    from installm.download import pull_model
    from installm.backends import select_backend
    from installm.gateway.app import start_server
    from installm.config import add_model, set_server_info

    # Resolve HF token: CLI flag > saved token > env var (handled by load_hf_token)
    hf_token = token or load_hf_token()

    loaded = {}
    for m in model:
        click.echo(f">> Pulling {m}...")
        path = pull_model(m, token=hf_token)
        click.echo(f"   Downloaded to {path}")

        be_name = backend or select_backend(m)
        click.echo(f">> Selected backend: {be_name}")
        add_model(m, backend=be_name)
        loaded[m] = be_name

    # Enable auth if requested
    import os
    if require_auth:
        os.environ["INSTALLM_REQUIRE_AUTH"] = "1"
        click.echo(">> Authentication enabled. Use 'installm auth create' to generate keys.")

    click.echo(f"\n>> Starting API server on {host}:{port}...")
    set_server_info(host, port)
    click.echo(f">> Your API is live at http://{host}:{port}")
    click.echo(f"   Export: export OPENAI_BASE_URL=http://{host}:{port}/v1\n")

    start_server(host, port, loaded)


@cli.command()
@click.option("--model", "-m", required=True,
              help="Hugging Face model ID to download.")
@click.option("--token", default=None, envvar="HF_TOKEN",
              help="HuggingFace token for gated models (one-off; use 'installm token set' to save permanently).")
def pull(model, token):
    """Download a model without starting the server.

    Example:
        installm pull --model gpt2
    """
    from installm.download import pull_model
    from installm.config import add_model

    hf_token = token or load_hf_token()
    click.echo(f">> Pulling {model}...")
    path = pull_model(model, token=hf_token)
    add_model(model)
    click.echo(f">> Done. Cached at {path}")


@cli.command(name="ls")
def list_cmd():
    """List all downloaded models and aliases."""
    models = list_models()
    aliases = list_aliases()

    if not models and not aliases:
        click.echo("No models installed. Run: installm pull --model <model_id>")
        return

    if models:
        click.echo(f"{'Model ID':<40} {'Backend':<15} {'Status':<12}")
        click.echo("-" * 67)
        for mid, info in models.items():
            be = info.get("backend") or "auto"
            status = info.get("status", "unknown")
            click.echo(f"{mid:<40} {be:<15} {status:<12}")

    if aliases:
        click.echo(f"\n{'Alias':<25} {'Model ID':<45}")
        click.echo("-" * 70)
        for alias, model_id in aliases.items():
            click.echo(f"{alias:<25} {model_id:<45}")


@cli.command()
@click.argument("alias")
@click.argument("model_id")
def alias(alias, model_id):
    """Create a short alias for a model ID.

    Example:
        installm alias llama meta-llama/Llama-3.1-8B-Instruct
    """
    set_alias(alias, model_id)
    click.echo(f">> Alias '{alias}' -> '{model_id}'")


@cli.command(name="unalias")
@click.argument("alias")
def unalias(alias):
    """Remove a model alias.

    Example:
        installm unalias llama
    """
    if remove_alias(alias):
        click.echo(f">> Removed alias '{alias}'.")
    else:
        click.echo(f">> Alias '{alias}' not found.")


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


# --- HuggingFace token management ---

@cli.group()
def token():
    """Manage your HuggingFace API token for gated models."""


@token.command(name="set")
@click.argument("hf_token")
def token_set(hf_token):
    """Save a HuggingFace token for gated model access.

    The token is stored in ~/.installm/state.json and used automatically
    by 'installm pull' and 'installm up' without needing to set HF_TOKEN.

    Example:
        installm token set hf_xxxxxxxxxxxxxxxxxxxxxxxx
    """
    save_hf_token(hf_token)
    masked = hf_token[:8] + "..." + hf_token[-4:]
    click.echo(f">> HuggingFace token saved ({masked}).")
    click.echo("   It will be used automatically for all future model downloads.")


@token.command(name="clear")
def token_clear():
    """Remove the saved HuggingFace token.

    Example:
        installm token clear
    """
    if clear_hf_token():
        click.echo(">> HuggingFace token removed.")
    else:
        click.echo(">> No token was saved.")


@token.command(name="status")
def token_status():
    """Show whether a HuggingFace token is currently saved.

    Example:
        installm token status
    """
    import os
    env_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    saved_token = load_hf_token()

    if env_token:
        masked = env_token[:8] + "..." + env_token[-4:]
        click.echo(f">> Token active via environment variable ({masked}).")
    elif saved_token:
        masked = saved_token[:8] + "..." + saved_token[-4:]
        click.echo(f">> Token saved in ~/.installm/state.json ({masked}).")
    else:
        click.echo(">> No HuggingFace token configured.")
        click.echo("   Run: installm token set <your_token>")
        click.echo("   Or:  export HF_TOKEN=<your_token>")


# --- Auth key management ---

@cli.group()
def auth():
    """Manage API key authentication."""


@auth.command(name="create")
@click.option("--label", "-l", default=None, help="Optional label for the key.")
def auth_create(label):
    """Generate a new API key.

    The full key is shown once — save it securely.

    Example:
        installm auth create --label "dev-laptop"
    """
    from installm.auth import create_key

    full_key, key_id = create_key(label)
    click.echo(f">> New API key created (ID: {key_id})")
    click.echo(f"   Key: {full_key}")
    click.echo("   Save this key — it will not be shown again.")
    click.echo(f"\n   Usage: export OPENAI_API_KEY={full_key}")


@auth.command(name="revoke")
@click.argument("key_id")
def auth_revoke(key_id):
    """Revoke an API key by its short ID.

    Example:
        installm auth revoke a1b2c3d4
    """
    from installm.auth import revoke_key

    if revoke_key(key_id):
        click.echo(f">> Key {key_id} revoked.")
    else:
        click.echo(f">> Key {key_id} not found.")


@auth.command(name="ls")
def auth_list():
    """List all active API keys (shows prefix only)."""
    from installm.auth import list_keys

    keys = list_keys()
    if not keys:
        click.echo("No API keys configured. Run: installm auth create")
        return

    click.echo(f"{'ID':<12} {'Prefix':<25} {'Label':<20} {'Created'}")
    click.echo("-" * 75)
    for kid, info in keys.items():
        import time as _t
        created = _t.strftime("%Y-%m-%d %H:%M", _t.localtime(info["created_at"]))
        click.echo(f"{kid:<12} {info['prefix']:<25} {info.get('label', ''):<20} {created}")


if __name__ == "__main__":
    cli()
