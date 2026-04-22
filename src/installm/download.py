"""Model download and cache management via Hugging Face Hub."""

from huggingface_hub import snapshot_download


def pull_model(model_id: str, revision: str = None, token: str = None) -> str:
    """Download a model from Hugging Face Hub using cached snapshot.

    Args:
        model_id: HuggingFace repo ID, e.g. "meta-llama/Llama-3.1-8B-Instruct".
        revision: Optional git revision (branch, tag, or commit hash).
        token: HuggingFace API token for gated models. If None, falls back
               to the token saved via `installm token set` or the HF_TOKEN
               environment variable (resolved by the caller).

    Returns the local cache path.
    """
    path = snapshot_download(
        repo_id=model_id,
        revision=revision,
        token=token or None,
        # Let HF Hub handle caching; re-runs skip unchanged files
    )
    return path
