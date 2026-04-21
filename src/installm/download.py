"""Model download and cache management via Hugging Face Hub."""

from huggingface_hub import snapshot_download


def pull_model(model_id: str, revision: str = None) -> str:
    """Download a model from Hugging Face Hub using cached snapshot.

    Returns the local cache path.
    """
    path = snapshot_download(
        repo_id=model_id,
        revision=revision,
        # Let HF Hub handle caching; re-runs skip unchanged files
    )
    return path
