"""GET /v1/models - list available models."""

import time
from fastapi import APIRouter
from installm.gateway.schemas import ModelInfo, ModelList
from installm.config import list_models, list_aliases

router = APIRouter()


@router.get("/v1/models", response_model=ModelList)
async def get_models():
    """Return all models registered in the InstaLLM manifest.

    Aliases are included as separate entries pointing to the same model.
    """
    manifest = list_models()
    aliases = list_aliases()

    data = [
        ModelInfo(
            id=model_id,
            created=info.get("added_at", int(time.time())),
            owned_by="installm",
        )
        for model_id, info in manifest.items()
    ]

    # Add alias entries (so clients can use the alias as model name)
    for alias, model_id in aliases.items():
        if model_id in manifest:
            data.append(ModelInfo(
                id=alias,
                created=manifest[model_id].get("added_at", int(time.time())),
                owned_by="installm",
            ))

    return ModelList(data=data)
