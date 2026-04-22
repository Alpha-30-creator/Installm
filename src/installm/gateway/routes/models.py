"""GET /v1/models - list available models."""

import time
from fastapi import APIRouter
from installm.gateway.schemas import ModelInfo, ModelList
from installm.config import list_models

router = APIRouter()


@router.get("/v1/models", response_model=ModelList)
async def get_models():
    """Return all models registered in the InstaLLM manifest."""
    manifest = list_models()
    data = [
        ModelInfo(
            id=model_id,
            created=info.get("added_at", int(time.time())),
            owned_by="installm",
        )
        for model_id, info in manifest.items()
    ]
    return ModelList(data=data)
