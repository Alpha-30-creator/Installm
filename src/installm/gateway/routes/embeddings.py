"""POST /v1/embeddings - generate text embeddings."""

from fastapi import APIRouter, HTTPException

from installm.gateway.schemas import EmbeddingRequest, EmbeddingResponse, EmbeddingObject, Usage
from installm.gateway.app import get_backends

router = APIRouter()


@router.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(req: EmbeddingRequest):
    """Generate embeddings for the given input text(s)."""
    backends = get_backends()
    backend = backends.get(req.model)
    if backend is None:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{req.model}' is not loaded. Run: installm up --model {req.model}",
        )

    # Normalise input to a list
    texts = [req.input] if isinstance(req.input, str) else req.input

    try:
        vectors = await backend.embed(texts)
    except NotImplementedError:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{req.model}' does not support embeddings.",
        )

    data = [
        EmbeddingObject(embedding=vec, index=i)
        for i, vec in enumerate(vectors)
    ]

    # Approximate token count
    total_tokens = sum(len(t.split()) for t in texts)

    return EmbeddingResponse(
        data=data,
        model=req.model,
        usage=Usage(
            prompt_tokens=total_tokens,
            completion_tokens=0,
            total_tokens=total_tokens,
        ),
    )
