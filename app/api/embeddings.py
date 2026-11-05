from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, StrictStr

from app.core.config import OLLAMA_EMBEDDING_MODEL
from app.services.ollama_embedding import (
    EmbeddingInputError,
    EmbeddingServiceError,
    OllamaEmbeddingService,
)


router = APIRouter()


class EmbeddingRequest(BaseModel):
    text: StrictStr


class EmbeddingResponse(BaseModel):
    model: str
    dimensions: int
    embedding: list[float]


@router.post("/embeddings", response_model=EmbeddingResponse)
def create_embedding(request: EmbeddingRequest) -> EmbeddingResponse:
    try:
        embedding = OllamaEmbeddingService().embed(request.text)
    except EmbeddingInputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    except EmbeddingServiceError:
        raise HTTPException(
            status_code=503, detail="Embedding service is unavailable."
        ) from None
    return EmbeddingResponse(
        model=OLLAMA_EMBEDDING_MODEL,
        dimensions=len(embedding),
        embedding=embedding,
    )
