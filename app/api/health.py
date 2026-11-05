from typing import Literal

from fastapi import APIRouter, Response
from pydantic import BaseModel

from app.services.dependency_health import get_dependency_health


router = APIRouter()


class OllamaHealth(BaseModel):
    status: Literal["ok", "error"]
    required_models_available: bool
    error: str | None = None


class QdrantHealth(BaseModel):
    status: Literal["ok", "error"]
    collection: str
    dimensions: int | None = None
    distance: str | None = None
    error: str | None = None


class DependenciesHealth(BaseModel):
    status: Literal["ok", "error"]
    ollama: OllamaHealth
    qdrant: QdrantHealth


@router.get(
    "/health/dependencies",
    response_model=DependenciesHealth,
    response_model_exclude_none=True,
    responses={503: {"model": DependenciesHealth}},
)
def health_dependencies(response: Response) -> DependenciesHealth:
    result = DependenciesHealth(**get_dependency_health())
    if result.status != "ok":
        response.status_code = 503
    return result


@router.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "ai-core",
        "version": "0.1.0",
    }
