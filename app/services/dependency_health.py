import json
from http.client import HTTPException
from typing import Any
from urllib.parse import quote, urlsplit
from urllib.request import urlopen

from app.core.config import (
    MAX_RESPONSE_BYTES,
    OLLAMA_BASE_URL,
    QDRANT_COLLECTION,
    QDRANT_URL,
    REQUEST_TIMEOUT_SECONDS,
)


class DependencyCheckError(Exception):
    """A dependency could not provide a valid health response."""


def fetch_dependency_json(base_url: str, path: str) -> dict[str, Any]:
    try:
        parsed_url = urlsplit(base_url)
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.hostname:
            raise DependencyCheckError("Invalid dependency URL configuration.")
        with urlopen(
            base_url.rstrip("/") + path, timeout=REQUEST_TIMEOUT_SECONDS
        ) as response:
            body = response.read(MAX_RESPONSE_BYTES + 1)
        if len(body) > MAX_RESPONSE_BYTES:
            raise DependencyCheckError("Dependency response exceeds size limit.")
        payload = json.loads(body)
    except (OSError, HTTPException, ValueError) as exc:
        raise DependencyCheckError(
            "Dependency unavailable or returned an invalid response."
        ) from exc
    if not isinstance(payload, dict):
        raise DependencyCheckError("Invalid dependency response.")
    return payload


def check_ollama() -> dict[str, Any]:
    try:
        payload = fetch_dependency_json(OLLAMA_BASE_URL, "/api/tags")
        models = payload.get("models")
        if not isinstance(models, list) or any(
            not isinstance(model, dict)
            or not isinstance(model.get("name"), str)
            for model in models
        ):
            raise DependencyCheckError("Invalid model list response.")
        names = {model["name"] for model in models}
        if "qwen2.5:3b" not in names or not names.intersection(
            {"embeddinggemma", "embeddinggemma:latest"}
        ):
            raise DependencyCheckError("Required models are missing.")
        return dict(status="ok", required_models_available=True)
    except DependencyCheckError as exc:
        return dict(
            status="error", required_models_available=False, error=str(exc)
        )


def check_qdrant() -> dict[str, Any]:
    try:
        if not QDRANT_COLLECTION.strip():
            raise DependencyCheckError("Collection name is not configured.")
        payload = fetch_dependency_json(
            QDRANT_URL, f"/collections/{quote(QDRANT_COLLECTION, safe='')}"
        )
        result = payload.get("result")
        if not isinstance(result, dict):
            raise DependencyCheckError("Invalid collection response.")
        config = result.get("config")
        params = config.get("params") if isinstance(config, dict) else None
        vectors = params.get("vectors") if isinstance(params, dict) else None
        if not isinstance(vectors, dict):
            raise DependencyCheckError("Invalid vector configuration.")
        if (
            result.get("status") != "green"
            or type(vectors.get("size")) is not int
            or vectors.get("size") != 768
            or vectors.get("distance") != "Cosine"
        ):
            raise DependencyCheckError(
                "Collection must be green with 768 dimensions and Cosine distance."
            )
        return dict(
            status="ok",
            collection=QDRANT_COLLECTION,
            dimensions=768,
            distance="Cosine",
        )
    except DependencyCheckError as exc:
        return dict(
            status="error", collection=QDRANT_COLLECTION, error=str(exc)
        )


def get_dependency_health() -> dict[str, Any]:
    ollama = check_ollama()
    qdrant = check_qdrant()
    healthy = ollama["status"] == "ok" and qdrant["status"] == "ok"
    return {
        "status": "ok" if healthy else "error",
        "ollama": ollama,
        "qdrant": qdrant,
    }
