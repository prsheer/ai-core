import json
import math
from http.client import HTTPException
from typing import Any, TypedDict
from urllib.error import HTTPError
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen

from app.core.config import (
    AI_CORE_QDRANT_COLLECTION,
    MAX_RESPONSE_BYTES,
    QDRANT_TIMEOUT_SECONDS,
    QDRANT_URL,
)


VECTOR_DIMENSIONS = 768
MAX_POINT_ID = (1 << 64) - 1


class VectorStoreInputError(ValueError):
    """Invalid vector store input; messages contain no supplied data."""


class VectorStoreError(Exception):
    """A vector store operation failed; messages are safe for consumers."""


class SearchResult(TypedDict):
    id: int
    score: float
    payload: dict[str, Any]


def _validate_tenant(tenant_id: str) -> None:
    if not isinstance(tenant_id, str) or not tenant_id.strip():
        raise VectorStoreInputError("A nonempty tenant ID is required.")


def _validate_vector(vector: list[float]) -> list[float]:
    if not isinstance(vector, list) or len(vector) != VECTOR_DIMENSIONS:
        raise VectorStoreInputError("Vector must contain exactly 768 values.")
    result = []
    for value in vector:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise VectorStoreInputError(
                "Vector values must be finite numbers."
            )
        try:
            number = float(value)
        except OverflowError:
            raise VectorStoreInputError(
                "Vector values must be finite numbers."
            ) from None
        if not math.isfinite(number):
            raise VectorStoreInputError(
                "Vector values must be finite numbers."
            )
        result.append(number)
    return result


def _validate_json(value: Any, depth: int = 0) -> None:
    """Require JSON types and string keys, with at most 20 nesting levels."""
    if depth > 20:
        raise VectorStoreInputError("Payload nesting exceeds the limit of 20.")
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float and math.isfinite(value):
        return
    if isinstance(value, list):
        for item in value:
            _validate_json(item, depth + 1)
        return
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        for item in value.values():
            _validate_json(item, depth + 1)
        return
    raise VectorStoreInputError("Payload must contain only valid JSON values.")


def _validate_payload(payload: dict[str, Any]) -> None:
    if not isinstance(payload, dict):
        raise VectorStoreInputError("Payload must be a JSON object.")
    _validate_json(payload)
    _validate_tenant(payload.get("tenant_id"))
    if payload.get("source_type") != "demo":
        raise VectorStoreInputError("Payload source_type must be demo.")


class QdrantVectorStore:
    """Demo vectors only. Future API tenants must come from trusted auth.

    Point IDs are collection-wide: callers must allocate stable, globally
    unique IDs across tenants to avoid overwriting another tenant's point.
    This service does not create collections or retry writes.
    """

    def upsert_point(
        self, point_id: int, vector: list[float], payload: dict[str, Any]
    ) -> None:
        if type(point_id) is not int or not 1 <= point_id <= MAX_POINT_ID:
            raise VectorStoreInputError("Point ID must be a positive uint64.")
        validated_vector = _validate_vector(vector)
        _validate_payload(payload)
        result = self._request("PUT", "/points?wait=true", {
            "points": [{
                "id": point_id,
                "vector": validated_vector,
                "payload": payload,
            }],
        })
        if (
            result.get("status") != "completed"
            or type(result.get("operation_id")) is not int
            or result["operation_id"] < 0
        ):
            raise VectorStoreError("Vector upsert completion not confirmed.")

    def search(
        self, vector: list[float], tenant_id: str, limit: int = 3
    ) -> list[SearchResult]:
        validated_vector = _validate_vector(vector)
        _validate_tenant(tenant_id)
        if type(limit) is not int or not 1 <= limit <= 10:
            raise VectorStoreInputError("Search limit must be from 1 to 10.")
        result = self._request("POST", "/points/query", {
            "query": validated_vector,
            "filter": {"must": [
                {"key": "tenant_id", "match": {"value": tenant_id}},
                {"key": "source_type", "match": {"value": "demo"}},
            ]},
            "limit": limit,
            "with_payload": True,
            "with_vector": False,
        })
        points = result.get("points")
        if not isinstance(points, list) or len(points) > limit:
            raise VectorStoreError("Invalid vector store search results.")
        matches: list[SearchResult] = []
        for point in points:
            if not isinstance(point, dict):
                raise VectorStoreError("Invalid vector store point.")
            point_id = point.get("id")
            score = point.get("score")
            payload = point.get("payload")
            if (
                type(point_id) is not int or not 1 <= point_id <= MAX_POINT_ID
                or isinstance(score, bool)
                or not isinstance(score, (int, float))
            ):
                raise VectorStoreError("Invalid vector store point data.")
            try:
                score = float(score)
                _validate_payload(payload)
            except (OverflowError, VectorStoreInputError):
                raise VectorStoreError(
                    "Vector store returned invalid point data."
                ) from None
            if not math.isfinite(score) or payload["tenant_id"] != tenant_id:
                raise VectorStoreError("Invalid vector store point data.")
            matches.append({
                "id": point_id, "score": score, "payload": payload,
            })
        return matches

    def _request(
        self, method: str, path: str, body: dict[str, Any]
    ) -> dict[str, Any]:
        if (
            not AI_CORE_QDRANT_COLLECTION.strip()
            or AI_CORE_QDRANT_COLLECTION == "easynet_knowledge"
            or not math.isfinite(QDRANT_TIMEOUT_SECONDS)
            or QDRANT_TIMEOUT_SECONDS <= 0
        ):
            raise VectorStoreError("Invalid AI Core vector store settings.")
        try:
            parsed_url = urlsplit(QDRANT_URL)
            if (
                parsed_url.scheme not in {"http", "https"}
                or not parsed_url.hostname
            ):
                raise VectorStoreError("Invalid vector store URL settings.")
            data = json.dumps(body, allow_nan=False).encode("utf-8")
            if len(data) > MAX_RESPONSE_BYTES:
                raise VectorStoreInputError(
                    "Vector request exceeds size limit."
                )
            collection = quote(AI_CORE_QDRANT_COLLECTION, safe="")
            request = Request(
                QDRANT_URL.rstrip("/") + f"/collections/{collection}" + path,
                data=data,
                headers={"Content-Type": "application/json"},
                method=method,
            )
            with urlopen(request, timeout=QDRANT_TIMEOUT_SECONDS) as response:
                if not 200 <= response.status < 300:
                    raise VectorStoreError("Vector store request rejected.")
                raw = response.read(MAX_RESPONSE_BYTES + 1)
        except HTTPError:
            raise VectorStoreError("Vector store request rejected.") from None
        except (OSError, HTTPException):
            raise VectorStoreError(
                "Vector store is unavailable or the request timed out."
            ) from None
        except VectorStoreInputError:
            raise
        except (ValueError, OverflowError):
            raise VectorStoreError("Invalid vector store request.") from None
        if len(raw) > MAX_RESPONSE_BYTES:
            raise VectorStoreError("Vector store response exceeds size limit.")
        try:
            response_body = json.loads(raw)
        except (ValueError, RecursionError):
            raise VectorStoreError("Invalid vector store response.") from None
        if (
            not isinstance(response_body, dict)
            or response_body.get("status") != "ok"
            or not isinstance(response_body.get("result"), dict)
        ):
            raise VectorStoreError("Invalid vector store operation response.")
        return response_body["result"]
