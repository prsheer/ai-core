import json
import math
from http.client import HTTPException
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from app.core.config import (
    EMBEDDING_DIMENSIONS,
    EMBEDDING_TIMEOUT_SECONDS,
    MAX_RESPONSE_BYTES,
    OLLAMA_BASE_URL,
    OLLAMA_EMBEDDING_MODEL,
)


# Character limit, not a token limit; Ollama also enforces model context limits.
MAX_EMBEDDING_TEXT_CHARACTERS = 8_000


class EmbeddingInputError(ValueError):
    """The input is unsuitable for embedding generation."""


class EmbeddingServiceError(Exception):
    """Embedding generation failed; messages are safe for API consumers."""


class OllamaEmbeddingService:
    """Generate one embedding for text of at most 8,000 characters."""

    def embed(self, text: str) -> list[float]:
        if not isinstance(text, str) or not text.strip():
            raise EmbeddingInputError("Text must be a nonempty string.")
        if len(text) > MAX_EMBEDDING_TEXT_CHARACTERS:
            raise EmbeddingInputError(
                "Text must not exceed 8,000 characters."
            )
        if (
            not OLLAMA_EMBEDDING_MODEL.strip()
            or EMBEDDING_DIMENSIONS <= 0
            or not math.isfinite(EMBEDDING_TIMEOUT_SECONDS)
            or EMBEDDING_TIMEOUT_SECONDS <= 0
        ):
            raise EmbeddingServiceError("Invalid embedding configuration.")

        try:
            parsed_url = urlsplit(OLLAMA_BASE_URL)
            if (
                parsed_url.scheme not in {"http", "https"}
                or not parsed_url.hostname
            ):
                raise EmbeddingServiceError("Invalid embedding configuration.")
            request = Request(
                OLLAMA_BASE_URL.rstrip("/") + "/api/embed",
                data=json.dumps({
                    "model": OLLAMA_EMBEDDING_MODEL,
                    "input": text,
                    "truncate": False,
                }).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(request, timeout=EMBEDDING_TIMEOUT_SECONDS) as response:
                if not 200 <= response.status < 300:
                    raise EmbeddingServiceError(
                        "Embedding provider rejected the request."
                    )
                body = response.read(MAX_RESPONSE_BYTES + 1)
        except HTTPError:
            raise EmbeddingServiceError(
                "Embedding provider rejected the request."
            ) from None
        except (OSError, HTTPException):
            raise EmbeddingServiceError(
                "Embedding provider is unavailable or the request timed out."
            ) from None
        except ValueError:
            raise EmbeddingServiceError(
                "Invalid embedding request configuration."
            ) from None

        if len(body) > MAX_RESPONSE_BYTES:
            raise EmbeddingServiceError("Embedding response exceeds size limit.")
        try:
            payload = json.loads(body)
        except (ValueError, UnicodeError):
            raise EmbeddingServiceError(
                "Embedding provider returned an invalid response."
            ) from None

        embeddings = payload.get("embeddings") if isinstance(payload, dict) else None
        if (
            not isinstance(embeddings, list)
            or len(embeddings) != 1
            or not isinstance(embeddings[0], list)
            or len(embeddings[0]) != EMBEDDING_DIMENSIONS
        ):
            raise EmbeddingServiceError(
                "Embedding response has an invalid shape or dimensions."
            )

        vector: list[float] = []
        for value in embeddings[0]:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise EmbeddingServiceError(
                    "Embedding response contains invalid numeric values."
                )
            try:
                number = float(value)
            except OverflowError:
                raise EmbeddingServiceError(
                    "Embedding response contains invalid numeric values."
                ) from None
            if not math.isfinite(number):
                raise EmbeddingServiceError(
                    "Embedding response contains invalid numeric values."
                )
            vector.append(number)
        return vector
