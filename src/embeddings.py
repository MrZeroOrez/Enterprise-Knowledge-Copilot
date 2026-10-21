"""Gemini embeddings module using the official google-genai SDK.

Generates dense vector embeddings for document chunks and user queries using
the configured Gemini embedding model (e.g. text-embedding-004).
Includes input validation, finite numeric verification, bounded exponential backoff
for transient rate limits, and zero network calls at import time.
"""

from __future__ import annotations

import logging
import math
import random
import time
from typing import Any

from google import genai
from google.genai import types
from google.genai.errors import APIError

from src.config import load_config

logger = logging.getLogger(__name__)

# Conservative bounds for free-tier resilience
DEFAULT_MAX_RETRIES = 3
DEFAULT_INITIAL_RETRY_DELAY = 1.0
DEFAULT_MAX_RETRY_DELAY = 10.0
DEFAULT_BATCH_SIZE = 16


# ==============================================================================
# Custom Exceptions
# ==============================================================================

class EmbeddingError(Exception):
    """Base exception for all embedding generation errors."""


class EmbeddingRateLimitError(EmbeddingError):
    """Raised when Gemini embedding API rate limits or quotas are exceeded."""


# ==============================================================================
# Client Management
# ==============================================================================

def get_genai_client(api_key: str | None = None) -> genai.Client:
    """Create a validated Gemini API client.

    Args:
        api_key: Optional explicit API key. If omitted, loads from config.

    Returns:
        Instantiated google.genai.Client.

    Raises:
        ValueError: If GEMINI_API_KEY is missing or set to placeholder.
    """
    if api_key is not None:
        if not api_key.strip():
            raise ValueError(
                "GEMINI_API_KEY is not set. Please obtain an API key from "
                "Google AI Studio (https://aistudio.google.com/) and set it in your .env file."
            )
        if api_key.strip() == "your_gemini_api_key_here":
            raise ValueError(
                "GEMINI_API_KEY is still set to the default placeholder value. "
                "Please replace it with your actual Gemini API key in your .env file."
            )
        resolved_key = api_key
    else:
        config = load_config()
        config.validate_api_key()
        resolved_key = config.gemini_api_key

    return genai.Client(api_key=resolved_key)


# ==============================================================================
# Vector Validation
# ==============================================================================

def validate_embedding_vector(vector: list[float], input_label: str = "text") -> None:
    """Validate that the returned embedding vector is non-empty and strictly finite.

    Args:
        vector: List of floating-point values from embedding response.
        input_label: Label for contextual error reporting.

    Raises:
        EmbeddingError: If vector is empty or contains non-numeric / NaN / Inf values.
    """
    if not vector or not isinstance(vector, list):
        raise EmbeddingError(f"Embedding API returned empty or non-list vector for {input_label}.")

    for idx, val in enumerate(vector):
        if not isinstance(val, (int, float)):
            raise EmbeddingError(
                f"Embedding vector contains invalid non-numeric value at index {idx}: {type(val).__name__}"
            )
        if math.isnan(val) or math.isinf(val):
            raise EmbeddingError(
                f"Embedding vector contains non-finite value (NaN or Inf) at index {idx}."
            )


# ==============================================================================
# Core Embedding Functions with Retry
# ==============================================================================

def _execute_with_retry(
    client: genai.Client,
    model: str,
    contents: str | list[str],
    config: types.EmbedContentConfig,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> Any:
    """Execute embed_content call with bounded exponential backoff and jitter."""
    delay = DEFAULT_INITIAL_RETRY_DELAY

    for attempt in range(1, max_retries + 1):
        try:
            return client.models.embed_content(
                model=model,
                contents=contents,
                config=config,
            )
        except APIError as exc:
            # 429 = ResourceExhausted (rate limit or quota), 503 = Service Unavailable
            status_code = getattr(exc, "code", None)
            is_rate_limit = status_code == 429 or "RESOURCE_EXHAUSTED" in str(exc)
            is_transient = status_code in (429, 500, 503, 504) or is_rate_limit

            if is_transient and attempt < max_retries:
                # Exponential backoff with jitter
                jitter = random.uniform(0.1, 0.5)
                sleep_time = min(delay + jitter, DEFAULT_MAX_RETRY_DELAY)
                logger.warning(
                    "Embedding API transient error (attempt %d/%d). Retrying in %.2fs. Code: %s",
                    attempt,
                    max_retries,
                    sleep_time,
                    status_code,
                )
                time.sleep(sleep_time)
                delay *= 2.0
            elif is_rate_limit:
                raise EmbeddingRateLimitError(
                    f"Gemini embedding API rate limit or quota exceeded: {exc}"
                ) from exc
            else:
                raise EmbeddingError(f"Gemini embedding API call failed: {exc}") from exc
        except Exception as exc:
            raise EmbeddingError(f"Unexpected error calling Gemini embedding API: {exc}") from exc

    raise EmbeddingError(f"Embedding request failed after {max_retries} retries.")


def embed_text(
    text: str,
    task_type: str = "RETRIEVAL_DOCUMENT",
    title: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    client: genai.Client | None = None,
) -> list[float]:
    """Generate an embedding vector for a single text input.

    Args:
        text: Text string to embed (must be non-empty and non-whitespace).
        task_type: Gemini task type ('RETRIEVAL_DOCUMENT' or 'RETRIEVAL_QUERY').
        title: Optional document title for document embeddings.
        api_key: Optional explicit API key.
        model: Optional explicit embedding model name (defaults to config).
        client: Optional pre-instantiated genai.Client (useful for unit testing).

    Returns:
        List of float values representing the embedding vector.

    Raises:
        ValueError: If input text is empty or invalid.
        EmbeddingError: If API call fails or returned vector is invalid.
        EmbeddingRateLimitError: If API quota or rate limit is exhausted.
    """
    if not isinstance(text, str):
        raise ValueError(f"Input text must be a string, got: {type(text).__name__}")

    if not text.strip():
        raise ValueError("Cannot generate embedding for empty or whitespace-only text.")

    active_client = client or get_genai_client(api_key=api_key)
    active_model = model or load_config().embedding_model

    config_kwargs: dict[str, Any] = {"task_type": task_type}
    if title:
        config_kwargs["title"] = title

    embed_config = types.EmbedContentConfig(**config_kwargs)

    response = _execute_with_retry(
        client=active_client,
        model=active_model,
        contents=text,
        config=embed_config,
    )

    if not response or not hasattr(response, "embeddings") or not response.embeddings:
        raise EmbeddingError("Embedding API response did not contain any embeddings.")

    vector = response.embeddings[0].values
    validate_embedding_vector(vector, input_label="text chunk")

    return list(vector)


def embed_query(
    query: str,
    api_key: str | None = None,
    model: str | None = None,
    client: genai.Client | None = None,
) -> list[float]:
    """Generate an embedding vector for a search query using RETRIEVAL_QUERY task type.

    Guarantees compatibility with vectors indexed using RETRIEVAL_DOCUMENT.

    Args:
        query: User search question or query string.
        api_key: Optional explicit API key.
        model: Optional explicit embedding model name.
        client: Optional pre-instantiated client.

    Returns:
        Query embedding vector as list of floats.
    """
    if not isinstance(query, str):
        raise ValueError(f"Query must be a string, got: {type(query).__name__}")

    if not query.strip():
        raise ValueError("Query string cannot be empty or whitespace.")

    return embed_text(
        text=query,
        task_type="RETRIEVAL_QUERY",
        api_key=api_key,
        model=model,
        client=client,
    )


def embed_texts(
    texts: list[str],
    task_type: str = "RETRIEVAL_DOCUMENT",
    batch_size: int = DEFAULT_BATCH_SIZE,
    api_key: str | None = None,
    model: str | None = None,
    client: genai.Client | None = None,
) -> list[list[float]]:
    """Generate embeddings for multiple texts using conservative batching.

    Args:
        texts: List of non-empty text strings to embed.
        task_type: Task type for embeddings ('RETRIEVAL_DOCUMENT' or 'RETRIEVAL_QUERY').
        batch_size: Maximum texts per API batch call (default: 16).
        api_key: Optional explicit API key.
        model: Optional explicit model name.
        client: Optional pre-instantiated client.

    Returns:
        List of embedding vectors, matching input text order.

    Raises:
        ValueError: If texts list is empty or any element is empty/non-string.
        EmbeddingError: If API call fails or validation fails.
    """
    if not texts:
        return []

    for idx, t in enumerate(texts):
        if not isinstance(t, str):
            raise ValueError(f"Text item at index {idx} must be a string, got: {type(t).__name__}")
        if not t.strip():
            raise ValueError(f"Text item at index {idx} cannot be empty or whitespace.")

    active_client = client or get_genai_client(api_key=api_key)
    active_model = model or load_config().embedding_model
    embed_config = types.EmbedContentConfig(task_type=task_type)

    all_embeddings: list[list[float]] = []

    # Process in bounded batches to respect API limits
    for batch_start in range(0, len(texts), batch_size):
        batch = texts[batch_start : batch_start + batch_size]

        response = _execute_with_retry(
            client=active_client,
            model=active_model,
            contents=batch,
            config=embed_config,
        )

        if not response or not hasattr(response, "embeddings") or not response.embeddings:
            raise EmbeddingError("Embedding API response did not contain embeddings for batch.")

        if len(response.embeddings) != len(batch):
            raise EmbeddingError(
                f"Embedding count mismatch: expected {len(batch)} vectors, "
                f"got {len(response.embeddings)}."
            )

        for idx, emb in enumerate(response.embeddings):
            vec = list(emb.values)
            validate_embedding_vector(vec, input_label=f"batch item {batch_start + idx}")
            all_embeddings.append(vec)

    return all_embeddings
