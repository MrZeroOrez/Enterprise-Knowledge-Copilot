"""Semantic retrieval and ranking module for Enterprise Knowledge Copilot.

Queries ChromaDB using dense vector embeddings generated via Gemini's official
google-genai SDK (task_type='RETRIEVAL_QUERY'). Ranks hits by cosine distance,
preserves citation metadata (source, page, chunk_id), supports source filtering,
configurable relevance thresholding, and prepares context for LLM grounding (Phase 5).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.config import AppConfig, load_config
from src.embeddings import embed_query
from src.vector_store import query_vector_store

logger = logging.getLogger(__name__)


def search_chunks(
    query: str,
    top_k: int | None = None,
    source_filter: str | None = None,
    score_threshold: float | None = None,
    max_distance: float | None = None,
    where: dict[str, Any] | None = None,
    collection_name: str | None = None,
    persist_dir: Path | str | None = None,
    config: AppConfig | None = None,
    embedding_client: Any | None = None,
    client: Any | None = None,
) -> list[dict[str, Any]]:
    """Retrieve the most relevant document chunks for a natural-language query.

    Flow:
        Query string -> validation -> Gemini query embedding (RETRIEVAL_QUERY)
        -> ChromaDB vector search (cosine space) -> source filtering
        -> optional distance thresholding -> deduplication -> ranked results.

    Distance Metric:
        ChromaDB uses cosine distance: distance = 1.0 - cosine_similarity.
        - 0.0 indicates identical directional orientation (maximum similarity).
        - 1.0 indicates orthogonal vectors (unrelated).
        - Smaller distance values represent higher relevance.

    Args:
        query: Natural-language search question or query.
        top_k: Number of chunks to return (defaults to RETRIEVAL_TOP_K).
        source_filter: Optional document filename to restrict retrieval to a single PDF.
        score_threshold: Alias for max_distance.
        max_distance: Maximum allowable cosine distance (0.0 to 2.0). Chunks with
                      distance > max_distance are excluded. None disables filtering.
        where: Optional custom ChromaDB metadata filter dictionary.
        collection_name: Optional target collection name.
        persist_dir: Optional custom database persistence path.
        config: Optional loaded AppConfig.
        embedding_client: Optional pre-instantiated Gemini client (for unit testing).
        client: Backward-compatible alias for embedding_client.

    Returns:
        List of ranked chunk dictionaries:
        [
            {
                "chunk_id": "handbook_p3_c0_abc123",
                "text": "Extracted document passage...",
                "source": "handbook.pdf",
                "page": 3,
                "distance": 0.182,
                "metadata": {...}
            },
            ...
        ]

    Raises:
        ValueError: If query is empty, top_k is invalid, or filters are malformed.
        EmbeddingError: If the Gemini embedding API call fails.
    """
    # 1. Validate query string
    if not isinstance(query, str):
        raise ValueError(f"Query must be a string, got: {type(query).__name__}")

    cleaned_query = query.strip()
    if not cleaned_query:
        raise ValueError("Query string cannot be empty or whitespace.")

    app_config = config or load_config()

    # 2. Resolve and validate top_k
    resolved_top_k = top_k if top_k is not None else app_config.retrieval_top_k
    if not isinstance(resolved_top_k, int) or isinstance(resolved_top_k, bool) or resolved_top_k <= 0:
        raise ValueError(f"top_k must be a positive integer (> 0), got: {resolved_top_k}")

    if resolved_top_k > app_config.retrieval_max_top_k:
        raise ValueError(
            f"top_k ({resolved_top_k}) exceeds maximum allowed limit ({app_config.retrieval_max_top_k})."
        )

    # 3. Resolve and validate distance threshold
    # Priority: explicit max_distance -> explicit score_threshold -> config -> None
    effective_max_distance = (
        max_distance
        if max_distance is not None
        else (score_threshold if score_threshold is not None else app_config.retrieval_distance_threshold)
    )

    if effective_max_distance is not None:
        if not isinstance(effective_max_distance, (int, float)) or not (0.0 <= effective_max_distance <= 2.0):
            raise ValueError(
                f"max_distance / score_threshold must be a float between 0.0 and 2.0, "
                f"got: {effective_max_distance}"
            )

    # 4. Construct metadata filter
    combined_where: dict[str, Any] | None = None
    if source_filter is not None:
        if not isinstance(source_filter, str) or not source_filter.strip():
            raise ValueError("source_filter must be a non-empty string.")
        source_clean = source_filter.strip()
        if where:
            combined_where = {"$and": [where, {"source": source_clean}]}
        else:
            combined_where = {"source": source_clean}
    elif where:
        combined_where = where

    # 5. Generate query embedding with task_type='RETRIEVAL_QUERY'
    active_client = embedding_client or client
    query_vector = embed_query(
        query=cleaned_query,
        api_key=app_config.gemini_api_key,
        model=app_config.embedding_model,
        client=active_client,
    )

    # 6. Query ChromaDB collection
    raw_hits = query_vector_store(
        query_embedding=query_vector,
        top_k=resolved_top_k,
        where=combined_where,
        collection_name=collection_name or app_config.chroma_collection_name,
        persist_dir=persist_dir or app_config.chroma_persist_dir,
    )

    if not raw_hits:
        return []

    # 7. Deduplicate and apply distance thresholding
    seen_chunk_ids: set[str] = set()
    filtered_hits: list[dict[str, Any]] = []

    for hit in raw_hits:
        chunk_id = hit.get("chunk_id", "")
        if chunk_id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(chunk_id)

        dist = hit.get("distance", 1.0)
        if effective_max_distance is not None and dist > effective_max_distance:
            continue

        filtered_hits.append(hit)

    return filtered_hits


def format_retrieval_context(hits: list[dict[str, Any]]) -> str:
    """Format retrieved document chunks into a structured context string for Phase 5 LLM prompting.

    Args:
        hits: List of retrieved chunk dictionaries from search_chunks.

    Returns:
        Structured string with source citations and page numbers.
    """
    if not hits:
        return "No relevant document passages found."

    sections: list[str] = []
    for idx, hit in enumerate(hits, start=1):
        source = hit.get("source", "unknown")
        page = hit.get("page", 1)
        chunk_id = hit.get("chunk_id", "")
        text = hit.get("text", "").strip()

        header = f"[Passage {idx} | Source: {source} | Page: {page} | ID: {chunk_id}]"
        sections.append(f"{header}\n{text}")

    return "\n\n---\n\n".join(sections)
