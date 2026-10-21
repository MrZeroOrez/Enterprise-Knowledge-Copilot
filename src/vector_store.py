"""ChromaDB vector store module for Enterprise Knowledge Copilot.

Manages persistent local ChromaDB storage, explicit external embeddings
(using embedding_function=None), metadata retention (source, page, chunk_id),
scoped document replacements, and vector similarity search.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection

from src.chunking import DocumentChunk
from src.config import load_config

logger = logging.getLogger(__name__)

# Cache of persistent clients keyed by resolved path to avoid multiple handles
_CLIENT_CACHE: dict[str, chromadb.PersistentClient] = {}


def get_chroma_client(persist_dir: Path | str | None = None) -> chromadb.PersistentClient:
    """Retrieve or create a persistent ChromaDB client for the configured path.

    Reuses existing client instances for the same path to prevent locking or duplicate handles.

    Args:
        persist_dir: Directory for database files. Defaults to config.chroma_persist_dir.

    Returns:
        chromadb.PersistentClient instance.
    """
    if persist_dir is None:
        target_dir = load_config().chroma_persist_dir
    else:
        target_dir = Path(persist_dir)

    resolved_path = str(target_dir.resolve())
    target_dir.mkdir(parents=True, exist_ok=True)

    if resolved_path not in _CLIENT_CACHE:
        _CLIENT_CACHE[resolved_path] = chromadb.PersistentClient(path=resolved_path)

    return _CLIENT_CACHE[resolved_path]


def get_or_create_collection(
    collection_name: str | None = None,
    persist_dir: Path | str | None = None,
) -> Collection:
    """Get an existing ChromaDB collection or create a new one with embedding_function=None.

    Setting embedding_function=None ensures ChromaDB relies exclusively on external
    Gemini embeddings and never attempts to download or run local embedding models.

    Args:
        collection_name: Target collection name. Defaults to config.chroma_collection_name.
        persist_dir: Storage directory path.

    Returns:
        ChromaDB Collection object.
    """
    config = load_config()
    name = collection_name or config.chroma_collection_name
    client = get_chroma_client(persist_dir=persist_dir)

    return client.get_or_create_collection(
        name=name,
        embedding_function=None,
        metadata={"hnsw:space": "cosine"},
    )


def sanitize_metadata(meta: dict[str, Any]) -> dict[str, str | int | float | bool]:
    """Ensure metadata contains only primitive types supported by ChromaDB.

    ChromaDB supports string, int, float, bool.
    """
    sanitized: dict[str, str | int | float | bool] = {}
    for k, v in meta.items():
        if isinstance(v, (str, int, float, bool)):
            sanitized[k] = v
        elif v is None:
            sanitized[k] = ""
        else:
            sanitized[k] = str(v)
    return sanitized


def add_chunks_to_vector_store(
    chunks: list[DocumentChunk],
    embeddings: list[list[float]],
    doc_hash: str = "",
    collection_name: str | None = None,
    persist_dir: Path | str | None = None,
) -> int:
    """Add document chunks and their precomputed embeddings into ChromaDB.

    Args:
        chunks: List of DocumentChunk objects.
        embeddings: Parallel list of embedding vectors.
        doc_hash: Optional document content hash for change detection.
        collection_name: Optional target collection name.
        persist_dir: Optional custom database directory.

    Returns:
        Number of chunks added.

    Raises:
        ValueError: If chunks and embeddings lengths do not match or are empty.
    """
    if not chunks:
        return 0

    if len(chunks) != len(embeddings):
        raise ValueError(
            f"Chunk count ({len(chunks)}) and embedding count ({len(embeddings)}) must match."
        )

    collection = get_or_create_collection(
        collection_name=collection_name,
        persist_dir=persist_dir,
    )

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, str | int | float | bool]] = []

    for chunk, emb in zip(chunks, embeddings, strict=True):
        ids.append(chunk.chunk_id)
        documents.append(chunk.text)

        meta = {
            "source": chunk.source,
            "source_filename": chunk.source_filename,
            "page": chunk.page,
            "page_number": chunk.page_number,
            "chunk_id": chunk.chunk_id,
            "chunk_index": chunk.chunk_index,
            "char_count": chunk.char_count,
        }
        if doc_hash:
            meta["doc_hash"] = doc_hash

        # Include additional user metadata if present
        if chunk.metadata:
            for k, v in chunk.metadata.items():
                if k not in meta:
                    meta[k] = v

        metadatas.append(sanitize_metadata(meta))

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )

    return len(ids)


def get_chunks_by_source(
    source: str,
    collection_name: str | None = None,
    persist_dir: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Retrieve all stored chunks for a given source document.

    Args:
        source: Document filename to filter by.
        collection_name: Target collection.
        persist_dir: Database directory.

    Returns:
        List of dictionaries with 'id', 'document', 'metadata', 'embedding'.
    """
    collection = get_or_create_collection(
        collection_name=collection_name,
        persist_dir=persist_dir,
    )

    results = collection.get(
        where={"source": source},
        include=["documents", "metadatas"],
    )

    records: list[dict[str, Any]] = []
    ids = results.get("ids", [])
    docs = results.get("documents", []) or []
    metas = results.get("metadatas", []) or []

    for chunk_id, doc, meta in zip(ids, docs, metas, strict=False):
        records.append({
            "chunk_id": chunk_id,
            "text": doc,
            "metadata": meta,
        })

    return records


def delete_chunks_by_source(
    source: str,
    collection_name: str | None = None,
    persist_dir: Path | str | None = None,
) -> int:
    """Delete all indexed chunks belonging to a specific source document.

    Leaves all chunks from other documents completely untouched.

    Args:
        source: Document filename whose chunks should be deleted.
        collection_name: Target collection.
        persist_dir: Database directory.

    Returns:
        Number of deleted chunks.
    """
    collection = get_or_create_collection(
        collection_name=collection_name,
        persist_dir=persist_dir,
    )

    existing = get_chunks_by_source(
        source=source,
        collection_name=collection_name,
        persist_dir=persist_dir,
    )
    count = len(existing)

    if count > 0:
        collection.delete(where={"source": source})

    return count


def get_collection_stats(
    collection_name: str | None = None,
    persist_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Retrieve summary statistics for the collection.

    Returns:
        Dictionary with 'total_chunks', 'distinct_sources', 'collection_name'.
    """
    collection = get_or_create_collection(
        collection_name=collection_name,
        persist_dir=persist_dir,
    )

    total_chunks = collection.count()
    distinct_sources: set[str] = set()

    if total_chunks > 0:
        results = collection.get(include=["metadatas"])
        for meta in results.get("metadatas", []) or []:
            if meta and "source" in meta:
                distinct_sources.add(str(meta["source"]))

    return {
        "collection_name": collection.name,
        "total_chunks": total_chunks,
        "distinct_sources": sorted(distinct_sources),
        "source_count": len(distinct_sources),
    }


def list_indexed_documents(
    collection_name: str | None = None,
    persist_dir: Path | str | None = None,
) -> list[dict[str, Any]]:
    """List all indexed documents in the collection with their chunk counts.

    Args:
        collection_name: Target collection name.
        persist_dir: Database directory.

    Returns:
        List of dicts: [{'source': 'doc.pdf', 'chunk_count': 5}, ...]
    """
    collection = get_or_create_collection(
        collection_name=collection_name,
        persist_dir=persist_dir,
    )

    total_chunks = collection.count()
    if total_chunks == 0:
        return []

    from collections import Counter

    counts: Counter[str] = Counter()
    results = collection.get(include=["metadatas"])
    for meta in results.get("metadatas", []) or []:
        if meta and "source" in meta:
            counts[str(meta["source"])] += 1

    return [
        {"source": src, "chunk_count": count}
        for src, count in sorted(counts.items())
    ]


def query_vector_store(
    query_embedding: list[float],
    top_k: int = 5,
    where: dict[str, Any] | None = None,
    collection_name: str | None = None,
    persist_dir: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Query the ChromaDB collection using a precomputed embedding vector.

    Args:
        query_embedding: Embedding vector for the search query.
        top_k: Number of most relevant chunks to return (must be > 0).
        where: Optional metadata filter dict.
        collection_name: Target collection.
        persist_dir: Database directory.

    Returns:
        List of matched chunk dictionaries, each containing:
        'chunk_id', 'text', 'source', 'page', 'distance', 'metadata'.
    """
    if top_k <= 0:
        raise ValueError(f"top_k must be a positive integer (> 0), got: {top_k}")

    collection = get_or_create_collection(
        collection_name=collection_name,
        persist_dir=persist_dir,
    )

    total = collection.count()
    if total == 0:
        return []

    # Bounded n_results cannot exceed collection size
    n_results = min(top_k, total)

    query_kwargs: dict[str, Any] = {
        "query_embeddings": [query_embedding],
        "n_results": n_results,
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        query_kwargs["where"] = where

    res = collection.query(**query_kwargs)

    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    distances = res.get("distances", [[]])[0]

    hits: list[dict[str, Any]] = []
    for chunk_id, doc, meta, dist in zip(ids, docs, metas, distances, strict=False):
        meta_dict = meta or {}
        hits.append({
            "chunk_id": chunk_id,
            "text": doc,
            "source": meta_dict.get("source", ""),
            "page": meta_dict.get("page", 1),
            "distance": dist,
            "metadata": meta_dict,
        })

    return hits


def reset_collection(
    collection_name: str | None = None,
    persist_dir: Path | str | None = None,
) -> None:
    """Delete and recreate the collection. Primarily used for testing or index rebuilding."""
    config = load_config()
    name = collection_name or config.chroma_collection_name
    client = get_chroma_client(persist_dir=persist_dir)

    try:
        client.delete_collection(name=name)
    except Exception:
        pass

    client.create_collection(
        name=name,
        embedding_function=None,
        metadata={"hnsw:space": "cosine"},
    )
