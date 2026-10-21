"""Deterministic text-chunking module for Enterprise Knowledge Copilot.

Splits document pages into discrete, overlapping chunks while strictly preserving
page boundaries, citation metadata (page number, source filename), and stable IDs
for grounded RAG retrieval and citation.
"""

from __future__ import annotations

import hashlib
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.document_loader import DocumentPage


class DocumentChunk:
    """Represents a discrete text chunk extracted from a document with citation metadata."""

    def __init__(
        self,
        chunk_id: str,
        text: str,
        page_number: int,
        source_filename: str,
        chunk_index: int,
        metadata: dict | None = None,
    ) -> None:
        self.chunk_id = chunk_id
        self.text = text
        self.page_number = page_number
        self.source_filename = source_filename
        self.chunk_index = chunk_index
        self.metadata = metadata or {}

    @property
    def source(self) -> str:
        """Alias for source_filename."""
        return self.source_filename

    @property
    def page(self) -> int:
        """Alias for page_number."""
        return self.page_number

    @property
    def char_count(self) -> int:
        """Return the character count of this chunk."""
        return len(self.text)

    def to_dict(self) -> dict:
        """Export chunk data as a dictionary compatible with downstream RAG pipelines."""
        return {
            "chunk_id": self.chunk_id,
            "source": self.source_filename,
            "page": self.page_number,
            "text": self.text,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        snippet = self.text[:40].replace("\n", " ") + "..." if len(self.text) > 40 else self.text
        return (
            f"DocumentChunk(id='{self.chunk_id}', page={self.page_number}, "
            f"source='{self.source_filename}', chars={self.char_count}, preview='{snippet}')"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DocumentChunk):
            return False
        return (
            self.chunk_id == other.chunk_id
            and self.text == other.text
            and self.page_number == other.page_number
            and self.source_filename == other.source_filename
            and self.chunk_index == other.chunk_index
            and self.metadata == other.metadata
        )


def validate_chunk_params(chunk_size: int, chunk_overlap: int) -> None:
    """Validate that chunking parameters satisfy RAG invariants.

    Args:
        chunk_size: Maximum character length of each chunk. Must be > 0.
        chunk_overlap: Overlapping characters between consecutive chunks.
                       Must be >= 0 and strictly < chunk_size.

    Raises:
        TypeError: If parameters are not integers.
        ValueError: If parameters violate numerical invariants.
    """
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool):
        raise TypeError(f"chunk_size must be an integer, got: {type(chunk_size).__name__}")
    if not isinstance(chunk_overlap, int) or isinstance(chunk_overlap, bool):
        raise TypeError(f"chunk_overlap must be an integer, got: {type(chunk_overlap).__name__}")

    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be a positive integer (> 0), got: {chunk_size}")

    if chunk_overlap < 0:
        raise ValueError(f"chunk_overlap must be non-negative (>= 0), got: {chunk_overlap}")

    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be strictly less than chunk_size ({chunk_size})"
        )


def compute_stable_chunk_id(source_filename: str, page_number: int, chunk_index: int, text: str) -> str:
    """Generate a deterministic, stable unique identifier for a document chunk.

    Format: <safe_name>_p<page>_c<index>_<hash>
    Avoids embedding sensitive document contents in identifier strings.
    """
    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:10]
    safe_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", source_filename) if source_filename else "doc"
    return f"{safe_name}_p{page_number}_c{chunk_index}_{content_hash}"


# Backwards compatibility alias
_compute_chunk_id = compute_stable_chunk_id


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    page_number: int = 1,
    source_filename: str = "",
) -> list[DocumentChunk]:
    """Deterministically split raw text into overlapping DocumentChunk objects.

    Guaranteed to terminate without infinite loops.
    Omits empty or whitespace-only chunks.

    Args:
        text: The raw text string to split.
        chunk_size: Maximum character count per chunk.
        chunk_overlap: Number of characters to overlap between consecutive chunks.
        page_number: Originating page number (1-indexed).
        source_filename: Originating document filename.

    Returns:
        List of DocumentChunk instances. Returns empty list if text is empty or whitespace.

    Raises:
        ValueError: If chunk_size or chunk_overlap violate invariants.
    """
    validate_chunk_params(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    if not text or not text.strip():
        return []

    # Normalize carriage returns
    normalized_text = text.replace("\r\n", "\n").replace("\r", "\n")
    text_length = len(normalized_text)

    # Step size is guaranteed to be >= 1 because chunk_overlap < chunk_size
    step_size = chunk_size - chunk_overlap

    chunks: list[DocumentChunk] = []
    start_idx = 0
    chunk_idx = 0

    while start_idx < text_length:
        end_idx = min(start_idx + chunk_size, text_length)
        chunk_content = normalized_text[start_idx:end_idx]

        # Only create chunk if it contains non-whitespace content
        if chunk_content.strip():
            chunk_id = compute_stable_chunk_id(
                source_filename=source_filename,
                page_number=page_number,
                chunk_index=chunk_idx,
                text=chunk_content,
            )

            chunk = DocumentChunk(
                chunk_id=chunk_id,
                text=chunk_content,
                page_number=page_number,
                source_filename=source_filename,
                chunk_index=chunk_idx,
                metadata={
                    "source": source_filename,
                    "source_filename": source_filename,
                    "page": page_number,
                    "page_number": page_number,
                    "chunk_index": chunk_idx,
                    "char_start": start_idx,
                    "char_end": end_idx,
                    "char_count": len(chunk_content),
                },
            )
            chunks.append(chunk)
            chunk_idx += 1

        # Break when end of text is reached
        if end_idx >= text_length:
            break

        start_idx += step_size

    return chunks


def chunk_pages(
    pages: list[DocumentPage],
    chunk_size: int = 1000,
    overlap: int = 150,
    chunk_overlap: int | None = None,
    deduplicate: bool = False,
) -> list[DocumentChunk]:
    """Chunk extracted document pages while preserving page boundaries and citation metadata.

    A single chunk is strictly confined to a single page; it never crosses page boundaries,
    ensuring unambiguous citation in downstream RAG generation.

    Args:
        pages: List of DocumentPage objects from document_loader.
        chunk_size: Maximum character count per chunk (default: 1000).
        overlap: Overlap in characters between consecutive chunks (default: 150).
        chunk_overlap: Optional alias for overlap for backward compatibility.
        deduplicate: When True, prevents embedding exact duplicate chunk contents.

    Returns:
        List of DocumentChunk objects across all provided pages.
    """
    effective_overlap = chunk_overlap if chunk_overlap is not None else overlap
    validate_chunk_params(chunk_size=chunk_size, chunk_overlap=effective_overlap)

    all_chunks: list[DocumentChunk] = []
    seen_hashes: set[str] = set()

    for page in pages:
        # Skip empty or whitespace-only pages
        if page.is_empty:
            continue

        page_chunks = chunk_text(
            text=page.text,
            chunk_size=chunk_size,
            chunk_overlap=effective_overlap,
            page_number=page.page_number,
            source_filename=page.source_filename,
        )

        for chunk in page_chunks:
            if deduplicate:
                chunk_hash = hashlib.sha256(chunk.text.strip().encode("utf-8")).hexdigest()
                if chunk_hash in seen_hashes:
                    continue
                seen_hashes.add(chunk_hash)

            all_chunks.append(chunk)

    return all_chunks


# Backward-compatible alias
chunk_document_pages = chunk_pages
