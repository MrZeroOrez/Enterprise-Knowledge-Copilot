"""Deterministic text-chunking module for Enterprise Knowledge Copilot.

Splits document text into overlapping chunks with preserved metadata (page number,
source filename, character offsets) for grounded RAG retrieval and citation.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.document_loader import DocumentPage


@dataclass(frozen=True)
class DocumentChunk:
    """Represents a discrete text chunk extracted from a document with citation metadata."""

    chunk_id: str
    text: str
    page_number: int
    source_filename: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)

    @property
    def char_count(self) -> int:
        """Return the character count of this chunk."""
        return len(self.text)


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


def _compute_chunk_id(source_filename: str, page_number: int, chunk_index: int, text: str) -> str:
    """Generate a deterministic, unique identifier for a document chunk."""
    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:10]
    safe_name = source_filename.replace(" ", "_") if source_filename else "doc"
    return f"{safe_name}_p{page_number}_c{chunk_index}_{content_hash}"


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    page_number: int = 1,
    source_filename: str = "",
) -> list[DocumentChunk]:
    """Deterministically split raw text into overlapping DocumentChunk objects.

    Args:
        text: The raw text string to split.
        chunk_size: Maximum character count per chunk.
        chunk_overlap: Number of characters to overlap between consecutive chunks.
        page_number: Page number from which this text originated (for citations).
        source_filename: Originating document filename (for citations).

    Returns:
        List of DocumentChunk instances. Returns empty list if text is empty or whitespace.

    Raises:
        ValueError: If chunk_size or chunk_overlap violate invariants.
    """
    validate_chunk_params(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    if not text or not text.strip():
        return []

    # Clean text: normalize carriage returns
    normalized_text = text.replace("\r\n", "\n")
    text_length = len(normalized_text)

    # Step size is the distance the window moves forward on each step
    step_size = chunk_size - chunk_overlap

    chunks: list[DocumentChunk] = []
    start_idx = 0
    chunk_idx = 0

    while start_idx < text_length:
        end_idx = min(start_idx + chunk_size, text_length)
        chunk_content = normalized_text[start_idx:end_idx]

        # Only create chunk if it contains non-whitespace content
        if chunk_content.strip():
            chunk_id = _compute_chunk_id(
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
                    "source_filename": source_filename,
                    "page_number": page_number,
                    "chunk_index": chunk_idx,
                    "char_start": start_idx,
                    "char_end": end_idx,
                    "char_count": len(chunk_content),
                },
            )
            chunks.append(chunk)
            chunk_idx += 1

        # If this chunk reached or exceeded the end of the text, stop
        if end_idx >= text_length:
            break

        start_idx += step_size

    return chunks


def chunk_document_pages(
    pages: list[DocumentPage],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    deduplicate: bool = False,
) -> list[DocumentChunk]:
    """Chunk a sequence of DocumentPages (e.g. from load_pdf), preserving page citations.

    Args:
        pages: List of DocumentPage objects from document_loader.
        chunk_size: Character length of each chunk.
        chunk_overlap: Overlapping characters between consecutive chunks.
        deduplicate: When True, prevents embedding exact duplicate chunk contents.

    Returns:
        List of DocumentChunk objects across all provided pages.
    """
    validate_chunk_params(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    all_chunks: list[DocumentChunk] = []
    seen_hashes: set[str] = set()

    for page in pages:
        if page.is_empty:
            continue

        page_chunks = chunk_text(
            text=page.text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
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
