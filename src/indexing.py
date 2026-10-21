"""Document indexing pipeline module for Enterprise Knowledge Copilot.

Coordinates Phase 2 PDF extraction and chunking with Phase 3 Gemini embeddings
and ChromaDB persistence. Includes change detection, deduplication, safe all-or-nothing
replacements, and robust error handling.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import logging
from pathlib import Path
from typing import Any, BinaryIO

from src.chunking import DocumentChunk, chunk_pages
from src.config import AppConfig, load_config
from src.document_loader import (
    DocumentPage,
    extract_text_from_pdf,
)
from src.embeddings import embed_texts
from src.vector_store import (
    add_chunks_to_vector_store,
    delete_chunks_by_source,
    get_chunks_by_source,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IndexingResult:
    """Represents the execution outcome of indexing a document."""

    source: str
    pages_processed: int
    chunks_created: int
    chunks_indexed: int
    status: str  # 'indexed', 'skipped_unchanged', 'replaced', 'empty'
    message: str

    @property
    def is_success(self) -> bool:
        """Return True if indexing succeeded or was safely skipped as unchanged."""
        return self.status in ("indexed", "skipped_unchanged", "replaced")

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary representation."""
        return {
            "source": self.source,
            "pages_processed": self.pages_processed,
            "chunks_created": self.chunks_created,
            "chunks_indexed": self.chunks_indexed,
            "status": self.status,
            "message": self.message,
            "is_success": self.is_success,
        }


def compute_document_hash(chunks: list[DocumentChunk]) -> str:
    """Compute a deterministic hash across all chunks of a document to detect content changes."""
    hasher = hashlib.sha256()
    for chunk in chunks:
        hasher.update(chunk.text.encode("utf-8"))
    return hasher.hexdigest()


def index_document(
    file_source: str | Path | bytes | BinaryIO,
    original_filename: str = "",
    chunk_size: int | None = None,
    overlap: int | None = None,
    force_reindex: bool = False,
    collection_name: str | None = None,
    persist_dir: Path | str | None = None,
    config: AppConfig | None = None,
    embedding_client: Any | None = None,
) -> IndexingResult:
    """Extract, chunk, embed, and index a PDF document into ChromaDB.

    Pipeline Flow:
        PDF -> Phase 2 validation & text extraction -> page-bounded chunking
        -> change detection / deduplication -> all-or-nothing Gemini embedding
        -> atomic ChromaDB replacement / insertion.

    Args:
        file_source: PDF file path, raw bytes, or file-like stream.
        original_filename: Document filename for citations and tracking.
        chunk_size: Optional character chunk size override.
        overlap: Optional character overlap override.
        force_reindex: If True, forces re-embedding even if content is unchanged.
        collection_name: Optional target collection name.
        persist_dir: Optional custom database persistence path.
        config: Optional loaded AppConfig.
        embedding_client: Optional pre-instantiated Gemini client (for unit testing).

    Returns:
        IndexingResult detailing pages processed and chunks indexed.

    Raises:
        FileValidationError: If PDF validation fails.
        EncryptedPdfError: If PDF is encrypted.
        CorruptPdfError: If PDF is corrupt.
        EmbeddingError: If embedding generation fails.
    """
    app_config = config or load_config()
    c_size = chunk_size or app_config.chunk_size
    c_overlap = overlap or app_config.chunk_overlap

    # 1. Resolve source name
    source_name = original_filename
    if not source_name and isinstance(file_source, (str, Path)):
        source_name = Path(file_source).name
    if not source_name:
        source_name = "document.pdf"

    # 2. Extract pages using Phase 2 loader
    pages: list[DocumentPage] = extract_text_from_pdf(
        source=file_source,
        original_filename=source_name,
        max_size_bytes=app_config.max_upload_size_bytes,
        require_text=False,
    )

    # 3. Check for empty or non-text document
    total_chars = sum(p.char_count for p in pages)
    if total_chars == 0:
        return IndexingResult(
            source=source_name,
            pages_processed=len(pages),
            chunks_created=0,
            chunks_indexed=0,
            status="empty",
            message=f"Document '{source_name}' contains no extractable text. No chunks indexed.",
        )

    # 4. Generate page-bounded chunks
    chunks: list[DocumentChunk] = chunk_pages(
        pages=pages,
        chunk_size=c_size,
        overlap=c_overlap,
        deduplicate=False,
    )

    if not chunks:
        return IndexingResult(
            source=source_name,
            pages_processed=len(pages),
            chunks_created=0,
            chunks_indexed=0,
            status="empty",
            message=f"Document '{source_name}' produced 0 chunks after chunking.",
        )

    # 5. Duplicate change detection
    doc_hash = compute_document_hash(chunks)
    existing_records = get_chunks_by_source(
        source=source_name,
        collection_name=collection_name,
        persist_dir=persist_dir,
    )

    if existing_records and not force_reindex:
        first_meta = existing_records[0].get("metadata", {})
        existing_hash = first_meta.get("doc_hash")
        if existing_hash == doc_hash and len(existing_records) == len(chunks):
            return IndexingResult(
                source=source_name,
                pages_processed=len(pages),
                chunks_created=len(chunks),
                chunks_indexed=len(existing_records),
                status="skipped_unchanged",
                message=f"Document '{source_name}' is unchanged. Duplicate indexing safely skipped.",
            )

    # 6. All-or-nothing embedding generation
    # We generate ALL embeddings before modifying the database. If embedding fails
    # halfway, the database remains completely untouched, preserving existing index!
    chunk_texts = [chunk.text for chunk in chunks]
    embeddings = embed_texts(
        texts=chunk_texts,
        task_type="RETRIEVAL_DOCUMENT",
        api_key=app_config.gemini_api_key,
        model=app_config.embedding_model,
        client=embedding_client,
    )

    # 7. Safe replacement: remove old version of this document only
    is_replacement = len(existing_records) > 0
    if is_replacement:
        delete_chunks_by_source(
            source=source_name,
            collection_name=collection_name,
            persist_dir=persist_dir,
        )

    # 8. Index new chunks with embeddings and metadata into ChromaDB
    indexed_count = add_chunks_to_vector_store(
        chunks=chunks,
        embeddings=embeddings,
        doc_hash=doc_hash,
        collection_name=collection_name,
        persist_dir=persist_dir,
    )

    status = "replaced" if is_replacement else "indexed"
    action_text = "replaced and re-indexed" if is_replacement else "successfully indexed"

    return IndexingResult(
        source=source_name,
        pages_processed=len(pages),
        chunks_created=len(chunks),
        chunks_indexed=indexed_count,
        status=status,
        message=f"Document '{source_name}' ({indexed_count} chunks) {action_text}.",
    )
