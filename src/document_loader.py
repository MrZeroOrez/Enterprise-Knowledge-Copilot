"""Document loader module for validating, saving, and extracting text from PDF files.

Uses pypdf to read PDF files page-by-page, preserving 1-indexed page numbers,
original filenames, and citation-critical metadata for downstream RAG pipelines.
Includes safe storage, size and signature validation, and detection of scanned documents.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
import re
from typing import BinaryIO
import uuid

from pypdf import PdfReader
from pypdf.errors import PdfReadError

logger = logging.getLogger(__name__)

# Default maximum upload size: 10 MB
DEFAULT_MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024
DEFAULT_UPLOAD_DIR = "data/uploads"

# PDF Magic Bytes Signature (%PDF-)
PDF_MAGIC_BYTES = b"%PDF-"


# ==============================================================================
# Custom Exception Hierarchy
# ==============================================================================

class DocumentProcessingError(Exception):
    """Base exception for all document processing and extraction errors."""


class FileValidationError(DocumentProcessingError, ValueError):
    """Raised when a file fails validation (size, signature, empty, extension, path traversal)."""


class EncryptedPdfError(DocumentProcessingError, ValueError):
    """Raised when a PDF file is encrypted or password-protected."""


class ScannedPdfError(DocumentProcessingError):
    """Raised when a PDF contains no extractable text (e.g., scanned/image-only document)."""


class CorruptPdfError(DocumentProcessingError, RuntimeError):
    """Raised when a PDF file structure is corrupted or unreadable."""


# ==============================================================================
# Data Structures
# ==============================================================================

class DocumentPage:
    """Represents the extracted text and metadata from a single PDF page."""

    def __init__(
        self,
        text: str,
        page_number: int,
        source_filename: str,
        metadata: dict | None = None,
    ) -> None:
        self.text = text
        self.page_number = page_number  # 1-indexed for human-friendly citations
        self.source_filename = source_filename
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
        """Return the number of characters in the page text."""
        return len(self.text)

    @property
    def is_empty(self) -> bool:
        """Check if the extracted text is empty or only whitespace."""
        return not self.text.strip()

    def to_dict(self) -> dict:
        """Export page data as a standard dictionary."""
        return {
            "source": self.source_filename,
            "page": self.page_number,
            "text": self.text,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        snippet = self.text[:40].replace("\n", " ") + "..." if len(self.text) > 40 else self.text
        return (
            f"DocumentPage(page={self.page_number}, source='{self.source_filename}', "
            f"chars={self.char_count}, preview='{snippet}')"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DocumentPage):
            return False
        return (
            self.text == other.text
            and self.page_number == other.page_number
            and self.source_filename == other.source_filename
            and self.metadata == other.metadata
        )


# ==============================================================================
# Text Cleaning and Normalization
# ==============================================================================

def normalize_whitespace(text: str) -> str:
    """Normalize excessive whitespace without destroying paragraph boundaries.

    - Strips null characters
    - Normalizes carriage returns (\\r\\n -> \\n, \\r -> \\n)
    - Collapses consecutive spaces/tabs on the same line to a single space
    - Collapses 3 or more consecutive newlines to 2 newlines (retaining paragraph breaks)
    - Trims leading and trailing whitespace
    """
    if not text:
        return ""

    # Clean null characters and standardise newlines
    cleaned = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")

    # Clean horizontal whitespace per line
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in cleaned.split("\n")]
    result = "\n".join(lines)

    # Collapse excessive vertical whitespace (3+ newlines -> 2 newlines)
    result = re.sub(r"\n{3,}", "\n\n", result)

    return result.strip()


# ==============================================================================
# Validation and Security
# ==============================================================================

def sanitize_filename(filename: str) -> str:
    """Sanitize user-provided filename to prevent path traversal and special characters.

    Args:
        filename: User-provided filename.

    Returns:
        Safe base filename strictly ending in .pdf.
    """
    stripped = filename.strip()
    base = Path(stripped).name.strip()

    # Fallback if filename is empty or only extension
    if not base or base.lower() == ".pdf":
        return "document.pdf"

    # Strip .pdf extension if present to sanitize stem
    if base.lower().endswith(".pdf"):
        stem = base[:-4]
    else:
        stem = base

    clean_stem = re.sub(r"[^a-zA-Z0-9_.-]", "_", stem)
    if not clean_stem.strip("._"):
        clean_stem = "document"

    return f"{clean_stem}.pdf"


def validate_pdf_bytes(
    file_bytes: bytes,
    original_filename: str = "",
    max_size_bytes: int = DEFAULT_MAX_UPLOAD_SIZE_BYTES,
) -> None:
    """Validate raw PDF bytes for size, signature, encryption, and basic readability.

    Args:
        file_bytes: Raw binary content of the file.
        original_filename: Document filename for error reporting.
        max_size_bytes: Configurable maximum allowed size in bytes.

    Raises:
        FileValidationError: If file is empty, oversized, lacks PDF signature, or has no pages.
        EncryptedPdfError: If file is password-protected or encrypted.
        CorruptPdfError: If file structure is corrupt or unreadable.
    """
    name = original_filename or "document.pdf"

    # 1. Non-empty check
    if not file_bytes or len(file_bytes) == 0:
        raise FileValidationError(f"PDF file '{name}' is completely empty (0 bytes).")

    # 2. Maximum size check
    if len(file_bytes) > max_size_bytes:
        size_mb = len(file_bytes) / (1024 * 1024)
        limit_mb = max_size_bytes / (1024 * 1024)
        raise FileValidationError(
            f"File '{name}' ({size_mb:.2f} MB) exceeds maximum allowed size of {limit_mb:.1f} MB."
        )

    # 3. PDF Signature check (%PDF- magic bytes within the first 1024 bytes)
    header_sample = file_bytes[:1024]
    if PDF_MAGIC_BYTES not in header_sample:
        raise FileValidationError(
            f"Invalid file signature: '{name}' is not a valid PDF file (missing %PDF- header)."
        )

    # 4. Attempt to parse using pypdf
    try:
        reader = PdfReader(io.BytesIO(file_bytes))

        # Check encryption
        if reader.is_encrypted:
            raise EncryptedPdfError(
                f"PDF file '{name}' is password-protected or encrypted. "
                f"Encrypted documents are not supported."
            )

        # Check page count
        if len(reader.pages) == 0:
            raise FileValidationError(f"PDF file '{name}' contains 0 pages.")

    except (FileValidationError, EncryptedPdfError):
        raise
    except PdfReadError as exc:
        raise CorruptPdfError(f"Corrupt or unreadable PDF file '{name}': {exc}") from exc
    except Exception as exc:
        raise CorruptPdfError(f"Unexpected error reading PDF file '{name}': {exc}") from exc


# ==============================================================================
# Safe Storage
# ==============================================================================

def save_uploaded_pdf(
    file_data: bytes | BinaryIO,
    original_filename: str,
    upload_dir: Path | str | None = None,
    max_size_bytes: int = DEFAULT_MAX_UPLOAD_SIZE_BYTES,
) -> tuple[Path, str]:
    """Validate and safely persist an uploaded PDF file to the uploads directory.

    Args:
        file_data: Raw bytes or file-like object (e.g. io.BytesIO or Streamlit UploadedFile).
        original_filename: User-provided filename.
        upload_dir: Destination directory. Defaults to configured upload_dir.
        max_size_bytes: Maximum allowed file size in bytes.

    Returns:
        Tuple of (saved_file_path, sanitized_original_filename).

    Raises:
        FileValidationError: If validation fails or path traversal is detected.
        EncryptedPdfError: If PDF is encrypted.
        CorruptPdfError: If PDF is corrupt.
    """
    # Read bytes from file-like object or raw bytes
    if isinstance(file_data, (bytes, bytearray)):
        content_bytes = bytes(file_data)
    elif hasattr(file_data, "read"):
        content_bytes = file_data.read()
        if hasattr(file_data, "seek"):
            file_data.seek(0)
    else:
        raise FileValidationError(f"Unsupported file data type: {type(file_data).__name__}")

    # Validate content
    validate_pdf_bytes(
        file_bytes=content_bytes,
        original_filename=original_filename,
        max_size_bytes=max_size_bytes,
    )

    # Target directory resolution
    target_dir = Path(upload_dir or DEFAULT_UPLOAD_DIR).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    # Sanitize filename and generate unique, collision-free storage name
    clean_name = sanitize_filename(original_filename)
    safe_storage_name = f"{uuid.uuid4().hex[:12]}_{clean_name}"
    dest_path = (target_dir / safe_storage_name).resolve()

    # Prevent path traversal attacks
    if not dest_path.is_relative_to(target_dir):
        raise FileValidationError("Path traversal attempt detected in uploaded filename.")

    # Prevent accidental overwrites
    if dest_path.exists():
        raise FileValidationError(f"Target file '{safe_storage_name}' already exists.")

    # Write file safely
    dest_path.write_bytes(content_bytes)

    return dest_path, clean_name


# ==============================================================================
# Text Extraction
# ==============================================================================

def check_for_scanned_document(pages: list[DocumentPage], filename: str = "") -> None:
    """Verify that the document contains extractable text, raising ScannedPdfError if not.

    Args:
        pages: List of extracted DocumentPage objects.
        filename: Document filename for reporting.

    Raises:
        ScannedPdfError: If no extractable text is present across all pages.
    """
    total_chars = sum(len(p.text.strip()) for p in pages)
    if total_chars == 0:
        doc_name = filename or "document"
        raise ScannedPdfError(
            f"No extractable text found in '{doc_name}'. "
            f"The document appears to be scanned or image-only. "
            f"OCR is not implemented yet."
        )


def extract_text_from_pdf(
    source: str | Path | bytes | BinaryIO,
    original_filename: str = "",
    max_size_bytes: int = DEFAULT_MAX_UPLOAD_SIZE_BYTES,
    require_text: bool = False,
) -> list[DocumentPage]:
    """Extract page-by-page text from a PDF path, bytes, or file-like object.

    Args:
        source: File path, raw bytes, or binary stream.
        original_filename: Original filename to store in citation metadata.
        max_size_bytes: Maximum allowed file size in bytes.
        require_text: If True, raises ScannedPdfError if the document has 0 extractable characters.

    Returns:
        List of DocumentPage objects, one for each page in the PDF.

    Raises:
        FileNotFoundError: If a provided file path does not exist.
        FileValidationError: If the file is invalid, empty, or oversized.
        EncryptedPdfError: If the PDF is password-protected or encrypted.
        CorruptPdfError: If the PDF is malformed or unreadable.
        ScannedPdfError: If require_text=True and no extractable text was found.
    """
    source_name = original_filename

    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found at: '{path.resolve()}'")
        if not path.is_file():
            raise FileValidationError(f"Path is a directory or special file: '{path.resolve()}'")
        if path.suffix.lower() != ".pdf":
            raise FileValidationError(f"Invalid file type. Expected a .pdf file, but got: '{path.name}'")
        content_bytes = path.read_bytes()
        if not source_name:
            source_name = path.name
    elif isinstance(source, (bytes, bytearray)):
        content_bytes = bytes(source)
        if not source_name:
            source_name = "document.pdf"
    elif hasattr(source, "read"):
        content_bytes = source.read()
        if hasattr(source, "seek"):
            source.seek(0)
        if not source_name:
            source_name = getattr(source, "name", "document.pdf")
    else:
        raise FileValidationError(f"Unsupported source type: {type(source).__name__}")

    # Validate PDF content bytes
    validate_pdf_bytes(
        file_bytes=content_bytes,
        original_filename=source_name,
        max_size_bytes=max_size_bytes,
    )

    pages: list[DocumentPage] = []

    try:
        reader = PdfReader(io.BytesIO(content_bytes))
        total_pages = len(reader.pages)

        for page_idx, page in enumerate(reader.pages, start=1):
            raw_text = page.extract_text() or ""
            cleaned_text = normalize_whitespace(raw_text)

            doc_page = DocumentPage(
                text=cleaned_text,
                page_number=page_idx,
                source_filename=source_name,
                metadata={
                    "source_path": str(source) if isinstance(source, (str, Path)) else "in-memory",
                    "total_pages": total_pages,
                    "char_count": len(cleaned_text),
                    "is_scanned_or_blank": len(cleaned_text) == 0,
                },
            )
            pages.append(doc_page)

    except PdfReadError as exc:
        raise CorruptPdfError(f"Corrupt or unreadable PDF file '{source_name}': {exc}") from exc
    except Exception as exc:
        raise CorruptPdfError(f"Unexpected error reading PDF file '{source_name}': {exc}") from exc

    if require_text:
        check_for_scanned_document(pages, filename=source_name)

    return pages


def load_pdf(file_path: str | Path, require_text: bool = False) -> list[DocumentPage]:
    """Load and extract text page-by-page from a PDF file path.

    Maintains full backward compatibility with Phase 1.
    """
    return extract_text_from_pdf(source=file_path, require_text=require_text)


def load_pdfs_from_directory(directory_path: str | Path) -> list[DocumentPage]:
    """Load all PDF files from a specified directory.

    Args:
        directory_path: Directory containing PDF files.

    Returns:
        List of DocumentPage objects from all discovered PDFs.

    Raises:
        FileNotFoundError: If the directory does not exist.
        NotADirectoryError: If the path is not a directory.
    """
    dir_p = Path(directory_path)

    if not dir_p.exists():
        raise FileNotFoundError(f"Directory not found: '{dir_p.resolve()}'")

    if not dir_p.is_dir():
        raise NotADirectoryError(f"Path is not a directory: '{dir_p.resolve()}'")

    pdf_files = sorted(dir_p.glob("*.pdf"))
    all_pages: list[DocumentPage] = []

    for pdf_file in pdf_files:
        pages = load_pdf(pdf_file)
        all_pages.extend(pages)

    return all_pages
