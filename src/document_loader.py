"""Document loader module for extracting text and metadata from PDF files.

Uses pypdf to read PDF files page-by-page, preserving page numbers, filenames,
and basic document metadata for accurate citations in RAG.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from pypdf import PdfReader
from pypdf.errors import PdfReadError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DocumentPage:
    """Represents the extracted text and metadata from a single PDF page."""

    text: str
    page_number: int  # 1-indexed for human-friendly citations
    source_filename: str
    metadata: dict = field(default_factory=dict)

    @property
    def char_count(self) -> int:
        """Return the number of characters in the page text."""
        return len(self.text)

    @property
    def is_empty(self) -> bool:
        """Check if the extracted text is empty or only whitespace."""
        return not self.text.strip()


def load_pdf(file_path: str | Path) -> list[DocumentPage]:
    """Load and extract text page-by-page from a PDF file.

    Args:
        file_path: Path to the PDF file to load.

    Returns:
        List of DocumentPage objects, one for each page in the PDF.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a PDF or is 0 bytes.
        RuntimeError: If pypdf fails to parse the PDF file.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF file not found at: '{path.resolve()}'")

    if not path.is_file():
        raise ValueError(f"Path is a directory or special file, not a regular file: '{path.resolve()}'")

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Invalid file type. Expected a .pdf file, but got: '{path.name}'")

    if path.stat().st_size == 0:
        raise ValueError(f"PDF file is completely empty (0 bytes): '{path.name}'")

    pages: list[DocumentPage] = []

    try:
        reader = PdfReader(str(path))
        total_pages = len(reader.pages)

        for page_idx, page in enumerate(reader.pages, start=1):
            extracted = page.extract_text() or ""
            # Clean up null characters or carriage returns while preserving newlines
            cleaned_text = extracted.replace("\x00", "").replace("\r\n", "\n")

            doc_page = DocumentPage(
                text=cleaned_text,
                page_number=page_idx,
                source_filename=path.name,
                metadata={
                    "source_path": str(path.resolve()),
                    "total_pages": total_pages,
                    "char_count": len(cleaned_text),
                },
            )
            pages.append(doc_page)

    except PdfReadError as exc:
        raise RuntimeError(f"Corrupt or unreadable PDF file '{path.name}': {exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"Unexpected error reading PDF file '{path.name}': {exc}") from exc

    return pages


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
