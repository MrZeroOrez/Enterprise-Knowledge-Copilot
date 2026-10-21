"""Unit tests for the PDF document loader module."""

from pathlib import Path
from pypdf import PdfWriter
import pytest
from src.document_loader import (
    DocumentPage,
    load_pdf,
    load_pdfs_from_directory,
)


class TestLoadPdf:
    """Tests for individual PDF file loading and error handling."""

    def test_file_not_found(self, tmp_path: Path):
        """Should raise FileNotFoundError when the target PDF does not exist."""
        non_existent = tmp_path / "missing_document.pdf"
        with pytest.raises(FileNotFoundError, match="PDF file not found"):
            load_pdf(non_existent)

    def test_invalid_extension(self, tmp_path: Path):
        """Should raise ValueError when the file is not a PDF."""
        text_file = tmp_path / "notes.txt"
        text_file.write_text("plain text contents", encoding="utf-8")

        with pytest.raises(ValueError, match="Invalid file type. Expected a .pdf file"):
            load_pdf(text_file)

    def test_directory_passed_as_file(self, tmp_path: Path):
        """Should raise ValueError when a directory path is passed."""
        sub_dir = tmp_path / "some_folder"
        sub_dir.mkdir()

        with pytest.raises(ValueError, match="Path is a directory or special file"):
            load_pdf(sub_dir)

    def test_empty_zero_byte_file(self, tmp_path: Path):
        """Should raise ValueError when a 0-byte PDF is encountered."""
        empty_pdf = tmp_path / "empty.pdf"
        empty_pdf.touch()

        with pytest.raises(ValueError, match="PDF file is completely empty"):
            load_pdf(empty_pdf)

    def test_corrupt_pdf_file(self, tmp_path: Path):
        """Should raise RuntimeError when a file has a .pdf extension but corrupt binary data."""
        corrupt_pdf = tmp_path / "corrupt.pdf"
        corrupt_pdf.write_bytes(b"NOT_A_VALID_PDF_HEADER_OR_BODY_12345")

        with pytest.raises(RuntimeError, match="Corrupt or unreadable PDF file"):
            load_pdf(corrupt_pdf)

    def test_valid_pdf_blank_page(self, tmp_path: Path):
        """Should successfully parse a valid PDF and return DocumentPage objects."""
        pdf_path = tmp_path / "blank_sample.pdf"
        writer = PdfWriter()
        writer.add_blank_page(width=100, height=100)
        with open(pdf_path, "wb") as f:
            writer.write(f)

        pages = load_pdf(pdf_path)

        assert len(pages) == 1
        page = pages[0]
        assert isinstance(page, DocumentPage)
        assert page.page_number == 1
        assert page.source_filename == "blank_sample.pdf"
        assert page.is_empty is True
        assert page.char_count == 0
        assert page.metadata["total_pages"] == 1


class TestLoadPdfsFromDirectory:
    """Tests for directory-level PDF loading."""

    def test_directory_not_found(self, tmp_path: Path):
        """Should raise FileNotFoundError when directory does not exist."""
        non_existent_dir = tmp_path / "non_existent_dir"
        with pytest.raises(FileNotFoundError, match="Directory not found"):
            load_pdfs_from_directory(non_existent_dir)

    def test_path_is_not_directory(self, tmp_path: Path):
        """Should raise NotADirectoryError when path points to a file."""
        sample_file = tmp_path / "sample.txt"
        sample_file.write_text("not a folder", encoding="utf-8")

        with pytest.raises(NotADirectoryError, match="Path is not a directory"):
            load_pdfs_from_directory(sample_file)

    def test_empty_directory_returns_empty_list(self, tmp_path: Path):
        """Should return an empty list when directory contains no PDF files."""
        empty_dir = tmp_path / "empty_dir"
        empty_dir.mkdir()

        result = load_pdfs_from_directory(empty_dir)
        assert result == []

    def test_multiple_pdfs_loaded(self, tmp_path: Path):
        """Should discover and load all PDFs in the directory."""
        pdf_dir = tmp_path / "pdf_folder"
        pdf_dir.mkdir()

        for name in ["doc_a.pdf", "doc_b.pdf"]:
            writer = PdfWriter()
            writer.add_blank_page(width=100, height=100)
            with open(pdf_dir / name, "wb") as f:
                writer.write(f)

        pages = load_pdfs_from_directory(pdf_dir)
        assert len(pages) == 2
        filenames = {p.source_filename for p in pages}
        assert filenames == {"doc_a.pdf", "doc_b.pdf"}
