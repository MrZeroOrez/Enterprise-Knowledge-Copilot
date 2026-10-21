"""Comprehensive unit tests for the PDF document loader module (Phase 2)."""

import io
from pathlib import Path

from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
import pytest

from src.document_loader import (
    CorruptPdfError,
    DocumentPage,
    EncryptedPdfError,
    FileValidationError,
    ScannedPdfError,
    extract_text_from_pdf,
    load_pdf,
    load_pdfs_from_directory,
    normalize_whitespace,
    sanitize_filename,
    save_uploaded_pdf,
    validate_pdf_bytes,
)


# ==============================================================================
# Helper to build in-memory synthetic PDFs for offline testing
# ==============================================================================

def make_synthetic_pdf(
    pages_text: list[str],
    password: str | None = None,
) -> bytes:
    """Generate in-memory PDF bytes with real extractable text using standard Type 1 fonts."""
    writer = PdfWriter()

    for text in pages_text:
        page = writer.add_blank_page(width=300, height=300)
        if text.strip():
            # Inject standard PDF text stream
            stream = DecodedStreamObject()
            escaped_text = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            stream.set_data(f"BT /F1 12 Tf 50 250 Td ({escaped_text}) Tj ET".encode("latin-1"))
            page[NameObject("/Contents")] = stream

            # Set Helvetica standard font in Resources
            font_dict = DictionaryObject()
            f1 = DictionaryObject()
            f1[NameObject("/Type")] = NameObject("/Font")
            f1[NameObject("/Subtype")] = NameObject("/Type1")
            f1[NameObject("/BaseFont")] = NameObject("/Helvetica")
            font_dict[NameObject("/F1")] = f1

            resources = DictionaryObject()
            resources[NameObject("/Font")] = font_dict
            page[NameObject("/Resources")] = resources

    if password:
        writer.encrypt(password)

    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


# ==============================================================================
# Test Cases
# ==============================================================================

class TestPdfValidationAndExtraction:
    """Tests for PDF validation, signature checks, extraction, and error handling."""

    def test_valid_pdf_with_extractable_text(self, tmp_path: Path):
        """Should extract text and attach correct metadata from a valid single-page PDF."""
        pdf_bytes = make_synthetic_pdf(["Enterprise Architecture Overview"])
        file_path = tmp_path / "handbook.pdf"
        file_path.write_bytes(pdf_bytes)

        pages = load_pdf(file_path)

        assert len(pages) == 1
        page = pages[0]
        assert isinstance(page, DocumentPage)
        assert page.page_number == 1
        assert page.page == 1
        assert page.source_filename == "handbook.pdf"
        assert page.source == "handbook.pdf"
        assert "Enterprise Architecture Overview" in page.text
        assert not page.is_empty
        assert page.char_count > 0

    def test_pdf_with_multiple_pages(self, tmp_path: Path):
        """Should correctly extract sequential pages and assign 1-based page numbers."""
        page_contents = [
            "Executive Summary",
            "Technical Requirements",
            "Security and Compliance",
        ]
        pdf_bytes = make_synthetic_pdf(page_contents)
        file_path = tmp_path / "system_spec.pdf"
        file_path.write_bytes(pdf_bytes)

        pages = load_pdf(file_path)

        assert len(pages) == 3
        for idx, expected_text in enumerate(page_contents, start=1):
            page = pages[idx - 1]
            assert page.page_number == idx
            assert page.page == idx
            assert expected_text in page.text
            assert page.source_filename == "system_spec.pdf"

    def test_page_containing_no_extractable_text(self, tmp_path: Path):
        """Pages without text should be represented as empty pages without crashing."""
        pdf_bytes = make_synthetic_pdf(["Page 1 has text", "", "Page 3 has text"])
        file_path = tmp_path / "mixed.pdf"
        file_path.write_bytes(pdf_bytes)

        pages = load_pdf(file_path)

        assert len(pages) == 3
        assert not pages[0].is_empty
        assert pages[1].is_empty
        assert pages[1].char_count == 0
        assert pages[1].metadata["is_scanned_or_blank"] is True
        assert not pages[2].is_empty

    def test_scanned_pdf_reports_ocr_not_implemented(self):
        """A PDF with zero extractable text should raise ScannedPdfError noting OCR is not implemented."""
        # 2 blank pages (simulating image-only/scanned documents without OCR)
        pdf_bytes = make_synthetic_pdf(["", ""])

        with pytest.raises(ScannedPdfError) as exc_info:
            extract_text_from_pdf(pdf_bytes, original_filename="scanned_receipt.pdf", require_text=True)

        err_msg = str(exc_info.value)
        assert "scanned_receipt.pdf" in err_msg
        assert "scanned or image-only" in err_msg
        assert "OCR is not implemented yet" in err_msg

    def test_empty_zero_byte_input(self):
        """Zero-byte file or empty bytes must be rejected as FileValidationError."""
        with pytest.raises(FileValidationError, match="completely empty"):
            validate_pdf_bytes(b"", original_filename="zero.pdf")

        with pytest.raises(FileValidationError, match="completely empty"):
            extract_text_from_pdf(b"", original_filename="zero.pdf")

    def test_file_larger_than_configured_limit(self):
        """Files exceeding the configured byte limit must be rejected."""
        dummy_large_bytes = b"%PDF-1.4 " + b"0" * 1024
        # Configure a small 500-byte limit to test without allocating 10MB
        with pytest.raises(FileValidationError, match="exceeds maximum allowed size"):
            validate_pdf_bytes(dummy_large_bytes, max_size_bytes=500)

    def test_file_with_incorrect_signature(self):
        """Files lacking the %PDF- magic bytes signature must be rejected."""
        fake_pdf = b"GIF89a this is an image, not a PDF"
        with pytest.raises(FileValidationError, match="Invalid file signature"):
            validate_pdf_bytes(fake_pdf, original_filename="avatar.pdf")

    def test_malformed_pdf_corrupt(self):
        """Corrupt PDF bytes with valid signature but broken syntax must raise CorruptPdfError."""
        corrupt_bytes = b"%PDF-1.4\nBROKEN_OBJECT_DATA_CORRUPTED_TRAILER"
        with pytest.raises(CorruptPdfError, match="Corrupt or unreadable PDF file"):
            extract_text_from_pdf(corrupt_bytes, original_filename="broken.pdf")

    def test_encrypted_pdf_handling(self):
        """Encrypted or password-protected PDFs must raise EncryptedPdfError."""
        encrypted_bytes = make_synthetic_pdf(["Secret Financial Data"], password="supersecretpassword")

        with pytest.raises(EncryptedPdfError, match="password-protected or encrypted"):
            extract_text_from_pdf(encrypted_bytes, original_filename="confidential.pdf")

    def test_accepts_file_like_bytesio(self):
        """Should accept file-like objects (e.g. io.BytesIO or Streamlit UploadedFile)."""
        pdf_bytes = make_synthetic_pdf(["Streamlit Upload Test"])
        stream = io.BytesIO(pdf_bytes)

        pages = extract_text_from_pdf(stream, original_filename="upload.pdf")
        assert len(pages) == 1
        assert "Streamlit Upload Test" in pages[0].text


class TestSafeStorage:
    """Tests for safe PDF saving and path traversal defense."""

    def test_safe_storage_success(self, tmp_path: Path):
        """Valid PDF should be safely written to upload directory with unique filename."""
        pdf_bytes = make_synthetic_pdf(["Safe storage content"])
        upload_dir = tmp_path / "uploads"

        dest_path, original_clean = save_uploaded_pdf(
            file_data=pdf_bytes,
            original_filename="company policy.pdf",
            upload_dir=upload_dir,
        )

        assert dest_path.exists()
        assert dest_path.parent == upload_dir.resolve()
        assert original_clean == "company_policy.pdf"
        assert dest_path.read_bytes() == pdf_bytes

    def test_path_traversal_prevention(self, tmp_path: Path):
        """Malicious paths with ../ should be sanitized and confined strictly to upload_dir."""
        pdf_bytes = make_synthetic_pdf(["Security test"])
        upload_dir = tmp_path / "uploads"

        dest_path, clean_name = save_uploaded_pdf(
            file_data=pdf_bytes,
            original_filename="../../etc/passwd.pdf",
            upload_dir=upload_dir,
        )

        assert dest_path.exists()
        assert dest_path.parent == upload_dir.resolve()
        assert ".." not in str(dest_path)
        assert clean_name == "passwd.pdf"

    def test_filename_sanitization(self):
        """Sanitizer should clean dangerous characters while preserving .pdf."""
        assert sanitize_filename("report.pdf") == "report.pdf"
        assert sanitize_filename("my report (v1).pdf") == "my_report__v1_.pdf"
        assert sanitize_filename("..\\..\\evil.pdf") == "evil.pdf"
        assert sanitize_filename("doc") == "doc.pdf"
        assert sanitize_filename("   .pdf   ") == "document.pdf"


class TestTextNormalizationAndDataStructures:
    """Tests for whitespace normalization and DocumentPage dictionary export."""

    def test_whitespace_normalization(self):
        """Should collapse excessive whitespace while preserving paragraph breaks."""
        raw = "Line 1 with    multiple    spaces.\r\n\r\n\r\n\r\nLine 2 after four newlines.\x00"
        normalized = normalize_whitespace(raw)

        # 4 newlines should collapse to 2 newlines (a single paragraph break)
        assert normalized == "Line 1 with multiple spaces.\n\nLine 2 after four newlines."

    def test_document_page_dictionary_export(self):
        """DocumentPage.to_dict should export standard fields for RAG pipelines."""
        page = DocumentPage(
            text="Employee benefits include 401k match.",
            page_number=5,
            source_filename="handbook.pdf",
            metadata={"department": "HR"},
        )

        data = page.to_dict()
        assert data["source"] == "handbook.pdf"
        assert data["page"] == 5
        assert data["text"] == "Employee benefits include 401k match."
        assert data["metadata"]["department"] == "HR"

    def test_directory_loading(self, tmp_path: Path):
        """load_pdfs_from_directory should aggregate pages across all valid PDFs."""
        pdf1 = tmp_path / "doc1.pdf"
        pdf2 = tmp_path / "doc2.pdf"
        pdf1.write_bytes(make_synthetic_pdf(["Doc 1 text"]))
        pdf2.write_bytes(make_synthetic_pdf(["Doc 2 text"]))

        pages = load_pdfs_from_directory(tmp_path)
        assert len(pages) == 2
        sources = {p.source_filename for p in pages}
        assert sources == {"doc1.pdf", "doc2.pdf"}
