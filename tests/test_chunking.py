"""Comprehensive unit tests for the chunking module (Phase 2)."""

import pytest

from src.chunking import (
    DocumentChunk,
    chunk_document_pages,
    chunk_pages,
    chunk_text,
    validate_chunk_params,
)
from src.document_loader import DocumentPage


class TestValidateChunkParams:
    """Tests for chunk parameter validation invariants."""

    def test_valid_params(self):
        """Should pass without error for valid chunk_size and chunk_overlap."""
        validate_chunk_params(chunk_size=1000, chunk_overlap=200)
        validate_chunk_params(chunk_size=100, chunk_overlap=0)
        validate_chunk_params(chunk_size=1, chunk_overlap=0)

    @pytest.mark.parametrize("invalid_size", [0, -1, -500])
    def test_invalid_chunk_size(self, invalid_size: int):
        """Should raise ValueError when chunk_size is 0 or negative."""
        with pytest.raises(ValueError, match="chunk_size must be a positive integer"):
            validate_chunk_params(chunk_size=invalid_size, chunk_overlap=10)

    def test_negative_overlap(self):
        """Should raise ValueError when chunk_overlap is negative."""
        with pytest.raises(ValueError, match="chunk_overlap must be non-negative"):
            validate_chunk_params(chunk_size=500, chunk_overlap=-1)

    @pytest.mark.parametrize(
        ("size", "overlap"),
        [(500, 500), (500, 600)],
    )
    def test_overlap_greater_or_equal_to_size(self, size: int, overlap: int):
        """Should raise ValueError when overlap >= chunk_size."""
        with pytest.raises(ValueError, match="chunk_overlap .* must be strictly less than"):
            validate_chunk_params(chunk_size=size, chunk_overlap=overlap)

    @pytest.mark.parametrize(
        ("size", "overlap"),
        [("500", 100), (500, "100"), (True, 50), (500, False)],
    )
    def test_non_integer_params(self, size, overlap):
        """Should raise TypeError when parameters are not pure integers."""
        with pytest.raises(TypeError):
            validate_chunk_params(chunk_size=size, chunk_overlap=overlap)


class TestChunkText:
    """Tests for chunk_text splitting behavior."""

    @pytest.mark.parametrize("empty_input", ["", "   ", "\n\n\t  \n", "    \r\n  "])
    def test_empty_or_whitespace_input(self, empty_input: str):
        """Empty or whitespace-only text should return an empty list of chunks."""
        chunks = chunk_text(empty_input, chunk_size=100, chunk_overlap=20)
        assert chunks == []

    def test_text_shorter_than_chunk_size(self):
        """Text shorter than chunk_size should produce exactly one chunk."""
        text = "Hello, world! This is a short enterprise document test."
        chunks = chunk_text(text, chunk_size=200, chunk_overlap=50)

        assert len(chunks) == 1
        assert chunks[0].text == text
        assert chunks[0].chunk_index == 0
        assert chunks[0].char_count == len(text)

    def test_text_equal_to_chunk_size(self):
        """Text exactly equal to chunk_size should produce exactly one chunk."""
        text = "A" * 100
        chunks = chunk_text(text, chunk_size=100, chunk_overlap=20)

        assert len(chunks) == 1
        assert chunks[0].text == text
        assert chunks[0].char_count == 100

    def test_sliding_window_overlap_and_continuity(self):
        """Sequential chunks should slide forward with exact configured overlap."""
        # Alphabet length 10: "abcdefghij"
        # chunk_size = 5, chunk_overlap = 2, step = 3
        # Chunk 0: index 0..5 -> "abcde"
        # Chunk 1: index 3..8 -> "defgh"
        # Chunk 2: index 6..10 -> "ghij"
        text = "abcdefghij"
        chunks = chunk_text(text, chunk_size=5, chunk_overlap=2)

        assert len(chunks) == 3
        assert chunks[0].text == "abcde"
        assert chunks[1].text == "defgh"
        assert chunks[2].text == "ghij"

        # Verify exact overlap between consecutive chunks
        assert chunks[0].text[-2:] == chunks[1].text[:2] == "de"
        assert chunks[1].text[-2:] == chunks[2].text[:2] == "gh"

    def test_preserves_metadata(self):
        """Chunks must preserve page numbers, source filename, and character offsets."""
        text = "Enterprise knowledge base documentation chunk."
        chunks = chunk_text(
            text=text,
            chunk_size=20,
            chunk_overlap=5,
            page_number=4,
            source_filename="hr_benefits.pdf",
        )

        assert len(chunks) > 1
        for idx, chunk in enumerate(chunks):
            assert chunk.page_number == 4
            assert chunk.page == 4
            assert chunk.source_filename == "hr_benefits.pdf"
            assert chunk.source == "hr_benefits.pdf"
            assert chunk.chunk_index == idx
            assert chunk.metadata["source"] == "hr_benefits.pdf"
            assert chunk.metadata["page"] == 4
            assert "char_start" in chunk.metadata
            assert "char_end" in chunk.metadata
            assert chunk.chunk_id.startswith("hr_benefits.pdf_p4_c")

    def test_chunking_is_deterministic(self):
        """Running chunk_text multiple times with the same input must produce identical outputs."""
        text = "The quick brown fox jumps over the lazy dog repeatedly for deterministic testing."
        run1 = chunk_text(text, chunk_size=25, chunk_overlap=5)
        run2 = chunk_text(text, chunk_size=25, chunk_overlap=5)

        assert len(run1) == len(run2)
        for c1, c2 in zip(run1, run2, strict=True):
            assert c1.chunk_id == c2.chunk_id
            assert c1.text == c2.text
            assert c1.metadata == c2.metadata

    def test_chunk_to_dict_export(self):
        """DocumentChunk.to_dict should export standard fields for downstream indexing."""
        chunk = DocumentChunk(
            chunk_id="policy_p1_c0_abc123",
            text="Remote work policy guidelines.",
            page_number=1,
            source_filename="policy.pdf",
            chunk_index=0,
            metadata={"source": "policy.pdf", "page": 1},
        )
        data = chunk.to_dict()
        assert data["chunk_id"] == "policy_p1_c0_abc123"
        assert data["source"] == "policy.pdf"
        assert data["page"] == 1
        assert data["text"] == "Remote work policy guidelines."


class TestChunkPages:
    """Tests for multi-page document chunking with chunk_pages."""

    def test_chunking_multiple_pages(self):
        """Should aggregate chunks across multiple DocumentPages while preserving page boundaries."""
        pages = [
            DocumentPage(text="Page one corporate strategy notes.", page_number=1, source_filename="doc.pdf"),
            DocumentPage(text="", page_number=2, source_filename="doc.pdf"),  # Empty page
            DocumentPage(text="Page three quarterly financial figures.", page_number=3, source_filename="doc.pdf"),
        ]

        chunks = chunk_pages(pages, chunk_size=20, overlap=5)

        # Page 2 was empty, so no chunks from page 2 should exist
        page_numbers_in_chunks = {chunk.page_number for chunk in chunks}
        assert 1 in page_numbers_in_chunks
        assert 3 in page_numbers_in_chunks
        assert 2 not in page_numbers_in_chunks

        # Verify page boundary preservation: each chunk has exactly 1 page citation
        for chunk in chunks:
            assert chunk.source == "doc.pdf"
            assert chunk.page in (1, 3)

    def test_chunk_pages_preserves_page_boundaries_strictly(self):
        """Chunks from different pages must never merge into a single ambiguous chunk."""
        pages = [
            DocumentPage(text="Short end of section.", page_number=1, source_filename="handbook.pdf"),
            DocumentPage(text="Start of new chapter.", page_number=2, source_filename="handbook.pdf"),
        ]

        chunks = chunk_pages(pages, chunk_size=1000, overlap=100)

        assert len(chunks) == 2
        assert chunks[0].page == 1
        assert chunks[0].text == "Short end of section."
        assert chunks[1].page == 2
        assert chunks[1].text == "Start of new chapter."

    def test_deduplication(self):
        """Should filter out exact duplicate chunks when deduplicate=True."""
        repeated_footer = "Confidential - Acme Corp Internal Use Only."
        pages = [
            DocumentPage(text=repeated_footer, page_number=1, source_filename="doc.pdf"),
            DocumentPage(text=repeated_footer, page_number=2, source_filename="doc.pdf"),
        ]

        chunks_deduped = chunk_pages(pages, chunk_size=100, overlap=10, deduplicate=True)
        chunks_with_dups = chunk_pages(pages, chunk_size=100, overlap=10, deduplicate=False)

        assert len(chunks_deduped) == 1
        assert len(chunks_with_dups) == 2

    def test_empty_pages_list(self):
        """Empty list of pages should return an empty list of chunks."""
        assert chunk_pages([]) == []
