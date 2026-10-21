"""Unit tests for the document indexing pipeline (Phase 3)."""

from pathlib import Path
from unittest.mock import MagicMock
import pytest

from src.embeddings import EmbeddingError
from src.indexing import IndexingResult, index_document
from src.vector_store import get_chunks_by_source, get_collection_stats
from tests.test_document_loader import make_synthetic_pdf


def make_mock_client(return_vectors: list[list[float]]):
    """Helper to construct a mock Gemini client returning given vectors."""
    mock_client = MagicMock()
    mock_embeddings = []
    for vec in return_vectors:
        emb_obj = MagicMock()
        emb_obj.values = vec
        mock_embeddings.append(emb_obj)
    resp = MagicMock()
    resp.embeddings = mock_embeddings
    mock_client.models.embed_content.return_value = resp
    return mock_client


@pytest.fixture
def temp_chroma_dir(tmp_path: Path) -> Path:
    """Fixture providing a temporary ChromaDB directory."""
    path = tmp_path / "chroma_index_test"
    path.mkdir(parents=True, exist_ok=True)
    return path


class TestDocumentIndexing:
    """Tests for the complete indexing workflow with mocked embeddings."""

    def test_successful_indexing(self, temp_chroma_dir: Path):
        """Should extract, chunk, embed, and index a PDF document."""
        pdf_bytes = make_synthetic_pdf(["Quarterly Financial Review Notes."])
        mock_client = make_mock_client([[0.1, 0.2, 0.3]])

        result = index_document(
            file_source=pdf_bytes,
            original_filename="finance.pdf",
            persist_dir=temp_chroma_dir,
            collection_name="idx_test",
            embedding_client=mock_client,
        )

        assert isinstance(result, IndexingResult)
        assert result.status == "indexed"
        assert result.chunks_indexed == 1
        assert result.source == "finance.pdf"
        assert result.is_success is True

        stored = get_chunks_by_source("finance.pdf", collection_name="idx_test", persist_dir=temp_chroma_dir)
        assert len(stored) == 1
        assert stored[0]["metadata"]["page"] == 1

    def test_empty_document_indexing(self, temp_chroma_dir: Path):
        """Empty documents (0 extractable text) should return status='empty' and not insert records."""
        pdf_bytes = make_synthetic_pdf(["", ""])  # 2 blank pages
        mock_client = MagicMock()

        result = index_document(
            file_source=pdf_bytes,
            original_filename="blank.pdf",
            persist_dir=temp_chroma_dir,
            collection_name="idx_test",
            embedding_client=mock_client,
        )

        assert result.status == "empty"
        assert result.chunks_indexed == 0
        mock_client.models.embed_content.assert_not_called()

    def test_duplicate_indexing_skipped_when_unchanged(self, temp_chroma_dir: Path):
        """Re-indexing an unchanged document should skip embedding API calls and return skipped_unchanged."""
        pdf_bytes = make_synthetic_pdf(["Company engineering guidelines."])
        mock_client = make_mock_client([[0.1, 0.2, 0.3]])

        # First indexing
        res1 = index_document(
            file_source=pdf_bytes,
            original_filename="engineering.pdf",
            persist_dir=temp_chroma_dir,
            collection_name="idx_test",
            embedding_client=mock_client,
        )
        assert res1.status == "indexed"
        assert mock_client.models.embed_content.call_count == 1

        # Second indexing of identical content
        res2 = index_document(
            file_source=pdf_bytes,
            original_filename="engineering.pdf",
            persist_dir=temp_chroma_dir,
            collection_name="idx_test",
            embedding_client=mock_client,
        )
        assert res2.status == "skipped_unchanged"
        # Crucial: embed_content was NOT called a second time, preserving quota!
        assert mock_client.models.embed_content.call_count == 1

    def test_reindexing_changed_content_replaces_old_chunks(self, temp_chroma_dir: Path):
        """Indexing changed content replaces old chunks and leaves other documents intact."""
        doc1_bytes = make_synthetic_pdf(["Document 1 content."])
        doc2_v1_bytes = make_synthetic_pdf(["Document 2 initial version."])
        doc2_v2_bytes = make_synthetic_pdf(["Document 2 updated second revision with new policy."])

        client1 = make_mock_client([[0.1, 0.1]])
        index_document(
            file_source=doc1_bytes,
            original_filename="doc1.pdf",
            persist_dir=temp_chroma_dir,
            collection_name="multi_test",
            embedding_client=client1,
        )

        client2 = make_mock_client([[0.2, 0.2]])
        index_document(
            file_source=doc2_v1_bytes,
            original_filename="doc2.pdf",
            persist_dir=temp_chroma_dir,
            collection_name="multi_test",
            embedding_client=client2,
        )

        stats_before = get_collection_stats(collection_name="multi_test", persist_dir=temp_chroma_dir)
        assert stats_before["total_chunks"] == 2

        # Now re-index doc2 with changed content
        client3 = make_mock_client([[0.9, 0.9]])
        res_update = index_document(
            file_source=doc2_v2_bytes,
            original_filename="doc2.pdf",
            persist_dir=temp_chroma_dir,
            collection_name="multi_test",
            embedding_client=client3,
        )

        assert res_update.status == "replaced"

        # Verify doc1 is still intact
        doc1_stored = get_chunks_by_source("doc1.pdf", collection_name="multi_test", persist_dir=temp_chroma_dir)
        assert len(doc1_stored) == 1

        # Verify doc2 was updated
        doc2_stored = get_chunks_by_source("doc2.pdf", collection_name="multi_test", persist_dir=temp_chroma_dir)
        assert len(doc2_stored) == 1
        assert "updated second revision" in doc2_stored[0]["text"]

    def test_embedding_failure_does_not_corrupt_index(self, temp_chroma_dir: Path):
        """If embedding generation fails, the previous index must remain intact."""
        # Initial valid index
        pdf_v1 = make_synthetic_pdf(["Stable handbook text."])
        client_ok = make_mock_client([[0.1, 0.2]])
        index_document(
            file_source=pdf_v1,
            original_filename="safe.pdf",
            persist_dir=temp_chroma_dir,
            collection_name="safe_test",
            embedding_client=client_ok,
        )

        # Attempt to re-index changed content with a client that throws EmbeddingError
        pdf_v2 = make_synthetic_pdf(["Changed text that fails to embed."])
        failing_client = MagicMock()
        failing_client.models.embed_content.side_effect = EmbeddingError("API Quota Error")

        with pytest.raises(EmbeddingError, match="API Quota Error"):
            index_document(
                file_source=pdf_v2,
                original_filename="safe.pdf",
                persist_dir=temp_chroma_dir,
                collection_name="safe_test",
                embedding_client=failing_client,
            )

        # Verify the database was NOT deleted or corrupted:
        stored = get_chunks_by_source("safe.pdf", collection_name="safe_test", persist_dir=temp_chroma_dir)
        assert len(stored) == 1
        assert "Stable handbook text." in stored[0]["text"]
