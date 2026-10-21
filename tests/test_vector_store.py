"""Unit tests for the ChromaDB vector store module (Phase 3)."""

from pathlib import Path
import pytest

from src.chunking import DocumentChunk
from src.vector_store import (
    add_chunks_to_vector_store,
    delete_chunks_by_source,
    get_chroma_client,
    get_chunks_by_source,
    get_collection_stats,
    get_or_create_collection,
    query_vector_store,
    reset_collection,
)


@pytest.fixture
def temp_db_dir(tmp_path: Path) -> Path:
    """Fixture providing an isolated temporary directory for ChromaDB persistence."""
    db_dir = tmp_path / "chromadb_test"
    db_dir.mkdir(parents=True, exist_ok=True)
    return db_dir


class TestChromaVectorStore:
    """Tests for ChromaDB client, collection lifecycle, and persistence."""

    def test_collection_creation_with_no_embedding_function(self, temp_db_dir: Path):
        """Should create a collection with embedding_function=None."""
        col = get_or_create_collection(
            collection_name="test_col",
            persist_dir=temp_db_dir,
        )
        assert col.name == "test_col"
        assert col.count() == 0

    def test_add_and_retrieve_chunks(self, temp_db_dir: Path):
        """Should add chunks with vectors and metadata, then retrieve them."""
        chunks = [
            DocumentChunk(
                chunk_id="handbook_p1_c0",
                text="Remote work guidelines and equipment policies.",
                page_number=1,
                source_filename="handbook.pdf",
                chunk_index=0,
            ),
            DocumentChunk(
                chunk_id="handbook_p2_c0",
                text="Health insurance and retirement match details.",
                page_number=2,
                source_filename="handbook.pdf",
                chunk_index=1,
            ),
        ]
        embeddings = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ]

        count = add_chunks_to_vector_store(
            chunks=chunks,
            embeddings=embeddings,
            doc_hash="hash123",
            collection_name="test_col",
            persist_dir=temp_db_dir,
        )
        assert count == 2

        stats = get_collection_stats(collection_name="test_col", persist_dir=temp_db_dir)
        assert stats["total_chunks"] == 2
        assert stats["distinct_sources"] == ["handbook.pdf"]

        # Retrieve by source
        stored = get_chunks_by_source(
            source="handbook.pdf",
            collection_name="test_col",
            persist_dir=temp_db_dir,
        )
        assert len(stored) == 2
        assert stored[0]["metadata"]["page"] == 1
        assert stored[1]["metadata"]["page"] == 2
        assert stored[0]["metadata"]["doc_hash"] == "hash123"

    def test_persistence_across_client_reopen(self, temp_db_dir: Path):
        """Data must persist when a new client connects to the same directory."""
        chunks = [
            DocumentChunk(
                chunk_id="doc_p1_c0",
                text="Persisted text content.",
                page_number=1,
                source_filename="doc.pdf",
                chunk_index=0,
            )
        ]
        add_chunks_to_vector_store(
            chunks=chunks,
            embeddings=[[0.7, 0.8, 0.9]],
            collection_name="persisted_col",
            persist_dir=temp_db_dir,
        )

        # Force a fresh client on the same directory
        fresh_client = get_chroma_client(persist_dir=temp_db_dir)
        col = fresh_client.get_collection(name="persisted_col", embedding_function=None)
        assert col.count() == 1

        records = get_chunks_by_source(
            source="doc.pdf",
            collection_name="persisted_col",
            persist_dir=temp_db_dir,
        )
        assert len(records) == 1
        assert records[0]["text"] == "Persisted text content."

    def test_query_vector_store_cosine(self, temp_db_dir: Path):
        """query_vector_store should rank chunks by similarity to the query vector."""
        chunks = [
            DocumentChunk(
                chunk_id="chunk_a",
                text="Apples and oranges are fruit.",
                page_number=1,
                source_filename="fruit.pdf",
                chunk_index=0,
            ),
            DocumentChunk(
                chunk_id="chunk_b",
                text="Cars and bicycles are vehicles.",
                page_number=1,
                source_filename="transport.pdf",
                chunk_index=0,
            ),
        ]
        # Vectors where chunk_a is closer to [1.0, 0.0]
        embeddings = [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
        add_chunks_to_vector_store(
            chunks=chunks,
            embeddings=embeddings,
            collection_name="query_test",
            persist_dir=temp_db_dir,
        )

        query_vec = [0.99, 0.01]
        hits = query_vector_store(
            query_embedding=query_vec,
            top_k=1,
            collection_name="query_test",
            persist_dir=temp_db_dir,
        )

        assert len(hits) == 1
        assert hits[0]["chunk_id"] == "chunk_a"
        assert hits[0]["source"] == "fruit.pdf"
        assert "distance" in hits[0]

    def test_delete_chunks_by_source_preserves_other_docs(self, temp_db_dir: Path):
        """Deleting one document's chunks must not remove chunks from another document."""
        chunks = [
            DocumentChunk(
                chunk_id="doc1_c0",
                text="Doc 1 text.",
                page_number=1,
                source_filename="doc1.pdf",
                chunk_index=0,
            ),
            DocumentChunk(
                chunk_id="doc2_c0",
                text="Doc 2 text.",
                page_number=1,
                source_filename="doc2.pdf",
                chunk_index=0,
            ),
        ]
        add_chunks_to_vector_store(
            chunks=chunks,
            embeddings=[[0.1, 0.2], [0.3, 0.4]],
            collection_name="del_test",
            persist_dir=temp_db_dir,
        )

        deleted = delete_chunks_by_source(
            source="doc1.pdf",
            collection_name="del_test",
            persist_dir=temp_db_dir,
        )
        assert deleted == 1

        stats = get_collection_stats(collection_name="del_test", persist_dir=temp_db_dir)
        assert stats["total_chunks"] == 1
        assert stats["distinct_sources"] == ["doc2.pdf"]

    def test_empty_collection_query(self, temp_db_dir: Path):
        """Querying an empty collection should return an empty list without error."""
        hits = query_vector_store(
            query_embedding=[0.1, 0.2],
            top_k=5,
            collection_name="empty_col",
            persist_dir=temp_db_dir,
        )
        assert hits == []
