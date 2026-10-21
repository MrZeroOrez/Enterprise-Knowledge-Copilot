"""Unit tests for the semantic retrieval module (Phase 4)."""

from pathlib import Path
from unittest.mock import MagicMock
import pytest

from src.chunking import DocumentChunk
from src.config import AppConfig
from src.retrieval import format_retrieval_context, search_chunks
from src.vector_store import add_chunks_to_vector_store


@pytest.fixture
def temp_chroma_dir(tmp_path: Path) -> Path:
    """Fixture providing a temporary ChromaDB directory."""
    path = tmp_path / "chroma_retrieval_test"
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_mock_client(query_vector: list[float]):
    """Helper to construct a mock Gemini client returning given vector for query."""
    mock_client = MagicMock()
    emb_obj = MagicMock()
    emb_obj.values = query_vector
    resp = MagicMock()
    resp.embeddings = [emb_obj]
    mock_client.models.embed_content.return_value = resp
    return mock_client


class TestRetrievalValidation:
    """Tests for retrieval parameter validation."""

    @pytest.mark.parametrize("empty_query", ["", "   ", "\t\n"])
    def test_empty_query_raises_value_error(self, empty_query: str):
        """Empty query string must raise ValueError."""
        with pytest.raises(ValueError, match="cannot be empty or whitespace"):
            search_chunks(empty_query, client=MagicMock())  # type: ignore

    def test_invalid_query_type_raises_value_error(self):
        """Non-string query must raise ValueError."""
        with pytest.raises(ValueError, match="Query must be a string"):
            search_chunks(12345, client=MagicMock())  # type: ignore

    def test_invalid_top_k(self):
        """top_k <= 0 or non-integer must raise ValueError."""
        with pytest.raises(ValueError, match="top_k must be a positive integer"):
            search_chunks("query", top_k=0)

        with pytest.raises(ValueError, match="top_k must be a positive integer"):
            search_chunks("query", top_k=-5)

        with pytest.raises(ValueError, match="top_k must be a positive integer"):
            search_chunks("query", top_k=True)  # bool is subclass of int

    def test_top_k_exceeds_max_limit(self):
        """top_k exceeding retrieval_max_top_k must raise ValueError."""
        cfg = AppConfig(
            gemini_api_key="mock",
            chat_model="gemini-3.5-flash",
            embedding_model="text-embedding-004",
            chunk_size=1000,
            chunk_overlap=200,
            upload_dir=Path("data/uploads"),
            chroma_persist_dir=Path("data/chromadb"),
            retrieval_top_k=5,
            retrieval_max_top_k=20,
        )
        with pytest.raises(ValueError, match="exceeds maximum allowed limit"):
            search_chunks("query", top_k=25, config=cfg)

    def test_invalid_source_filter(self):
        """Invalid source_filter must raise ValueError."""
        with pytest.raises(ValueError, match="source_filter must be a non-empty string"):
            search_chunks("query", source_filter="", client=MagicMock())

        with pytest.raises(ValueError, match="source_filter must be a non-empty string"):
            search_chunks("query", source_filter=123, client=MagicMock())  # type: ignore

    def test_invalid_distance_threshold(self):
        """Distance threshold out of [0.0, 2.0] must raise ValueError."""
        with pytest.raises(ValueError, match="must be a float between 0.0 and 2.0"):
            search_chunks("query", max_distance=-0.1, client=MagicMock())

        with pytest.raises(ValueError, match="must be a float between 0.0 and 2.0"):
            search_chunks("query", score_threshold=2.5, client=MagicMock())


class TestRetrievalExecution:
    """Tests for retrieval execution against ChromaDB with mocked query embeddings."""

    def test_search_on_empty_collection(self, temp_chroma_dir: Path):
        """Searching an empty collection should return an empty list without error."""
        mock_client = make_mock_client([0.1, 0.2, 0.3])
        results = search_chunks(
            query="Employee benefits",
            top_k=5,
            collection_name="empty_retrieval",
            persist_dir=temp_chroma_dir,
            embedding_client=mock_client,
        )
        assert results == []

    def test_search_returns_ranked_chunks_with_metadata(self, temp_chroma_dir: Path):
        """Should retrieve and rank matching chunks preserving citation metadata."""
        chunks = [
            DocumentChunk(
                chunk_id="policy_p1_c0",
                text="Employees are entitled to 20 days of paid annual leave.",
                page_number=1,
                source_filename="vacation_policy.pdf",
                chunk_index=0,
            ),
            DocumentChunk(
                chunk_id="hardware_p1_c0",
                text="Laptops must be returned to IT upon departure.",
                page_number=1,
                source_filename="hardware_policy.pdf",
                chunk_index=0,
            ),
        ]
        # Vectors where first chunk is [1.0, 0.0] and second is [0.0, 1.0]
        embeddings = [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
        add_chunks_to_vector_store(
            chunks=chunks,
            embeddings=embeddings,
            collection_name="search_test",
            persist_dir=temp_chroma_dir,
        )

        # Query vector close to first chunk [0.99, 0.01]
        mock_client = make_mock_client([0.99, 0.01])
        results = search_chunks(
            query="How many annual vacation days do I get?",
            top_k=1,
            collection_name="search_test",
            persist_dir=temp_chroma_dir,
            embedding_client=mock_client,
        )

        assert len(results) == 1
        hit = results[0]
        assert hit["chunk_id"] == "policy_p1_c0"
        assert hit["source"] == "vacation_policy.pdf"
        assert hit["page"] == 1
        assert "paid annual leave" in hit["text"]
        assert "distance" in hit

        # Verify task_type was RETRIEVAL_QUERY
        call_kwargs = mock_client.models.embed_content.call_args.kwargs
        assert call_kwargs["config"].task_type == "RETRIEVAL_QUERY"

    def test_source_filter_filters_results(self, temp_chroma_dir: Path):
        """Filtering by source_filename should only return chunks from that file."""
        chunks = [
            DocumentChunk(
                chunk_id="handbook_c1",
                text="Working hours are 9am to 5pm.",
                page_number=1,
                source_filename="handbook.pdf",
                chunk_index=0,
            ),
            DocumentChunk(
                chunk_id="it_c1",
                text="Never share your system password.",
                page_number=1,
                source_filename="security.pdf",
                chunk_index=0,
            ),
        ]
        embeddings = [
            [1.0, 0.0],
            [0.9, 0.1],  # also quite close to [1.0, 0.0]
        ]
        add_chunks_to_vector_store(
            chunks=chunks,
            embeddings=embeddings,
            collection_name="filter_test",
            persist_dir=temp_chroma_dir,
        )

        mock_client = make_mock_client([1.0, 0.0])
        results = search_chunks(
            query="office hours",
            top_k=5,
            source_filter="handbook.pdf",
            collection_name="filter_test",
            persist_dir=temp_chroma_dir,
            embedding_client=mock_client,
        )

        assert len(results) == 1
        assert results[0]["source"] == "handbook.pdf"
        assert results[0]["chunk_id"] == "handbook_c1"

    def test_max_distance_threshold_filtering(self, temp_chroma_dir: Path):
        """Chunks with distance > max_distance should be filtered out."""
        chunks = [
            DocumentChunk(
                chunk_id="close_chunk",
                text="Annual leave entitlement is 25 days.",
                page_number=1,
                source_filename="hr.pdf",
                chunk_index=0,
            ),
            DocumentChunk(
                chunk_id="far_chunk",
                text="Unrelated rocket propulsion chemistry.",
                page_number=1,
                source_filename="propulsion.pdf",
                chunk_index=0,
            ),
        ]
        # close chunk has distance ~0.0 to query [1.0, 0.0]
        # far chunk is orthogonal [0.0, 1.0], cosine distance = 1.0
        embeddings = [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
        add_chunks_to_vector_store(
            chunks=chunks,
            embeddings=embeddings,
            collection_name="thresh_test",
            persist_dir=temp_chroma_dir,
        )

        mock_client = make_mock_client([1.0, 0.0])
        # Strict threshold: max_distance = 0.2
        results = search_chunks(
            query="vacation leave",
            top_k=5,
            max_distance=0.2,
            collection_name="thresh_test",
            persist_dir=temp_chroma_dir,
            embedding_client=mock_client,
        )

        assert len(results) == 1
        assert results[0]["chunk_id"] == "close_chunk"
        assert results[0]["distance"] < 0.2

    def test_default_top_k_from_config(self, temp_chroma_dir: Path):
        """When top_k is None, it should use config.retrieval_top_k."""
        cfg = AppConfig(
            gemini_api_key="mock",
            chat_model="gemini-3.5-flash",
            embedding_model="text-embedding-004",
            chunk_size=1000,
            chunk_overlap=200,
            upload_dir=Path("data/uploads"),
            chroma_persist_dir=temp_chroma_dir,
            retrieval_top_k=2,
            retrieval_max_top_k=50,
        )
        chunks = [
            DocumentChunk(chunk_id=f"c_{i}", text=f"Chunk {i}", page_number=1, source_filename="f.pdf", chunk_index=i)
            for i in range(5)
        ]
        embeddings = [[float(i), 1.0] for i in range(5)]
        add_chunks_to_vector_store(
            chunks=chunks,
            embeddings=embeddings,
            collection_name="default_k_test",
            persist_dir=temp_chroma_dir,
        )

        mock_client = make_mock_client([0.0, 1.0])
        results = search_chunks(
            query="test",
            top_k=None,
            config=cfg,
            collection_name="default_k_test",
            embedding_client=mock_client,
        )
        assert len(results) <= 2


class TestFormatRetrievalContext:
    """Tests for format_retrieval_context formatting helper."""

    def test_format_empty_hits(self):
        """Empty hits list should return friendly fallback message."""
        result = format_retrieval_context([])
        assert result == "No relevant document passages found."

    def test_format_multiple_hits(self):
        """Should format hits with passage numbers, source citations, pages, and IDs."""
        hits = [
            {
                "chunk_id": "doc1_p1_c0",
                "source": "handbook.pdf",
                "page": 1,
                "text": "Office hours are 9:00 AM to 5:00 PM.",
                "distance": 0.12,
            },
            {
                "chunk_id": "doc1_p2_c0",
                "source": "handbook.pdf",
                "page": 2,
                "text": "Remote work is permitted 2 days per week.",
                "distance": 0.25,
            },
        ]
        context = format_retrieval_context(hits)
        assert "[Passage 1 | Source: handbook.pdf | Page: 1 | ID: doc1_p1_c0]" in context
        assert "Office hours are 9:00 AM to 5:00 PM." in context
        assert "[Passage 2 | Source: handbook.pdf | Page: 2 | ID: doc1_p2_c0]" in context
        assert "Remote work is permitted 2 days per week." in context
        assert "---" in context
