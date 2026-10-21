"""Unit tests for the semantic retrieval module (Phase 3)."""

from pathlib import Path
from unittest.mock import MagicMock
import pytest

from src.chunking import DocumentChunk
from src.retrieval import search_chunks
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

    def test_invalid_top_k(self):
        """top_k <= 0 or non-integer must raise ValueError."""
        with pytest.raises(ValueError, match="top_k must be a positive integer"):
            search_chunks("query", top_k=0)

        with pytest.raises(ValueError, match="top_k must be a positive integer"):
            search_chunks("query", top_k=-5)


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
