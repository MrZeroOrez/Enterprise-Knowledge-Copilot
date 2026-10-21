"""Unit tests for the retrieval evaluation and relevance metrics module (Phase 4)."""

import json
from pathlib import Path
from unittest.mock import MagicMock
import pytest

from src.chunking import DocumentChunk
from src.evaluation import (
    compute_precision_at_k,
    compute_recall_at_k,
    compute_reciprocal_rank,
    evaluate_retrieval,
    load_eval_dataset,
)
from src.retrieval import search_chunks
from src.vector_store import add_chunks_to_vector_store


class TestMetricsCalculation:
    """Tests for core relevance metrics: Recall@k, Precision@k, and RR."""

    def test_recall_at_k_perfect(self):
        """Should return 1.0 when all expected chunks are retrieved."""
        retrieved = ["c1", "c2", "c3"]
        expected = ["c1", "c2"]
        score = compute_recall_at_k(retrieved, expected, k=2)
        assert score == 1.0

    def test_recall_at_k_partial(self):
        """Should return proportion of expected chunks retrieved."""
        retrieved = ["c1", "c99", "c3"]
        expected = ["c1", "c2"]
        score = compute_recall_at_k(retrieved, expected, k=3)
        assert score == 0.5

    def test_recall_at_k_zero(self):
        """Should return 0.0 when no expected chunks are retrieved."""
        retrieved = ["c88", "c99"]
        expected = ["c1", "c2"]
        assert compute_recall_at_k(retrieved, expected) == 0.0

    def test_recall_unanswerable_query(self):
        """Unanswerable query (empty expected) should return 1.0 if empty, 0.0 if not."""
        assert compute_recall_at_k([], []) == 1.0
        assert compute_recall_at_k(["some_chunk"], []) == 0.0

    def test_precision_at_k_perfect(self):
        """Should return 1.0 when all retrieved items are relevant."""
        retrieved = ["c1", "c2"]
        expected = ["c1", "c2", "c3"]
        assert compute_precision_at_k(retrieved, expected, k=2) == 1.0

    def test_precision_at_k_partial(self):
        """Should compute fraction of retrieved items that are relevant."""
        retrieved = ["c1", "noise_1", "c2", "noise_2"]
        expected = ["c1", "c2"]
        assert compute_precision_at_k(retrieved, expected, k=4) == 0.5

    def test_precision_empty_cases(self):
        """Test precision edge cases with empty lists."""
        assert compute_precision_at_k([], []) == 1.0
        assert compute_precision_at_k([], ["c1"]) == 0.0
        assert compute_precision_at_k(["c1"], []) == 0.0

    def test_reciprocal_rank_ranks(self):
        """Should calculate reciprocal of 1-based rank of first relevant item."""
        expected = ["target"]
        assert compute_reciprocal_rank(["target", "noise"], expected) == 1.0
        assert compute_reciprocal_rank(["noise", "target"], expected) == 0.5
        assert pytest.approx(compute_reciprocal_rank(["n1", "n2", "target"], expected), 0.001) == 0.3333
        assert compute_reciprocal_rank(["n1", "n2"], expected) == 0.0

    def test_reciprocal_rank_unanswerable(self):
        """Unanswerable query (empty expected) should return 1.0 if empty, 0.0 otherwise."""
        assert compute_reciprocal_rank([], []) == 1.0
        assert compute_reciprocal_rank(["noise"], []) == 0.0


class TestDatasetLoading:
    """Tests for loading and validating evaluation datasets."""

    def test_load_non_existent_dataset(self, tmp_path: Path):
        """Loading non-existent file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_eval_dataset(tmp_path / "missing.json")

    def test_load_invalid_json(self, tmp_path: Path):
        """Malformed JSON should raise ValueError."""
        bad_json = tmp_path / "bad.json"
        bad_json.write_text("{not valid json}", encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid JSON"):
            load_eval_dataset(bad_json)

    def test_load_missing_required_keys(self, tmp_path: Path):
        """Dataset items missing required keys should raise ValueError."""
        bad_item = tmp_path / "incomplete.json"
        bad_item.write_text(json.dumps([{"id": "q1", "query": "hello"}]), encoding="utf-8")
        with pytest.raises(ValueError, match="missing keys"):
            load_eval_dataset(bad_item)

    def test_load_valid_fixture(self):
        """Should successfully load and validate the project eval_dataset fixture."""
        fixture_path = Path(__file__).parent / "fixtures" / "eval_dataset.json"
        data = load_eval_dataset(fixture_path)
        assert len(data) >= 10
        for item in data:
            assert "id" in item
            assert "query" in item
            assert "expected_chunk_ids" in item
            assert "query_type" in item


class TestEvaluateRetrievalRunner:
    """Tests for evaluate_retrieval runner with simulated retrieval functions."""

    def test_evaluate_with_mock_retriever_strings(self):
        """evaluate_retrieval should work with retrieval functions returning chunk ID strings."""
        dataset = [
            {"id": "q1", "query": "vacation", "expected_chunk_ids": ["c1"], "query_type": "exact_match"},
            {"id": "q2", "query": "crypto", "expected_chunk_ids": [], "query_type": "unanswerable"},
        ]

        def mock_retrieve(query: str):
            if "vacation" in query:
                return ["c1", "c2"]
            return []  # unanswerable correctly returned nothing

        summary = evaluate_retrieval(dataset, mock_retrieve, k=2)

        assert summary.total_queries == 2
        assert summary.answerable_queries == 1
        assert summary.unanswerable_queries == 1
        assert summary.mean_recall == 1.0
        assert summary.mean_reciprocal_rank == 1.0
        assert summary.hit_rate == 1.0
        assert "exact_match" in summary.by_type
        assert "unanswerable" in summary.by_type

        # Verify serializability
        summary_dict = summary.to_dict()
        assert isinstance(summary_dict, dict)
        assert len(summary_dict["results"]) == 2

    def test_evaluate_with_mock_retriever_dicts(self):
        """evaluate_retrieval should parse list of hit dictionaries from search_chunks."""
        dataset = [
            {"id": "q1", "query": "hours", "expected_chunk_ids": ["chunk_hours"], "query_type": "exact_match"},
        ]

        def mock_search(query: str):
            return [
                {"chunk_id": "chunk_hours", "source": "handbook.pdf", "distance": 0.12},
                {"chunk_id": "chunk_other", "source": "handbook.pdf", "distance": 0.45},
            ]

        summary = evaluate_retrieval(dataset, mock_search, k=5)
        assert summary.total_queries == 1
        assert summary.mean_recall == 1.0
        assert summary.mean_precision == 0.5  # 1 hit out of 2 returned
        assert summary.mean_reciprocal_rank == 1.0
        assert summary.results[0].retrieved_ids == ["chunk_hours", "chunk_other"]

    def test_evaluate_against_synthetic_chromadb_dataset(self, tmp_path: Path):
        """End-to-end evaluation using synthetic dataset fixture against ChromaDB."""
        fixture_path = Path(__file__).parent / "fixtures" / "eval_dataset.json"
        dataset = load_eval_dataset(fixture_path)

        # Chunks corresponding to the dataset
        chunks = [
            DocumentChunk("handbook_p1_c0", "Office hours are 9 AM to 5 PM Monday to Friday.", 1, "employee_handbook.pdf", 0),
            DocumentChunk("handbook_p2_c0", "Remote work is permitted 2 days per week with manager approval.", 2, "employee_handbook.pdf", 0),
            DocumentChunk("handbook_p3_c0", "Employees receive 20 days paid annual leave and 10 days sick leave.", 3, "employee_handbook.pdf", 0),
            DocumentChunk("it_p1_c0", "Passwords must be 14 characters and rotated every 90 days.", 1, "it_security_policy.pdf", 0),
            DocumentChunk("it_p2_c0", "Company laptops must be returned to IT within 3 business days of departure.", 2, "it_security_policy.pdf", 0),
            DocumentChunk("travel_p1_c0", "Meal per diem is $75 domestic and $120 international. Alcohol is prohibited.", 1, "travel_reimbursement.pdf", 0),
            DocumentChunk("travel_p2_c0", "Business class flights require VP approval. Domestic flights use economy.", 2, "travel_reimbursement.pdf", 0),
        ]

        embeddings = [[1.0 if i == j else 0.0 for j in range(7)] for i in range(7)]

        chroma_dir = tmp_path / "eval_chroma"
        add_chunks_to_vector_store(chunks, embeddings, collection_name="eval_collection", persist_dir=chroma_dir)

        def embed_side_effect(model, contents, config):
            q = contents.lower()
            vec = [0.0] * 7
            if "hours" in q:
                vec[0] = 1.0
            elif "home" in q or "remote" in q or "telecommuting" in q:
                vec[1] = 1.0
            elif "vacation" in q or "leave" in q:
                vec[2] = 1.0
            elif "password" in q:
                vec[3] = 1.0
            elif "equipment" in q or "laptop" in q:
                vec[4] = 1.0
            elif "meal" in q or "alcohol" in q:
                vec[5] = 1.0
            elif "air travel" in q or "flight" in q:
                vec[6] = 1.0
            else:
                # Noise vector with distance ~0.622 to any unit basis vector
                vec = [0.1] * 7

            emb_obj = MagicMock()
            emb_obj.values = vec
            resp = MagicMock()
            resp.embeddings = [emb_obj]
            return resp

        mock_client = MagicMock()
        mock_client.models.embed_content.side_effect = embed_side_effect

        def retrieve_fn(q: str):
            return search_chunks(
                query=q,
                top_k=2,
                max_distance=0.5,
                collection_name="eval_collection",
                persist_dir=chroma_dir,
                embedding_client=mock_client,
            )

        summary = evaluate_retrieval(dataset, retrieve_fn, k=2)

        assert summary.total_queries == 12
        assert summary.answerable_queries == 10
        assert summary.unanswerable_queries == 2
        assert summary.hit_rate >= 0.80
        assert summary.mean_reciprocal_rank >= 0.80

