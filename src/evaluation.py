"""Retrieval evaluation and relevance scoring metrics (Phase 4).

Provides deterministic offline metrics for evaluating semantic search:
- Recall@k: Proportion of expected relevant chunks retrieved in top k.
- Precision@k: Proportion of top-k retrieved chunks that are relevant.
- Reciprocal Rank (RR): Multiplicative inverse of the rank of the first relevant chunk.
- Mean Reciprocal Rank (MRR): Average Reciprocal Rank across all evaluated queries.
"""

from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EvaluationResult:
    """Individual query evaluation result."""

    query_id: str
    query: str
    query_type: str
    expected_ids: list[str]
    retrieved_ids: list[str]
    recall: float
    precision: float
    reciprocal_rank: float
    is_hit: bool


@dataclass(frozen=True)
class EvaluationSummary:
    """Aggregated evaluation summary across a dataset."""

    total_queries: int
    answerable_queries: int
    unanswerable_queries: int
    mean_recall: float
    mean_precision: float
    mean_reciprocal_rank: float
    hit_rate: float
    by_type: dict[str, dict[str, float]]
    results: list[EvaluationResult]

    def to_dict(self) -> dict[str, Any]:
        """Convert evaluation summary to a serializable dictionary."""
        return asdict(self)


def compute_recall_at_k(
    retrieved_ids: list[str],
    expected_ids: list[str],
    k: int | None = None,
) -> float:
    """Compute Recall@k.

    Recall@k measures the fraction of relevant documents that were successfully
    retrieved in the top k results:
        Recall@k = |Retrieved_k ∩ Expected| / |Expected|

    Special cases:
        - If expected_ids is empty (unanswerable / out-of-domain query):
            - Returns 1.0 if retrieved_ids is also empty (correctly suppressed).
            - Returns 0.0 if retrieved_ids is non-empty (false positive retrieval).

    Args:
        retrieved_ids: Ordered list of retrieved chunk IDs.
        expected_ids: List of ground-truth relevant chunk IDs.
        k: Optional cutoff rank. If provided, only top k retrieved_ids are used.

    Returns:
        Recall score as a float between 0.0 and 1.0.
    """
    effective_retrieved = retrieved_ids[:k] if k is not None else retrieved_ids
    unique_expected = set(expected_ids)

    if not unique_expected:
        return 1.0 if not effective_retrieved else 0.0

    hits = set(effective_retrieved) & unique_expected
    return len(hits) / len(unique_expected)


def compute_precision_at_k(
    retrieved_ids: list[str],
    expected_ids: list[str],
    k: int | None = None,
) -> float:
    """Compute Precision@k.

    Precision@k measures the fraction of retrieved documents in the top k that are relevant:
        Precision@k = |Retrieved_k ∩ Expected| / |Retrieved_k|

    Special cases:
        - If retrieved_ids is empty:
            - Returns 1.0 if expected_ids is also empty (correctly returned nothing).
            - Returns 0.0 if expected_ids is non-empty (missed relevant documents).
        - If expected_ids is empty but retrieved_ids is non-empty:
            - Returns 0.0 (all retrieved results are irrelevant noise).

    Args:
        retrieved_ids: Ordered list of retrieved chunk IDs.
        expected_ids: List of ground-truth relevant chunk IDs.
        k: Optional cutoff rank.

    Returns:
        Precision score as a float between 0.0 and 1.0.
    """
    effective_retrieved = retrieved_ids[:k] if k is not None else retrieved_ids
    unique_expected = set(expected_ids)

    if not effective_retrieved:
        return 1.0 if not unique_expected else 0.0

    if not unique_expected:
        return 0.0

    hits = set(effective_retrieved) & unique_expected
    return len(hits) / len(effective_retrieved)


def compute_reciprocal_rank(
    retrieved_ids: list[str],
    expected_ids: list[str],
    k: int | None = None,
) -> float:
    """Compute Reciprocal Rank (RR) for a single query.

    RR evaluates ranking quality by taking the reciprocal of the 1-based rank
    of the *first* relevant chunk:
        RR = 1.0 / rank_of_first_relevant_chunk
        RR = 0.0 if no relevant chunk appears in the top k results.

    Special cases:
        - If expected_ids is empty:
            - Returns 1.0 if retrieved_ids is empty (correctly avoided false positives).
            - Returns 0.0 if retrieved_ids is non-empty.

    Args:
        retrieved_ids: Ordered list of retrieved chunk IDs.
        expected_ids: List of ground-truth relevant chunk IDs.
        k: Optional cutoff rank.

    Returns:
        Reciprocal rank float between 0.0 and 1.0.
    """
    effective_retrieved = retrieved_ids[:k] if k is not None else retrieved_ids
    unique_expected = set(expected_ids)

    if not unique_expected:
        return 1.0 if not effective_retrieved else 0.0

    for rank, chunk_id in enumerate(effective_retrieved, start=1):
        if chunk_id in unique_expected:
            return 1.0 / rank

    return 0.0


def load_eval_dataset(dataset_path: Path | str) -> list[dict[str, Any]]:
    """Load and validate an evaluation dataset from a JSON file.

    Args:
        dataset_path: Path to the JSON dataset.

    Returns:
        List of parsed sample dictionaries.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If JSON formatting is invalid or required keys are missing.
    """
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Evaluation dataset file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in evaluation dataset: {exc}") from exc

    if not isinstance(data, list):
        raise ValueError("Evaluation dataset root must be a JSON array of query objects.")

    required_keys = {"id", "query", "expected_chunk_ids"}
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Dataset entry at index {idx} must be a JSON object.")
        missing = required_keys - set(item.keys())
        if missing:
            raise ValueError(f"Dataset entry at index {idx} (id={item.get('id')}) is missing keys: {missing}")

    return data


def evaluate_retrieval(
    dataset: list[dict[str, Any]],
    retrieve_fn: Callable[[str], list[str] | list[dict[str, Any]]],
    k: int = 5,
) -> EvaluationSummary:
    """Run evaluation across all queries in a dataset using a retrieval function.

    Args:
        dataset: List of dataset query entries.
        retrieve_fn: Callable taking a query string and returning either a list of chunk IDs
            or a list of hit dictionaries containing 'chunk_id'.
        k: Retrieval top_k cutoff for evaluation.

    Returns:
        EvaluationSummary containing overall metrics, per-type breakdown, and individual results.
    """
    results: list[EvaluationResult] = []
    type_metrics: dict[str, list[tuple[float, float, float, bool]]] = defaultdict(list)

    total_queries = len(dataset)
    answerable_count = 0
    unanswerable_count = 0

    for item in dataset:
        qid = str(item.get("id", ""))
        query = str(item.get("query", ""))
        qtype = str(item.get("query_type", "general"))
        expected_ids = list(item.get("expected_chunk_ids", []))

        is_answerable = len(expected_ids) > 0
        if is_answerable:
            answerable_count += 1
        else:
            unanswerable_count += 1

        raw_retrieved = retrieve_fn(query)
        # Handle both list[str] and list[dict]
        retrieved_ids: list[str] = []
        for r in raw_retrieved:
            if isinstance(r, dict):
                cid = r.get("chunk_id")
                if cid:
                    retrieved_ids.append(str(cid))
            elif isinstance(r, str):
                retrieved_ids.append(r)

        recall = compute_recall_at_k(retrieved_ids, expected_ids, k=k)
        precision = compute_precision_at_k(retrieved_ids, expected_ids, k=k)
        rr = compute_reciprocal_rank(retrieved_ids, expected_ids, k=k)
        is_hit = bool(set(retrieved_ids[:k]) & set(expected_ids)) if is_answerable else (len(retrieved_ids[:k]) == 0)

        result = EvaluationResult(
            query_id=qid,
            query=query,
            query_type=qtype,
            expected_ids=expected_ids,
            retrieved_ids=retrieved_ids[:k],
            recall=recall,
            precision=precision,
            reciprocal_rank=rr,
            is_hit=is_hit,
        )
        results.append(result)
        type_metrics[qtype].append((recall, precision, rr, is_hit))

    # Compute overall averages
    if total_queries > 0:
        mean_recall = sum(r.recall for r in results) / total_queries
        mean_precision = sum(r.precision for r in results) / total_queries
        mrr = sum(r.reciprocal_rank for r in results) / total_queries
        hit_rate = sum(1 for r in results if r.is_hit) / total_queries
    else:
        mean_recall = mean_precision = mrr = hit_rate = 0.0

    # Compute breakdown by type
    by_type: dict[str, dict[str, float]] = {}
    for qtype, scores in type_metrics.items():
        n = len(scores)
        by_type[qtype] = {
            "count": float(n),
            "mean_recall": round(sum(s[0] for s in scores) / n, 4),
            "mean_precision": round(sum(s[1] for s in scores) / n, 4),
            "mrr": round(sum(s[2] for s in scores) / n, 4),
            "hit_rate": round(sum(1 for s in scores if s[3]) / n, 4),
        }

    return EvaluationSummary(
        total_queries=total_queries,
        answerable_queries=answerable_count,
        unanswerable_queries=unanswerable_count,
        mean_recall=round(mean_recall, 4),
        mean_precision=round(mean_precision, 4),
        mean_reciprocal_rank=round(mrr, 4),
        hit_rate=round(hit_rate, 4),
        by_type=by_type,
        results=results,
    )
