"""Unit tests for the Gemini embeddings module (Phase 3)."""

from unittest.mock import MagicMock
import pytest
from google.genai.errors import APIError

from src.embeddings import (
    EmbeddingError,
    EmbeddingRateLimitError,
    embed_query,
    embed_text,
    embed_texts,
    get_genai_client,
    validate_embedding_vector,
)


def make_mock_embedding_response(vectors: list[list[float]]):
    """Create a mock response object mirroring google.genai client.models.embed_content."""
    response = MagicMock()
    mock_embeddings = []
    for vec in vectors:
        emb_obj = MagicMock()
        emb_obj.values = vec
        mock_embeddings.append(emb_obj)
    response.embeddings = mock_embeddings
    return response


class TestEmbeddingsValidation:
    """Tests for embedding parameter validation and vector checks."""

    def test_missing_api_key(self):
        """Should raise ValueError when GEMINI_API_KEY is not set or placeholder."""
        with pytest.raises(ValueError, match="GEMINI_API_KEY is not set"):
            get_genai_client(api_key="")

        with pytest.raises(ValueError, match="still set to the default placeholder"):
            get_genai_client(api_key="your_gemini_api_key_here")

    @pytest.mark.parametrize("empty_input", ["", "   ", "\n\t  "])
    def test_empty_text_input(self, empty_input):
        """Empty or whitespace-only text should raise ValueError."""
        with pytest.raises(ValueError, match="empty or whitespace-only text"):
            embed_text(empty_input, client=MagicMock())

    def test_non_string_input(self):
        """Non-string inputs should raise ValueError."""
        with pytest.raises(ValueError, match="Input text must be a string"):
            embed_text(12345, client=MagicMock())  # type: ignore

    def test_validate_vector_invalid(self):
        """Non-numeric or non-finite vector components should raise EmbeddingError."""
        with pytest.raises(EmbeddingError, match="empty or non-list"):
            validate_embedding_vector([])

        with pytest.raises(EmbeddingError, match="non-numeric"):
            validate_embedding_vector([0.1, "not-a-number", 0.3])  # type: ignore

        with pytest.raises(EmbeddingError, match="non-finite"):
            validate_embedding_vector([0.1, float("nan"), 0.3])

        with pytest.raises(EmbeddingError, match="non-finite"):
            validate_embedding_vector([0.1, float("inf"), 0.3])


class TestMockedEmbeddings:
    """Tests for embed_text, embed_query, and embed_texts with mocked API responses."""

    def test_embed_text_success(self):
        """Should return float vector from valid mocked client."""
        mock_client = MagicMock()
        mock_client.models.embed_content.return_value = make_mock_embedding_response(
            [[0.12, -0.45, 0.89]]
        )

        vector = embed_text("Enterprise document chunk text.", client=mock_client)

        assert vector == [0.12, -0.45, 0.89]
        mock_client.models.embed_content.assert_called_once()
        call_kwargs = mock_client.models.embed_content.call_args.kwargs
        assert call_kwargs["contents"] == "Enterprise document chunk text."
        assert call_kwargs["config"].task_type == "RETRIEVAL_DOCUMENT"

    def test_embed_query_uses_retrieval_query_task(self):
        """embed_query must specify task_type='RETRIEVAL_QUERY'."""
        mock_client = MagicMock()
        mock_client.models.embed_content.return_value = make_mock_embedding_response(
            [[0.5, 0.5, 0.0]]
        )

        vector = embed_query("What is the company leave policy?", client=mock_client)

        assert vector == [0.5, 0.5, 0.0]
        call_kwargs = mock_client.models.embed_content.call_args.kwargs
        assert call_kwargs["config"].task_type == "RETRIEVAL_QUERY"

    def test_embed_texts_batch_success(self):
        """embed_texts should process a batch and return parallel vectors."""
        mock_client = MagicMock()
        mock_client.models.embed_content.return_value = make_mock_embedding_response(
            [[0.1, 0.2], [0.3, 0.4]]
        )

        texts = ["First chunk content", "Second chunk content"]
        vectors = embed_texts(texts, client=mock_client, batch_size=10)

        assert len(vectors) == 2
        assert vectors[0] == [0.1, 0.2]
        assert vectors[1] == [0.3, 0.4]

    def test_empty_embeddings_response(self):
        """API returning empty response should raise EmbeddingError."""
        mock_client = MagicMock()
        empty_resp = MagicMock()
        empty_resp.embeddings = []
        mock_client.models.embed_content.return_value = empty_resp

        with pytest.raises(EmbeddingError, match="did not contain any embeddings"):
            embed_text("Some text", client=mock_client)

    def test_rate_limit_error_raised(self):
        """ResourceExhausted or 429 status code should raise EmbeddingRateLimitError."""
        mock_client = MagicMock()
        rate_error = APIError(429, {"error": {"message": "Resource has been exhausted (e.g. check quota)."}})
        mock_client.models.embed_content.side_effect = rate_error

        with pytest.raises(EmbeddingRateLimitError, match="rate limit or quota exceeded"):
            embed_text("High traffic query", client=mock_client)
