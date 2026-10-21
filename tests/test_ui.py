"""Unit tests for the Streamlit UI helper functions and state management (Phase 6)."""

from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from app import (
    format_citation_markdown,
    handle_chat_query,
    handle_document_upload,
    init_session_state,
)
from src.config import AppConfig
from src.document_loader import ScannedPdfError
from src.indexing import IndexingResult
from src.rag_pipeline import RAGRateLimitError


class TestSessionState:
    """Tests for Streamlit session state initialization and persistence."""

    def test_init_session_state_creates_messages_list(self):
        """Should initialize 'messages' key as an empty list if not present."""
        state: dict[str, list] = {}
        init_session_state(state=state)
        assert "messages" in state
        assert state["messages"] == []

    def test_init_session_state_preserves_existing_messages(self):
        """Should not overwrite existing chat messages when already initialized."""
        existing = [{"role": "user", "content": "Hello"}]
        state = {"messages": existing}
        init_session_state(state=state)
        assert len(state["messages"]) == 1
        assert state["messages"][0]["content"] == "Hello"


class TestCitationFormatting:
    """Tests for format_citation_markdown helper."""

    def test_format_empty_citations(self):
        """Empty citation list should return an empty string."""
        assert format_citation_markdown([]) == ""

    def test_format_single_citation(self):
        """Single citation should format cleanly with source and page."""
        citations = [{"source": "policy.pdf", "page": 4}]
        md = format_citation_markdown(citations)
        assert "**Sources:**" in md
        assert "- `policy.pdf` — Page 4" in md

    def test_format_multiple_citations_deduplicates(self):
        """Should format multiple sources and deduplicate identical source/page pairs."""
        citations = [
            {"source": "handbook.pdf", "page": 1},
            {"source": "it_guide.pdf", "page": 5},
            {"source": "handbook.pdf", "page": 1},  # duplicate
        ]
        md = format_citation_markdown(citations)
        assert "- `handbook.pdf` — Page 1" in md
        assert "- `it_guide.pdf` — Page 5" in md
        # Should only have 2 bullet lines
        bullets = [line for line in md.split("\n") if line.startswith("- ")]
        assert len(bullets) == 2


class TestDocumentUploadHandling:
    """Tests for handle_document_upload batch indexing."""

    def test_handle_empty_upload_list(self):
        """Empty files list should return empty result list."""
        assert handle_document_upload([]) == []

    def test_handle_successful_upload(self, monkeypatch):
        """Should index uploaded file and return summary dictionary."""
        mock_file = MagicMock()
        mock_file.name = "sample.pdf"
        mock_file.getvalue.return_value = b"%PDF-1.4 mock content"

        mock_result = IndexingResult(
            source="sample.pdf",
            pages_processed=3,
            chunks_created=6,
            chunks_indexed=6,
            status="indexed",
            message="Indexed 6 chunks.",
        )
        monkeypatch.setattr("app.index_document", lambda **kwargs: mock_result)

        results = handle_document_upload([mock_file])
        assert len(results) == 1
        assert results[0]["filename"] == "sample.pdf"
        assert results[0]["status"] == "indexed"
        assert results[0]["pages"] == 3
        assert results[0]["chunks"] == 6
        assert results[0]["error"] is None

    def test_handle_file_with_error_does_not_abort_batch(self, monkeypatch):
        """When one document fails, record its error and continue processing others."""
        good_file = MagicMock()
        good_file.name = "good.pdf"
        good_file.getvalue.return_value = b"%PDF-good"

        bad_file = MagicMock()
        bad_file.name = "scanned.pdf"
        bad_file.getvalue.return_value = b"%PDF-bad"

        def mock_index(**kwargs):
            fname = kwargs.get("original_filename")
            if fname == "scanned.pdf":
                raise ScannedPdfError("No extractable text found.")
            return IndexingResult(
                source="good.pdf",
                pages_processed=2,
                chunks_created=4,
                chunks_indexed=4,
                status="indexed",
                message="Indexed 4 chunks.",
            )

        monkeypatch.setattr("app.index_document", mock_index)

        results = handle_document_upload([bad_file, good_file])
        assert len(results) == 2
        # First file recorded error
        assert results[0]["filename"] == "scanned.pdf"
        assert results[0]["status"] == "error"
        assert "No extractable text" in results[0]["error"]
        # Second file succeeded
        assert results[1]["filename"] == "good.pdf"
        assert results[1]["status"] == "indexed"


class TestChatQueryHandling:
    """Tests for handle_chat_query wrapper."""

    def test_empty_question_returns_error(self):
        """Empty or whitespace query should return friendly validation error."""
        res = handle_chat_query("   ")
        assert res["status"] == "error"
        assert "non-empty question" in res["answer"]

    def test_successful_query_invocation(self, monkeypatch):
        """Should delegate to answer_question and return its structured output."""
        mock_response = {
            "answer": "Leave entitlement is 20 days. [handbook.pdf, p. 1]",
            "citations": [{"source": "handbook.pdf", "page": 1}],
            "status": "answered",
        }
        monkeypatch.setattr("app.answer_question", lambda **kwargs: mock_response)

        res = handle_chat_query("How much leave do I get?")
        assert res["status"] == "answered"
        assert "20 days" in res["answer"]
        assert len(res["citations"]) == 1

    def test_rate_limit_error_mapping(self, monkeypatch):
        """Rate limit error should map to user-friendly retry message."""
        def mock_fail(**kwargs):
            raise RAGRateLimitError("Rate limit exceeded")

        monkeypatch.setattr("app.answer_question", mock_fail)

        res = handle_chat_query("Query triggering rate limit")
        assert res["status"] == "error"
        assert "rate limit or quota was exceeded" in res["answer"]

    def test_value_error_mapping(self, monkeypatch):
        """ValueError should map to friendly configuration message."""
        def mock_fail(**kwargs):
            raise ValueError("GEMINI_API_KEY is not set.")

        monkeypatch.setattr("app.answer_question", mock_fail)

        res = handle_chat_query("Query triggering config error")
        assert res["status"] == "error"
        assert "Configuration error" in res["answer"]


class TestAppImportSafety:
    """Tests verifying app.py can be imported without network calls."""

    def test_app_module_imports_cleanly(self):
        """Importing app.py must not make external API calls or crash."""
        import app
        assert hasattr(app, "main")
        assert hasattr(app, "render_sidebar")
