"""Enterprise Knowledge Copilot - Streamlit Web Application (Phase 6).

Provides an interactive user interface for uploading enterprise PDFs,
triggering document indexing, asking natural-language questions, and viewing
grounded answers accompanied by verifiable page-level citations.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import streamlit as st

from src.config import AppConfig, load_config
from src.document_loader import (
    CorruptPdfError,
    EncryptedPdfError,
    FileValidationError,
    ScannedPdfError,
)
from src.embeddings import EmbeddingError, EmbeddingRateLimitError
from src.indexing import index_document
from src.rag_pipeline import (
    INSUFFICIENT_EVIDENCE_MESSAGE,
    RAGGenerationError,
    RAGRateLimitError,
    answer_question,
)
from src.vector_store import get_collection_stats, list_indexed_documents

logger = logging.getLogger(__name__)


# ==============================================================================
# UI Helper Functions (Pure / Testable)
# ==============================================================================

def init_session_state(state: Any = None) -> None:
    """Initialize session state variables for conversation history.

    Args:
        state: State dictionary to populate (defaults to st.session_state).
    """
    target = st.session_state if state is None else state
    if "messages" not in target:
        target["messages"] = []


def format_citation_markdown(citations: list[dict[str, Any]]) -> str:
    """Format a list of validated citation dictionaries into clean markdown.

    Args:
        citations: List of citation dicts containing 'source' and 'page'.

    Returns:
        Formatted markdown string with unique source and page bullets.
    """
    if not citations:
        return ""

    seen: set[tuple[str, int]] = set()
    lines = ["\n\n**Sources:**"]

    for citation in citations:
        source = citation.get("source", "unknown")
        page = citation.get("page", 1)
        key = (source, page)
        if key not in seen:
            seen.add(key)
            lines.append(f"- `{source}` — Page {page}")

    return "\n".join(lines)


def handle_document_upload(
    files: list[Any],
    config: AppConfig | None = None,
    collection_name: str | None = None,
    persist_dir: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Process and index a batch of uploaded PDF files.

    Args:
        files: List of file-like objects (e.g. Streamlit UploadedFile).
        config: Optional application configuration.
        collection_name: Optional target collection name.
        persist_dir: Optional database directory.

    Returns:
        List of result dictionaries summarizing indexing status per file:
        [{'filename': str, 'status': str, 'pages': int, 'chunks': int, 'error': str | None}]
    """
    results: list[dict[str, Any]] = []
    if not files:
        return results

    app_config = config or load_config()

    for file_obj in files:
        fname = getattr(file_obj, "name", "uploaded.pdf")
        try:
            # Read file bytes safely
            content_bytes = file_obj.getvalue() if hasattr(file_obj, "getvalue") else file_obj.read()

            result = index_document(
                file_input=content_bytes,
                original_filename=fname,
                collection_name=collection_name or app_config.chroma_collection_name,
                persist_dir=persist_dir or app_config.chroma_persist_dir,
                config=app_config,
            )

            results.append({
                "filename": fname,
                "status": result.status,
                "pages": result.pages_processed,
                "chunks": result.chunks_indexed,
                "error": None,
            })

        except (FileValidationError, ScannedPdfError, EncryptedPdfError, CorruptPdfError) as exc:
            results.append({
                "filename": fname,
                "status": "error",
                "pages": 0,
                "chunks": 0,
                "error": str(exc),
            })
        except (EmbeddingRateLimitError, RAGRateLimitError) as exc:
            results.append({
                "filename": fname,
                "status": "error",
                "pages": 0,
                "chunks": 0,
                "error": f"Gemini API rate limit exceeded: {exc}",
            })
        except Exception as exc:
            logger.error("Failed to index %s: %s", fname, exc, exc_info=True)
            results.append({
                "filename": fname,
                "status": "error",
                "pages": 0,
                "chunks": 0,
                "error": f"Indexing failed: {exc}",
            })

    return results


def handle_chat_query(
    question: str,
    source_filter: str | None = None,
    config: AppConfig | None = None,
    collection_name: str | None = None,
    persist_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Execute grounded question answering through the Phase 5 RAG pipeline.

    Catches and maps exceptions into safe, user-friendly responses.

    Args:
        question: User query string.
        source_filter: Optional document filename to isolate search.
        config: Optional loaded configuration.
        collection_name: Optional collection name.
        persist_dir: Optional database path.

    Returns:
        Structured result dict matching Phase 5 response schema.
    """
    clean_q = question.strip() if question else ""
    if not clean_q:
        return {
            "answer": "Please enter a non-empty question.",
            "citations": [],
            "status": "error",
        }

    app_config = config or load_config()

    try:
        return answer_question(
            question=clean_q,
            source_filter=source_filter,
            collection_name=collection_name or app_config.chroma_collection_name,
            persist_dir=persist_dir or app_config.chroma_persist_dir,
            config=app_config,
        )

    except (RAGRateLimitError, EmbeddingRateLimitError) as exc:
        return {
            "answer": (
                "The Gemini API rate limit or quota was exceeded. "
                "Please wait a few moments before trying again."
            ),
            "citations": [],
            "status": "error",
        }
    except ValueError as exc:
        return {
            "answer": f"Configuration error: {exc}",
            "citations": [],
            "status": "error",
        }
    except (RAGGenerationError, EmbeddingError) as exc:
        return {
            "answer": f"Could not generate an answer: {exc}",
            "citations": [],
            "status": "error",
        }
    except Exception as exc:
        logger.error("Unexpected error in chat query: %s", exc, exc_info=True)
        return {
            "answer": f"An unexpected error occurred: {exc}",
            "citations": [],
            "status": "error",
        }


# ==============================================================================
# Streamlit Interface Rendering
# ==============================================================================

def render_sidebar(config: AppConfig) -> str | None:
    """Render the sidebar for document upload, indexing, and management.

    Returns:
        Selected source_filter filename, or None if searching all documents.
    """
    st.sidebar.header("📁 Document Management")

    # 1. PDF File Uploader
    uploaded_files = st.sidebar.file_uploader(
        "Upload PDF Documents",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload digital PDF policy documents or manuals (up to 10 MB each).",
    )

    # 2. Explicit Indexing Trigger
    if st.sidebar.button("Index Documents", type="primary", use_container_width=True):
        if not uploaded_files:
            st.sidebar.warning("Please select at least one PDF file before clicking Index.")
        else:
            with st.sidebar.status("Indexing documents...", expanded=True) as status_box:
                results = handle_document_upload(uploaded_files, config=config)
                success_count = sum(1 for r in results if r["status"] == "indexed")
                skipped_count = sum(1 for r in results if r["status"] == "skipped_unchanged")
                error_count = sum(1 for r in results if r["status"] == "error")

                for r in results:
                    fname = r["filename"]
                    if r["status"] == "indexed":
                        st.write(f"✅ **{fname}**: Indexed ({r['pages']} pages, {r['chunks']} chunks)")
                    elif r["status"] == "skipped_unchanged":
                        st.write(f"ℹ️ **{fname}**: Unchanged (skipped)")
                    else:
                        st.write(f"❌ **{fname}**: {r['error']}")

                if error_count == 0:
                    status_box.update(
                        label=f"Indexing Complete! ({success_count} indexed, {skipped_count} skipped)",
                        state="complete",
                    )
                else:
                    status_box.update(
                        label=f"Indexing finished with {error_count} error(s).",
                        state="error",
                    )

    st.sidebar.markdown("---")

    # 3. Document Library Overview
    st.sidebar.subheader("📚 Indexed Library")

    if st.sidebar.button("🔄 Refresh Library", use_container_width=True):
        st.rerun()

    try:
        stats = get_collection_stats(
            collection_name=config.chroma_collection_name,
            persist_dir=config.chroma_persist_dir,
        )
        total_chunks = stats.get("total_chunks", 0)
        docs = list_indexed_documents(
            collection_name=config.chroma_collection_name,
            persist_dir=config.chroma_persist_dir,
        )
    except Exception as exc:
        st.sidebar.error(f"Could not load library: {exc}")
        total_chunks = 0
        docs = []

    if total_chunks == 0 or not docs:
        st.sidebar.caption("No documents indexed yet. Upload PDFs above to begin.")
        selected_source = None
    else:
        st.sidebar.caption(f"Total Chunks: **{total_chunks}** across **{len(docs)}** document(s)")
        for doc in docs:
            st.sidebar.text(f"• {doc['source']} ({doc['chunk_count']} chunks)")

        # Optional search filter
        doc_names = [d["source"] for d in docs]
        scope_options = ["All Documents"] + doc_names
        chosen_scope = st.sidebar.selectbox(
            "Filter Search Scope",
            options=scope_options,
            help="Limit search to a specific document or search across the entire library.",
        )
        selected_source = None if chosen_scope == "All Documents" else chosen_scope

    st.sidebar.markdown("---")

    # 4. Clear Chat Control
    if st.sidebar.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.sidebar.caption("💡 *Clearing chat history does not delete indexed documents.*")

    return selected_source


def main() -> None:
    """Main Streamlit application entry point."""
    st.set_page_config(
        page_title="Enterprise Knowledge Copilot",
        page_icon="🤖",
        layout="wide",
    )

    init_session_state()

    # Load configuration
    try:
        config = load_config()
    except Exception as exc:
        st.error(f"Configuration Error: {exc}")
        st.stop()

    # Header section
    st.title("🤖 Enterprise Knowledge Copilot")
    st.markdown(
        "Ask questions about your enterprise documents and receive **evidence-grounded answers** "
        "backed by verifiable page-level source citations."
    )

    # API Key status banner
    if config.is_api_key_set():
        st.caption(f"🟢 **API Configured** | Chat Model: `{config.chat_model}` | Embeddings: `{config.embedding_model}`")
    else:
        st.warning(
            "⚠️ **GEMINI_API_KEY is not configured.** Please set your Gemini API key in the `.env` file "
            "to enable document indexing and answer generation."
        )

    # Render sidebar and get document filter
    source_filter = render_sidebar(config)

    # Empty State Guidance
    if not st.session_state.messages:
        with st.container():
            st.info(
                "👋 **Welcome!** Follow these easy steps to get started:\n\n"
                "1. Upload one or more digital PDF documents in the left sidebar.\n"
                "2. Click **Index Documents** to extract text and generate vector embeddings.\n"
                "3. Type your question below to receive an accurate answer with exact document citations."
            )

    # Render Chat History
    for msg in st.session_state.messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        citations = msg.get("citations", [])

        with st.chat_message(role):
            st.markdown(content)
            if role == "assistant" and citations:
                citation_md = format_citation_markdown(citations)
                if citation_md:
                    st.markdown(citation_md)

    # Chat Input
    user_query = st.chat_input("Ask a question about your indexed documents...")
    if user_query:
        if not user_query.strip():
            return

        # Append and display user message
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Assistant response container
        with st.chat_message("assistant"):
            with st.spinner("Searching document evidence and generating grounded answer..."):
                response = handle_chat_query(
                    question=user_query,
                    source_filter=source_filter,
                    config=config,
                )

            answer_text = response.get("answer", INSUFFICIENT_EVIDENCE_MESSAGE)
            citations = response.get("citations", [])
            status = response.get("status", "answered")

            st.markdown(answer_text)

            if citations:
                citation_md = format_citation_markdown(citations)
                if citation_md:
                    st.markdown(citation_md)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer_text,
                "citations": citations,
                "status": status,
            })


if __name__ == "__main__":
    main()
