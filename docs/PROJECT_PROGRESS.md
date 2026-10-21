# Enterprise Knowledge Copilot — Project Progress Log

This document tracks the incremental implementation progress, test verification records, and roadmap status across all project phases.

---

## Phase 1 — Project Foundation, PDF Extraction & Deterministic Text Chunking

- **Status**: Completed & Verified
- **Date**: October 2026
- **Target Repository**: `https://github.com/MrZeroOrez/Enterprise-Knowledge-Copilot.git`
- **Default Branch**: `main`

### 1. Features Completed
- [x] Initialized modular project architecture matching enterprise coding standards.
- [x] Configured safe environment loading via `python-dotenv` and immutable `AppConfig` dataclass in [`src/config.py`](file:///c:/Users/Admin/OneDrive/Desktop/LLM_PROJECTS/enterprise-knowledge-copilot/src/config.py).
- [x] Enforced strict chunking parameter boundary checks (`chunk_size > 0`, `chunk_overlap >= 0`, `chunk_overlap < chunk_size`).
- [x] Implemented API key secret masking (`repr(config)` redacts secrets with `[SET]` / `[NOT SET]`).
- [x] Isolated API key checks so unit tests run 100% offline without requiring paid API keys or active internet connections.
- [x] Created PDF extraction loader in [`src/document_loader.py`](file:///c:/Users/Admin/OneDrive/Desktop/LLM_PROJECTS/enterprise-knowledge-copilot/src/document_loader.py) using `pypdf`, extracting page-by-page text while preserving 1-indexed page numbers and source filenames for citations.
- [x] Implemented deterministic sliding-window chunking in [`src/chunking.py`](file:///c:/Users/Admin/OneDrive/Desktop/LLM_PROJECTS/enterprise-knowledge-copilot/src/chunking.py) with collision-resistant chunk IDs and duplicate content suppression.
- [x] Created clean architecture stub files for upcoming phases (`embeddings.py`, `vector_store.py`, `retrieval.py`, `rag_pipeline.py`, `app.py`).
- [x] Configured strict `.gitignore` preventing `.env`, secrets, caches, virtual environments, SQLite databases, and user uploads from entering Git.

### 2. Files Created or Modified
- `.gitignore` — Version control ignore rules
- `.env.example` — Environment template
- `requirements.txt` — Minimal Phase 1 dependencies (`pypdf`, `python-dotenv`, `pytest`)
- `src/__init__.py` — Package marker
- `src/config.py` — Safe configuration loader and validator
- `src/document_loader.py` — PDF text extractor with citation metadata
- `src/chunking.py` — Deterministic sliding-window text chunker
- `src/embeddings.py` — Stub (Phase 3)
- `src/vector_store.py` — Stub (Phase 3)
- `src/retrieval.py` — Stub (Phase 4)
- `src/rag_pipeline.py` — Stub (Phase 5)
- `app.py` — UI stub (Phase 6)
- `tests/__init__.py` — Test package marker
- `tests/test_config.py` — Config validation & secret masking unit tests (8 tests)
- `tests/test_chunking.py` — Text chunking, overlap & deduplication unit tests (22 tests)
- `tests/test_document_loader.py` — PDF parsing & error handling unit tests (10 tests)
- `data/uploads/.gitkeep` — Upload directory tracker
- `docs/architecture.md` — Architecture specification & Mermaid flowcharts
- `docs/PROJECT_PROGRESS.md` — Incremental progress log (this file)
- `README.md` — Beginner setup instructions & PowerShell commands

### 3. Tests Executed and Results
- **Command**: `pytest -v` (within project virtual environment)
- **Result**: `40 passed in 1.67s` (100% pass rate)
  - `tests/test_chunking.py`: 22 PASSED
  - `tests/test_config.py`: 8 PASSED
  - `tests/test_document_loader.py`: 10 PASSED

### 4. Known Issues and Limitations
- Pure `pypdf` extracts digitally embedded text; scanned image PDFs without an OCR text layer will return empty text.
- External API calls (Gemini embeddings, Gemini chat) and local vector persistence (ChromaDB) are architectural placeholders awaiting Phase 3.
- User interface and REST endpoints are planned for Phase 6 and Phase 8.

### 5. Next Phase
- **Phase 2**: PDF upload and document extraction validation (Completed).

---

## Phase 2 — PDF Processing and Document Preparation

- **Status**: Completed & Verified
- **Date**: October 2026
- **Target Repository**: `https://github.com/MrZeroOrez/Enterprise-Knowledge-Copilot.git`
- **Default Branch**: `main`

### 1. Features Completed
- [x] **PDF Validation**: Implemented strict checking of file size (configurable max upload size, default 10 MB), binary signature (`%PDF-` magic bytes), and page count.
- [x] **Safe Storage Pipeline (`save_uploaded_pdf`)**: Persists uploads safely into `data/uploads/` using UUID-prefixed collision-free filenames, sanitizing filenames to prevent path traversal (`../../`) and overwriting.
- [x] **Input Flexibility**: Accepts both raw `bytes` and file-like objects (e.g., `io.BytesIO`, Streamlit `UploadedFile`).
- [x] **Password/Encryption Handling**: Detects encrypted or password-protected PDFs and raises a clear `EncryptedPdfError` explaining they are unsupported.
- [x] **Scanned Document Detection**: Detects documents containing 0 extractable characters across all pages and raises `ScannedPdfError` explicitly reporting that OCR is not implemented yet.
- [x] **Whitespace Normalization**: Preserves paragraph breaks (`\n\n`) while collapsing null bytes, carriage returns, and excessive spaces/newlines.
- [x] **Page Citation Preservation**: Preserves 1-indexed `page_number` (`page`) and original filename (`source`) across both extracted pages (`DocumentPage`) and chunked records (`DocumentChunk`).
- [x] **Strict Page Boundary Preservation**: `chunk_pages()` chunks each page independently, ensuring chunks never merge content across page boundaries to avoid ambiguous citations.
- [x] **Custom Error Hierarchy**: Built clean exceptions (`DocumentProcessingError`, `FileValidationError`, `EncryptedPdfError`, `ScannedPdfError`, `CorruptPdfError`) providing actionable user-facing messages.
- [x] **Automated Synthetic Testing**: Created an offline Type 1 font PDF builder (`make_synthetic_pdf`) that tests multi-page text extraction, whitespace normalization, corrupted bytes, empty files, size limits, and encryption without private files or internet access.

### 2. Files Created or Modified
- `src/config.py` — Added `MAX_UPLOAD_SIZE_MB`, `max_upload_size_bytes`, and validation.
- `src/document_loader.py` — Added `FileValidationError`, `EncryptedPdfError`, `ScannedPdfError`, `CorruptPdfError`, `validate_pdf_bytes`, `sanitize_filename`, `save_uploaded_pdf`, `normalize_whitespace`, `extract_text_from_pdf`, and metadata properties.
- `src/chunking.py` — Added `chunk_pages()` interface, metadata aliases (`source`, `page`), dictionary export (`to_dict()`), and boundary preservation.
- `tests/test_config.py` — Added upload size configuration tests (10 tests total).
- `tests/test_document_loader.py` — Comprehensive suite covering all PDF edge cases (15 tests total).
- `tests/test_chunking.py` — Multi-page boundary, overlap, and dictionary export tests (25 tests total).
- `.env.example` — Added `MAX_UPLOAD_SIZE_MB=10`.
- `.gitignore` — Explicitly added `data/chroma_db/` and `*.pyc`.
- `docs/architecture.md` — Updated pipeline diagram and component status table.
- `docs/PROJECT_PROGRESS.md` — Documented Phase 2 progress (this file).
- `README.md` — Updated with Phase 2 capabilities, file size limits, and test instructions.

### 3. Tests Executed and Results
- **Command**: `.\.venv\Scripts\python.exe -m pytest -v`
- **Result**: **50 passed in 1.21s** (100% pass rate)
  - `tests/test_chunking.py`: 25 PASSED
  - `tests/test_config.py`: 10 PASSED
  - `tests/test_document_loader.py`: 15 PASSED

### 4. Known Issues and Limitations
- **No OCR Support**: Scanned or image-only PDFs do not yield text; the system identifies them and reports that OCR is not yet implemented.
- **Single-page Chunks**: Chunks are strictly confined within page boundaries to ensure citations are unambiguous. Multi-page spanning chunks are intentionally disallowed.
- **Future Components**: Embeddings, vector database (ChromaDB), Streamlit UI, and LangGraph workflow will be implemented in upcoming phases.

### 5. Next Phase
- **Phase 3**: Gemini Embeddings and ChromaDB Indexing (Completed).

---

## Phase 3 — Gemini Embeddings and ChromaDB Vector Indexing

- **Status**: Completed & Verified
- **Date**: October 2026
- **Target Repository**: `https://github.com/MrZeroOrez/Enterprise-Knowledge-Copilot.git`
- **Default Branch**: `main`

### 1. Features Completed
- [x] **Gemini Embeddings Integration (`src/embeddings.py`)**: Connected to Google Gemini API using official `google-genai` SDK, using configured embedding model (`text-embedding-004`).
- [x] **Task Type Specialization**: Applied `RETRIEVAL_DOCUMENT` for chunk embeddings and `RETRIEVAL_QUERY` for search queries, guaranteeing vector compatibility.
- [x] **Resilience & Bounded Backoff**: Implemented exponential backoff with jitter for transient errors (429 ResourceExhausted, 503) without infinite retry loops.
- [x] **Vector Quality Validation**: Enforced non-empty, finite numeric validation (detects NaN/Inf) without returning fabricated fallback vectors.
- [x] **Zero-Import Network Guarantee**: Module imports make zero network requests; API keys are validated only when embedding methods are explicitly invoked.
- [x] **ChromaDB Vector Store (`src/vector_store.py`)**: Persistent local storage in `data/chromadb/` with explicit `embedding_function=None` to ensure ChromaDB never downloads or executes heavy local models.
- [x] **Metadata Preservation**: Stores chunk IDs, source document filename, 1-indexed page number, and document content hashes.
- [x] **Document Indexing Pipeline (`src/indexing.py`)**: Coordinated Phase 2 extraction/chunking with Phase 3 embedding and storage into a unified `index_document()` function.
- [x] **Duplicate Detection**: Computes document-level SHA256 content hashes to skip re-indexing unchanged documents (`skipped_unchanged`), conserving API quota.
- [x] **All-or-Nothing Replacement**: Pre-generates and validates all chunk embeddings before touching the vector database. If embedding generation fails, existing index remains untouched.
- [x] **Scoped Document Replacement**: Updates or deletes chunks belonging strictly to the targeted document while keeping records for all other documents completely intact.
- [x] **Retrieval Interface (`src/retrieval.py`)**: Added `search_chunks()` returning ranked hits with cosine distance metrics and citation metadata.
- [x] **100% Mocked Offline Tests**: Added 29 new unit tests covering embedding validation, API retries, rate limits, ChromaDB persistence, change detection, and retrieval ranking.

### 2. Files Created or Modified
- `requirements.txt` — Added `google-genai` and `chromadb`.
- `.env.example` — Added `CHROMA_COLLECTION_NAME=enterprise_documents`.
- `src/config.py` — Added `CHROMA_COLLECTION_NAME`, environment aliases (`GEMINI_MODEL`, `GEMINI_EMBED_MODEL`, `CHROMA_PERSIST_DIR`), and `validate_embedding_model()`.
- `src/embeddings.py` — Implemented Gemini embeddings with official SDK, bounded backoff, and vector checks.
- `src/vector_store.py` — Implemented persistent ChromaDB operations with `embedding_function=None` and metadata filtering.
- `src/indexing.py` — Implemented end-to-end `index_document()` pipeline with change detection and atomic replacement.
- `src/retrieval.py` — Implemented `search_chunks()` semantic similarity search.
- `tests/test_config.py` — Added tests for collection name, aliases, and embedding model validation.
- `tests/test_embeddings.py` — 11 unit tests for embedding generation, batching, and error handling.
- `tests/test_vector_store.py` — 6 unit tests for ChromaDB storage, query ranking, and persistence.
- `tests/test_indexing.py` — 5 unit tests for pipeline indexing, duplicate skips, and safe replacement.
- `tests/test_retrieval.py` — 6 unit tests for semantic search and top-k ranking.
- `docs/architecture.md` — Updated pipeline diagram and component status table.
- `docs/PROJECT_PROGRESS.md` — Documented Phase 3 progress (this file).
- `README.md` — Updated with embeddings guide, ChromaDB rationale, manual testing guide, and quota details.

### 3. Tests Executed and Results
- **Command**: `.\.venv\Scripts\python.exe -m pytest -v`
- **Result**: **79 passed in 16.63s** (100% pass rate)
  - `tests/test_chunking.py`: 25 PASSED
  - `tests/test_config.py`: 11 PASSED
  - `tests/test_document_loader.py`: 15 PASSED
  - `tests/test_embeddings.py`: 11 PASSED
  - `tests/test_indexing.py`: 5 PASSED
  - `tests/test_retrieval.py`: 6 PASSED
  - `tests/test_vector_store.py`: 6 PASSED

### 4. Known Issues and Limitations
- **Gemini Free-Tier Rate Limits**: The free tier has rate limits (e.g. 15 RPM for free Gemini APIs); our bounded backoff handles transient spikes, but large document batches should remain modest.
- **Model Consistency**: Document chunks and search queries must use the same embedding model (`text-embedding-004`). If the model is changed in `.env`, the collection must be re-indexed.
- **RAG Generation**: LLM question answering and citations synthesis will be implemented in Phase 5.

### 5. Next Phase
- **Phase 4**: Semantic Retrieval and Relevance Checks (Completed).

---

## Phase 4 — Semantic Retrieval, Ranking, and Relevance Testing

- **Status**: Completed & Verified
- **Date**: October 2026
- **Target Repository**: `https://github.com/MrZeroOrez/Enterprise-Knowledge-Copilot.git`
- **Default Branch**: `main`

### 1. Features Completed
- [x] **Configurable & Bounded Top-k**: Added `RETRIEVAL_TOP_K` (default 5) and `RETRIEVAL_MAX_TOP_K` (default 50) in `src/config.py` with boundary checks (`top_k > 0`, `top_k <= max_top_k`, non-boolean).
- [x] **Cosine Distance Thresholding**: Added `max_distance` / `score_threshold` filtering and `RETRIEVAL_DISTANCE_THRESHOLD` config to suppress irrelevant chunks before prompt injection.
- [x] **Document Source Filtering**: Added `source_filter` parameter in `search_chunks()` supporting document segregation and multi-document routing via ChromaDB `$and` metadata filters.
- [x] **Duplicate Chunk Suppression**: Handled edge cases where vector queries return duplicate chunks, ensuring each chunk ID appears at most once in retrieved results.
- [x] **Context Window Formatting (`format_retrieval_context`)**: Implemented clean markdown formatting of retrieved chunks with source filenames, 1-indexed pages, and passage IDs for downstream Phase 5 prompt building.
- [x] **Retrieval Evaluation Metrics (`src/evaluation.py`)**: Implemented offline calculation of Recall@k, Precision@k, Reciprocal Rank (RR), and Mean Reciprocal Rank (MRR), with graceful handling of unanswerable queries.
- [x] **Synthetic Evaluation Benchmark (`tests/fixtures/eval_dataset.json`)**: Built a 12-query benchmark covering exact match, paraphrase, cross-page facts, multi-document routing, and unanswerable queries.
- [x] **Comprehensive Test Suite**: Added 30 new unit tests (109 tests total across 8 test modules) covering retrieval boundaries, thresholding, source filtering, context formatting, evaluation formulas, and end-to-end ChromaDB evaluation.

### 2. Files Created or Modified
- `src/config.py` — Added retrieval configuration parameters (`retrieval_top_k`, `retrieval_max_top_k`, `retrieval_distance_threshold`) and validation.
- `src/retrieval.py` — Implemented bounded top-k, distance filtering, source filtering, duplicate suppression, and `format_retrieval_context()`.
- `src/evaluation.py` — Implemented `EvaluationResult`, `EvaluationSummary`, `compute_recall_at_k()`, `compute_precision_at_k()`, `compute_reciprocal_rank()`, `load_eval_dataset()`, and `evaluate_retrieval()`.
- `tests/fixtures/eval_dataset.json` — 12 synthetic evaluation test cases.
- `tests/test_config.py` — Added 4 new tests for retrieval configuration and bounds (15 tests total).
- `tests/test_retrieval.py` — Added 9 new tests for validation, source filter, distance thresholds, and context formatting (15 tests total).
- `tests/test_evaluation.py` — 16 new tests for evaluation metrics, dataset loading, and end-to-end ChromaDB benchmark.
- `.env.example` — Documented `RETRIEVAL_TOP_K`, `RETRIEVAL_MAX_TOP_K`, and `RETRIEVAL_DISTANCE_THRESHOLD`.
- `docs/architecture.md` — Updated pipeline diagram, status table, and Phase 4 architecture notes.
- `docs/PROJECT_PROGRESS.md` — Documented Phase 4 progress (this file).
- `README.md` — Updated with semantic retrieval concepts, ranking, thresholding, evaluation metrics, and test instructions.

### 3. Tests Executed and Results
- **Command**: `.\.venv\Scripts\python.exe -m pytest -v`
- **Result**: **109 passed in 20.52s** (100% pass rate)
  - `tests/test_chunking.py`: 25 PASSED
  - `tests/test_config.py`: 15 PASSED
  - `tests/test_document_loader.py`: 15 PASSED
  - `tests/test_embeddings.py`: 11 PASSED
  - `tests/test_evaluation.py`: 16 PASSED
  - `tests/test_indexing.py`: 5 PASSED
  - `tests/test_retrieval.py`: 15 PASSED
  - `tests/test_vector_store.py`: 6 PASSED

### 4. Known Issues and Limitations
- **Threshold Calibration**: The optimal `RETRIEVAL_DISTANCE_THRESHOLD` varies depending on embedding model density and domain vocabulary. The default is left unconstrained (`None`) so all top-k items are returned unless explicitly configured.
- **RAG Generation**: Question answering and citation synthesis with Gemini 2.5 Flash will be implemented in Phase 5.

### 5. Next Phase
- **Phase 5**: Grounded Generation and Citations (Completed).

---

## Phase 5 — Grounded LLM Answer Generation with Verifiable Source Citations

- **Status**: Completed & Verified
- **Date**: October 2026
- **Target Repository**: `https://github.com/MrZeroOrez/Enterprise-Knowledge-Copilot.git`
- **Default Branch**: `main`

### 1. Features Completed
- [x] **Grounded RAG Pipeline (`src/rag_pipeline.py`)**: Implemented high-level `answer_question()` integrating Phase 4 semantic retrieval, bounded context budgeting, Gemini generation, and citation validation.
- [x] **Context Window Budgeting (`build_rag_context`)**: Formats retrieved chunks with reference IDs (`[S1]`, `[S2]`, ...), enforces character budget (`RAG_MAX_CONTEXT_CHARS`, default 20,000 chars), preserves complete passages without breaking sentences, and sets `context_truncated=True` if budget is exceeded.
- [x] **Grounded System Prompt**: Enforces evidence-only grounding, prompt-injection defense (untrusted PDF text is treated solely as data, never instructions), conflict explanations across differing documents, and honest fallback when evidence is lacking.
- [x] **Deterministic Citation Validation (`validate_and_format_citations`)**: Detects model citation markers `[S1]`, verifies them against trusted application retrieval metadata, maps them to `[<source>, p. <page>]`, and rejects/strips hallucinated citation IDs (`[S99]`).
- [x] **Honest Absence Reporting**: When vector store is empty, returns `status="no_documents"`. When no chunks match or pass distance threshold, returns `status="insufficient_evidence"` with a standard fallback message without calling Gemini.
- [x] **Resilience & Bounded Backoff (`generate_chat_answer`)**: Implemented bounded exponential backoff with jitter for transient 429 rate limit or 503 errors when calling Gemini.
- [x] **Zero-Import Network Guarantee & Offline Tests**: 100% of unit tests run offline using mocks with zero API quota consumption.
- [x] **Configuration Enhancements (`src/config.py`)**: Added `RAG_MAX_CONTEXT_CHARS`, `RAG_TOP_K` alias, and validation methods.

### 2. Files Created or Modified
- `src/config.py` — Added `rag_max_context_chars`, `validate_chat_model()`, `validate_rag_params()`, and `RAG_TOP_K` alias.
- `src/rag_pipeline.py` — Implemented complete RAG pipeline, context building, grounded answer generation, citation validation, and custom exceptions (`RAGError`, `RAGGenerationError`, `RAGRateLimitError`).
- `.env.example` — Added `RAG_MAX_CONTEXT_CHARS` and `RAG_TOP_K`.
- `tests/test_config.py` — Added tests for RAG configuration and chat model validation (19 tests total).
- `tests/test_rag_pipeline.py` — 21 unit tests covering question validation, context budgeting, citation replacement, hallucination rejection, rate limit retries, prompt injection defense, and full end-to-end integration.
- `docs/architecture.md` — Updated flowchart with Phase 5 RAG flow, status table, and Phase 5 architecture notes.
- `docs/PROJECT_PROGRESS.md` — Documented Phase 5 progress (this file).
- `README.md` — Updated with RAG concepts, context budgeting, citation validation, manual verification instructions, and test counts.

### 3. Tests Executed and Results
- **Command**: `.\.venv\Scripts\python.exe -m pytest -v`
- **Result**: **134 passed in 21.17s** (100% pass rate)
  - `tests/test_chunking.py`: 25 PASSED
  - `tests/test_config.py`: 19 PASSED
  - `tests/test_document_loader.py`: 16 PASSED
  - `tests/test_embeddings.py`: 11 PASSED
  - `tests/test_evaluation.py`: 16 PASSED
  - `tests/test_indexing.py`: 5 PASSED
  - `tests/test_rag_pipeline.py`: 21 PASSED
  - `tests/test_retrieval.py`: 15 PASSED
  - `tests/test_vector_store.py`: 6 PASSED

### 4. Known Issues and Limitations
- **Citation Validity vs. Factual Entailment**: The pipeline deterministically guarantees that every citation references a chunk actually present in the LLM's retrieved context. Mathematical claim-level entailment verification is planned for Phase 11.
- **Single-Turn Context**: Currently answers each question independently; multi-turn short-term conversation context is planned for Phase 7.

### 5. Next Phase
- **Phase 6**: Streamlit Interactive Web Application (Completed).

---

## Phase 6 — Streamlit Document Chat Interface

- **Status**: Completed & Verified
- **Date**: October 2026
- **Target Repository**: `https://github.com/MrZeroOrez/Enterprise-Knowledge-Copilot.git`
- **Default Branch**: `main`

### 1. Features Completed
- [x] **Interactive Streamlit Web UI (`app.py`)**: Built full document assistant interface with header, document management sidebar, library overview, and active chat interface.
- [x] **Safe On-Demand Indexing**: Explicit `Index Documents` button prevents accidental indexing on script reruns; reports real-time per-file status (pages processed, chunks indexed, or specific errors).
- [x] **Document Library Overview**: Displays distinct indexed documents and total chunk statistics via `list_indexed_documents()` and `get_collection_stats()`, with a manual refresh button.
- [x] **Search Scope Filter**: Enables limiting semantic search to a specific document or searching across all indexed documents.
- [x] **Active Conversation History**: Preserves chat messages in `st.session_state.messages` across reruns during the session.
- [x] **Verifiable Source Citation Cards**: Displays verified source filenames and page numbers (`[filename.pdf, p. N]`) beneath assistant answers.
- [x] **Clear Chat Control**: Resets active conversation messages safely without modifying indexed documents, uploaded PDFs, or ChromaDB data.
- [x] **Graceful Error Handling**: Maps rate limits (429), missing configuration, and document validation errors into clean user-facing notices without raw stack traces.
- [x] **Zero-Import Network Guarantee & Offline Tests**: Added 14 new automated unit tests (`tests/test_ui.py` and `tests/test_vector_store.py`) running 100% offline without live API keys.

### 2. Files Created or Modified
- `app.py` — Complete Streamlit application implementation with modular helper functions.
- `requirements.txt` — Added active dependency `streamlit>=1.35.0`.
- `src/vector_store.py` — Added `list_indexed_documents()` helper returning per-document chunk counts.
- `tests/test_ui.py` — 13 unit tests for session state, citation formatting, upload handling, chat queries, and import safety.
- `tests/test_vector_store.py` — Added test for `list_indexed_documents()` (7 tests total).
- `docs/architecture.md` — Updated pipeline diagram, status table, and Phase 6 architecture notes.
- `docs/PROJECT_PROGRESS.md` — Documented Phase 6 progress (this file).
- `README.md` — Updated with Streamlit launch command (`python -m streamlit run app.py`), capabilities, test counts, and workflow.

### 3. Tests Executed and Results
- **Command**: `.\.venv\Scripts\python.exe -m pytest -v`
- **Result**: **148 passed in 39.18s** (100% pass rate)
  - `tests/test_chunking.py`: 25 PASSED
  - `tests/test_config.py`: 19 PASSED
  - `tests/test_document_loader.py`: 16 PASSED
  - `tests/test_embeddings.py`: 11 PASSED
  - `tests/test_evaluation.py`: 16 PASSED
  - `tests/test_indexing.py`: 5 PASSED
  - `tests/test_rag_pipeline.py`: 21 PASSED
  - `tests/test_retrieval.py`: 15 PASSED
  - `tests/test_ui.py`: 13 PASSED
  - `tests/test_vector_store.py`: 7 PASSED

### 4. Known Issues and Limitations
- **Single-User Local Session**: Chat history is stored in Streamlit session state and resets upon browser refresh. Persistent multi-user conversation memory is planned for Phase 9.
- **Single-Turn Context**: Follow-up questions without explicit context are treated independently. Short-term conversational context is planned for Phase 7.

### 5. Next Phase
- **Phase 7**: Short-term Conversation History and Context Window Management (multi-turn query rewriting, conversational sliding context, and token budgeting).

