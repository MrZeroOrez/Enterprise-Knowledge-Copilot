# Enterprise Knowledge Copilot

A production-minded, beginner-friendly AI document assistant that enables users to upload enterprise PDF documents, ask questions, and receive grounded answers backed by precise citations (filename and page numbers).

Designed for resource-constrained laptops:
- **No Ollama, Docker, or local LLMs** required.
- Minimal RAM and disk usage.
- Uses lightweight local Python libraries and Google's free-tier Gemini API.

---

## Current Status: Phase 2 (PDF Processing & Document Preparation)

> [!NOTE]  
> We have completed **Phase 2**. The PDF validation, safe storage, page-level text extraction with whitespace normalization, scanned document detection, and deterministic page-bounded chunking are fully implemented and verified with 50 automated tests. Embeddings, ChromaDB, and UI will be added in upcoming phases.

---

## Project Structure

```text
enterprise-knowledge-copilot/
├── app.py                      # UI entry point (Planned for Phase 6)
├── requirements.txt            # Phase 1 dependencies (lightweight)
├── .env.example                # Environment variable template
├── .gitignore                  # Git ignore rules (secrets, venvs, caches)
├── README.md                   # Beginner setup guide (this file)
├── src/
│   ├── __init__.py             # Package marker
│   ├── config.py               # Safe environment & config loader
│   ├── document_loader.py      # PDF text and metadata extraction
│   ├── chunking.py             # Deterministic chunking with overlap
│   ├── embeddings.py           # Gemini embeddings (Planned for Phase 3)
│   ├── vector_store.py         # ChromaDB interface (Planned for Phase 3)
│   ├── retrieval.py            # Semantic retrieval (Planned for Phase 4)
│   └── rag_pipeline.py         # Grounded RAG pipeline (Planned for Phase 5)
├── tests/
│   ├── __init__.py
│   ├── test_config.py          # Config validation tests
│   ├── test_chunking.py        # Chunking & overlap tests
│   └── test_document_loader.py # PDF extraction tests
├── data/
│   └── uploads/                # Document storage (.gitkeep)
└── docs/
    ├── architecture.md         # Phased architecture & system design
    └── PROJECT_PROGRESS.md     # Incremental progress log across phases
```

---

## Getting Started (Windows PowerShell)

Follow these step-by-step commands in **Windows PowerShell**:

### 1. Open PowerShell and navigate to the project directory

```powershell
cd c:\Users\Admin\OneDrive\Desktop\LLM_PROJECTS\enterprise-knowledge-copilot
```

### 2. Create a Python Virtual Environment

A virtual environment keeps your project packages isolated from the global system:

```powershell
python -m venv .venv
```

### 3. Activate the Virtual Environment

```powershell
.\.venv\Scripts\Activate.ps1
```

> [!TIP]  
> If PowerShell displays a script execution policy error (`running scripts is disabled on this system`), enable script execution for your current PowerShell session by running:  
> `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process`  
> Then re-run `.\.venv\Scripts\Activate.ps1`. When activated, your prompt will show `(.venv)`.

### 4. Install Phase 1 Dependencies

Upgrade pip and install the lightweight Phase 1 dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Create Your Local `.env` Configuration

Copy the `.env.example` template to `.env`:

```powershell
Copy-Item .env.example .env
```

> [!IMPORTANT]  
> Running the **Phase 1 unit tests does not require an active Gemini API key** or internet connection. When you reach Phase 3, you will obtain a free API key from [Google AI Studio](https://aistudio.google.com/) and paste it into `.env`.

### 6. Run the Automated Unit Test Suite

Execute all unit tests using `pytest`:

```powershell
pytest -v
```

All tests should pass cleanly!

---

## Implemented Capabilities

### 1. PDF Validation & Safe Storage (`src/document_loader.py`)
- **Size Validation**: Rejects files larger than `MAX_UPLOAD_SIZE_MB` (default 10 MB, configurable via environment).
- **Signature Check**: Verifies the binary `%PDF-` magic header before parsing.
- **Safe Storage (`save_uploaded_pdf`)**: Saves uploaded documents to `data/uploads/` using sanitized, UUID-prefixed filenames. Fully defends against directory traversal attacks (`../../`) and accidental overwrites.
- **Input Flexibility**: Directly accepts raw `bytes`, bytearrays, or file-like streams (`io.BytesIO`, Streamlit `UploadedFile`).
- **Encrypted PDF Handling**: Catches password-protected or encrypted PDFs and raises `EncryptedPdfError`.

### 2. PDF Text Extraction & Scanned Detection (`src/document_loader.py`)
- **Page-by-Page Extraction**: Extracts text per page using `pypdf`, tracking 1-indexed `page_number` (`page`) and original `source_filename` (`source`).
- **Whitespace Normalization**: Normalizes line breaks, removes null characters, collapses extra horizontal spacing, and retains double newlines (`\n\n`) for paragraph separation.
- **Scanned Document Detection**: Detects documents where all pages lack extractable text (e.g. image-only scans) and raises `ScannedPdfError`, explicitly stating that OCR is not yet implemented.

### 3. Page-Bounded Deterministic Chunking (`src/chunking.py`)
- **Interface (`chunk_pages`)**: Accepts `pages`, `chunk_size`, `overlap`, and `deduplicate`.
- **Page Boundary Preservation**: Chunks are strictly confined within their originating page; content from adjacent pages is never merged, ensuring 100% unambiguous source citations.
- **Stable Identifiers**: Generates deterministic, content-hashed IDs (`<source>_p<page>_c<idx>_<hash>`) without leaking document text.
- **Dictionary Export**: `chunk.to_dict()` and `page.to_dict()` provide standard dictionaries for downstream RAG consumption.

---

## Supported PDF Behavior & File-Size Limits

- **Default Upload Limit**: 10 MB (configured via `MAX_UPLOAD_SIZE_MB` in `.env`).
- **Standard Text PDFs**: Fully supported. Text and page citations are accurately preserved.
- **Multi-Page Documents**: Fully supported. Empty pages within a multi-page document are handled gracefully without aborting extraction.
- **Password-Protected / Encrypted PDFs**: Rejected with `EncryptedPdfError`.
- **Scanned / Image-Only PDFs**: Detected and rejected with `ScannedPdfError` explaining that OCR is not implemented.

---

## Known Limitations

- **No OCR Support**: Only digital text embedded in PDFs can be extracted. Optical Character Recognition (OCR) for scanned images is not supported.
- **No Vector DB or LLM Integration Yet**: Embedding generation and vector search will be implemented in Phase 3.

---

## Next Up: Phase 3

In **Phase 3 (Gemini Embeddings & ChromaDB Indexing)**, we will:
- Integrate the official `google-genai` Python SDK.
- Configure Gemini `text-embedding-004` (or configured embedding model).
- Set up local persistent ChromaDB vector storage under `data/chromadb/`.
- Index document chunks with metadata and stable IDs for semantic retrieval.
