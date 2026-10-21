# Enterprise Knowledge Copilot

A production-minded, beginner-friendly AI document assistant that enables users to upload enterprise PDF documents, ask questions, and receive grounded answers backed by precise citations (filename and page numbers).

Designed for resource-constrained laptops:
- **No Ollama, Docker, or local LLMs** required.
- Minimal RAM and disk usage.
- Uses lightweight local Python libraries and Google's free-tier Gemini API.

---

## Current Status: Phase 1 (Foundation Setup)

> [!NOTE]  
> We are currently in **Phase 1**. Only the core foundation, PDF text extraction, deterministic text chunking, configuration validation, and unit test suites are implemented. Future features such as Gemini API embeddings, ChromaDB, Streamlit UI, FastAPI, and LangGraph will be introduced incrementally in upcoming phases.

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

## What Was Built in Phase 1

1. **Configuration Loader (`src/config.py`)**:
   - Safe parsing of environment variables.
   - Strict validation of chunk size and overlap boundaries.
   - Secret masking (`repr(config)` redacts API keys).
   - Zero network dependencies for running tests.

2. **PDF Extraction (`src/document_loader.py`)**:
   - Page-by-page text extraction using `pypdf`.
   - Normalization of newlines and whitespace.
   - Preserves citation metadata: page numbers (1-indexed) and source filenames.
   - Robust error handling for non-PDFs, empty files, and corrupt files.

3. **Deterministic Chunking (`src/chunking.py`)**:
   - Sliding-window text chunking with configurable overlap.
   - Input validation (e.g., rejects `chunk_overlap >= chunk_size` or negative values).
   - Generates deterministic, hash-based chunk IDs.
   - Supports duplicate chunk filtering across repeated headers/footers.

4. **Comprehensive Test Suite (`tests/`)**:
   - 100% offline unit tests covering configuration, chunking, and PDF loading.

---

## Next Up: Phase 2

In **Phase 2**, we will implement:
- Document upload validation and sanity checks.
- Document extraction verification with sample enterprise PDFs.
- Handling of edge cases (multi-column text, scanned pages without OCR, password-protected PDFs).
