# Enterprise Knowledge Copilot

A production-minded, beginner-friendly AI document assistant that enables users to upload enterprise PDF documents, ask questions, and receive grounded answers backed by precise citations (filename and page numbers).

Designed for resource-constrained laptops:
- **No Ollama, Docker, or local LLMs** required.
- Minimal RAM and disk usage.
- Uses lightweight local Python libraries and Google's free-tier Gemini API.

---

## Current Status: Phase 4 (Semantic Retrieval, Ranking, and Relevance Testing)

> [!NOTE]  
> We have completed **Phase 4**. The project now features robust semantic retrieval with configurable top-k ranking, distance thresholding (cosine distance filtering), document-specific source filtering, duplicate chunk suppression, and context formatting for LLM prompts. It also includes an automated evaluation module computing Recall@k, Precision@k, and Mean Reciprocal Rank (MRR) against a 12-query synthetic benchmark. All 109 automated unit tests pass offline with zero network calls.

---

## Project Structure

```text
enterprise-knowledge-copilot/
├── app.py                      # UI entry point (Planned for Phase 6)
├── requirements.txt            # Project dependencies (pypdf, dotenv, google-genai, chromadb, pytest)
├── .env.example                # Environment variable template
├── .gitignore                  # Git ignore rules (secrets, venvs, caches, vector db)
├── README.md                   # Beginner setup guide (this file)
├── src/
│   ├── __init__.py             # Package marker
│   ├── config.py               # Safe environment & config loader (retrieval settings)
│   ├── document_loader.py      # PDF text & metadata extraction
│   ├── chunking.py             # Deterministic chunking with page boundary preservation
│   ├── embeddings.py           # Gemini embeddings via google-genai SDK
│   ├── vector_store.py         # Persistent local ChromaDB store
│   ├── indexing.py             # End-to-end PDF indexing pipeline
│   ├── retrieval.py            # Semantic retrieval, thresholding & context formatting (Phase 4)
│   ├── evaluation.py           # Retrieval metrics: Recall@k, Precision@k, MRR (Phase 4)
│   └── rag_pipeline.py         # Grounded RAG pipeline (Planned for Phase 5)
├── tests/
│   ├── __init__.py
│   ├── fixtures/
│   │   └── eval_dataset.json   # 12 synthetic evaluation queries (Phase 4)
│   ├── test_config.py          # Config validation & secret masking tests (15 tests)
│   ├── test_document_loader.py # PDF validation & extraction tests (15 tests)
│   ├── test_chunking.py        # Chunking & overlap tests (25 tests)
│   ├── test_embeddings.py      # Embedding generation & retry tests (11 tests)
│   ├── test_vector_store.py    # ChromaDB operations & persistence tests (6 tests)
│   ├── test_indexing.py        # Pipeline & deduplication tests (5 tests)
│   ├── test_retrieval.py       # Semantic search, thresholding & filter tests (15 tests)
│   └── test_evaluation.py      # Metrics calculation & benchmark tests (16 tests)
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

### 4. Install Project Dependencies

Upgrade pip and install dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Configure Your Local `.env` File

Copy `.env.example` to `.env`:

```powershell
Copy-Item .env.example .env
```

Open `.env` in a text editor and configure your Gemini API key:
```dotenv
GEMINI_API_KEY=your_actual_gemini_api_key_here
GEMINI_EMBEDDING_MODEL=text-embedding-004
GEMINI_CHAT_MODEL=gemini-2.5-flash
CHROMA_PERSIST_DIRECTORY=data/chromadb
CHROMA_COLLECTION_NAME=enterprise_documents
```

> [!TIP]  
> You can obtain a free Gemini API key from [Google AI Studio](https://aistudio.google.com/).

### 6. Run the Automated Unit Test Suite

Run all unit tests using `pytest`:

```powershell
pytest -v
```

All 109 unit tests run **100% offline** using mocked API responses. **Running tests makes zero API calls and consumes zero quota.**

---

## Understanding the Core Concepts

### What are Embeddings?
An **embedding** is a way of translating human language (words, sentences, document chunks) into a dense list of numbers (a vector). For example, `text-embedding-004` turns a chunk of text into a list of 768 floating-point numbers. Sentences with similar meanings end up mathematically close to each other in this 768-dimensional space. This allows our Copilot to find relevant information based on **conceptual meaning**, not just exact keyword matches.

### Keyword Search vs. Semantic Retrieval
- **Keyword Search (e.g. grep, BM25)**: Matches exact words or substrings. If a policy document says *"Employees receive 20 days of paid annual leave"* and the user asks *"How much vacation time do I get?"*, keyword search fails because the word *"vacation"* never appears in the text.
- **Semantic Retrieval**: Encodes the underlying meaning into vector space. In `text-embedding-004`, *"vacation time"* and *"paid annual leave"* have nearly identical vector representations, enabling high-precision retrieval even with zero shared words.

### Ranking and Cosine Distance
ChromaDB calculates the distance between the query vector and document chunk vectors using **cosine distance**:
$$\text{Cosine Distance} = 1.0 - \text{Cosine Similarity}$$
- **Distance $\approx 0.0$**: Strong semantic similarity (exact concept match).
- **Distance $\approx 1.0$**: Orthogonal / unrelated concepts.
- **Distance $> 1.0$**: Opposing or diametrically distant concepts.

Results are returned ranked in ascending order by distance (closest first).

### Relevance Thresholding & Guardrails
A vector database will always return the mathematical top-k nearest neighbors, even if the user asks a completely irrelevant question (e.g. *"How do rockets work?"* asked against an HR handbook). Without guardrails, irrelevant chunks pollute the LLM's context window.

We address this with **Relevance Thresholding** (`RETRIEVAL_DISTANCE_THRESHOLD` or `max_distance`):
- Any chunk whose distance exceeds the threshold is discarded before entering the LLM prompt.
- If all chunks exceed the threshold, retrieval returns an empty list, allowing Phase 5 to cleanly report that no matching evidence was found.

### Retrieval Evaluation Metrics
To measure retrieval quality without manual inspection, Phase 4 implements an offline evaluation harness (`src/evaluation.py`):
1. **Recall@k**: Fraction of ground-truth relevant chunks that appear in the top-k results:
   $$\text{Recall@k} = \frac{|\text{Retrieved}_k \cap \text{Expected}|}{|\text{Expected}|}$$
2. **Precision@k**: Fraction of top-k retrieved chunks that are actually relevant:
   $$\text{Precision@k} = \frac{|\text{Retrieved}_k \cap \text{Expected}|}{|\text{Retrieved}_k|}$$
3. **Mean Reciprocal Rank (MRR)**: Evaluates ranking quality by measuring how high the first relevant chunk appears:
   $$\text{RR} = \frac{1}{\text{rank of first relevant chunk}}$$
   $$\text{MRR} = \frac{1}{N} \sum_{i=1}^N \text{RR}_i$$

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

### 4. Gemini Embeddings (`src/embeddings.py`)
- **Official SDK**: Built using Google's official `google-genai` Python library.
- **Task Types**: Uses `RETRIEVAL_DOCUMENT` for chunk vectors and `RETRIEVAL_QUERY` for search questions.
- **Bounded Exponential Backoff**: Automatically handles transient 429 / ResourceExhausted rate limits with retries and jitter.
- **Finite Vector Verification**: Validates returned vectors, rejecting NaN, Inf, or empty responses without fabricating data.

### 5. Persistent ChromaDB Vector Store (`src/vector_store.py`)
- **Explicit Embeddings**: Operates with `embedding_function=None` to ensure zero local model downloads.
- **Citation Metadata**: Preserves `source`, `page`, `chunk_id`, and `doc_hash` on every record.
- **Scoped Document Deletion**: Allows cleanly updating or deleting one document's chunks without touching other documents.

### 6. End-to-End Indexing Pipeline (`src/indexing.py`)
- **Unified Pipeline (`index_document`)**: PDF -> validation -> extraction -> chunking -> embedding -> ChromaDB.
- **Duplicate Detection**: Computes document-level content hashes; skips re-indexing unchanged documents (`skipped_unchanged`) to conserve API quota.
- **All-or-Nothing Replacement**: Computes all embeddings before touching ChromaDB. If embedding fails halfway, the previous index remains untouched.

### 7. Semantic Retrieval & Relevance Calibration (`src/retrieval.py` — Phase 4)
- **Top-k & Bounded Limits**: Configurable default (`RETRIEVAL_TOP_K=5`) with strict upper bound validation (`RETRIEVAL_MAX_TOP_K=50`).
- **Distance Filtering**: Rejection of irrelevant chunks via `max_distance` (or `RETRIEVAL_DISTANCE_THRESHOLD`).
- **Source Filtering**: Direct document-level filtering (`source_filter="handbook.pdf"`) supporting multi-document routing and cross-document segregation.
- **Deduplication**: Automatically suppresses duplicate chunk IDs returned by vector queries.
- **Context Window Formatting (`format_retrieval_context`)**: Formats retrieved passages with citation labels (`[Passage 1 | Source: ... | Page: ... | ID: ...]`) ready for LLM prompt augmentation.

### 8. Retrieval Evaluation Harness (`src/evaluation.py` — Phase 4)
- **Deterministic Metrics**: Computes Recall@k, Precision@k, Reciprocal Rank, and Mean Reciprocal Rank (MRR).
- **Synthetic Benchmark**: Includes a 12-query evaluation fixture (`tests/fixtures/eval_dataset.json`) covering exact matches, paraphrasing, cross-page facts, multi-document routing, and unanswerable queries.

---

## Manual Retrieval Verification (Optional)

Once you set a real `GEMINI_API_KEY` in `.env`, you can test semantic search and context formatting from PowerShell:

```powershell
python -c "
from src.retrieval import search_chunks, format_retrieval_context

# Search indexed chunks with distance filtering
hits = search_chunks('What are the company policies regarding remote work?', top_k=3, max_distance=0.6)
print(f'Retrieved {len(hits)} relevant passages:')
for h in hits:
    print(f'- Page {h[\"page\"]} ({h[\"source\"]}, dist={h[\"distance\"]:.3f}): {h[\"text\"][:80]}...')

# Format context as it will appear in the Phase 5 LLM prompt
context_block = format_retrieval_context(hits)
print('\n--- Formatted Context for LLM ---')
print(context_block)
"
```

---

## Free-Tier Quotas & Limitations

- **Gemini Free-Tier Rate Limits**: Free Gemini API keys have rate limits (typically 15 requests per minute for embedding models). Our built-in retry backoff handles momentary bursts, but very large multi-hundred-page PDFs should be uploaded in moderation.
- **Model Consistency**: Document chunks and search queries must use the same embedding model (`text-embedding-004`). If you change the model in `.env`, the existing collection should be rebuilt.
- **No OCR Support**: Only digital text embedded in PDFs can be extracted. Optical Character Recognition (OCR) for scanned images is not supported.

---

## Next Up: Phase 5

In **Phase 5 (Grounded Generation and Citations)**, we will:
- Integrate Gemini 2.5 Flash for question answering grounded strictly in retrieved context.
- Enforce strict citation formatting (`[Source: filename.pdf, Page: N]`).
- Implement honest fallback behavior when context is insufficient to prevent hallucination.
