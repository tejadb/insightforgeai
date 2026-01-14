# InsightForge AI Components

Brief overview of the components created so far.

## 1. `doc_parser.py`
**Core Parsing Engine**
- Wrapper around `unstructured.partition.auto`.
- Handles file loading (path or bytes).
- Configures OCR and "hi_res" strategies.
- Returns raw `List[Element]`.

## 2. `elements_processor.py`
**Post-Processing & Formatting**
- Transforms raw `List[Element]` into usable text formats.
- `doc_context()`: Combines all elements into a single formatted string for LLM prompting.

## 3. `prompt_builder.py`
**Prompt Engineering**
- Generates specific prompts for Summary, Q&A, Tips, and Slides.
- Enforces JSON schemas for reliable output.
- Injects user instructions clearly.

## 4. `llm_client.py`
**AI Gateway**
- Async wrapper for OpenAI API.
- Handles `json_mode` enforcement.
- Standardizes error handling.

## 5. `repo.py`
**Data Access Layer (DAL)**
- Manages Supabase interactions.
- **Documents**: Upload/Download files, update status/content.
- **Insights**: Create jobs, save results.

## 6. `pipeline.py`
**Business Logic / Orchestrator**
- `process_document(doc_id)`: The background parsing job.
- `run_insight_by_id(insight_id)`: The background generation job (used by Redis worker).
- Connects Repo -> Parser -> LLM -> Repo.

## 7. `api.py`
**FastAPI API**
- Accepts uploads, creates DB rows (`pending`), enqueues Redis jobs.
- Provides polling endpoints for document/insight status.

## 8. `if_queue.py`
**Redis Enqueue Helpers (ARQ)**
- Small helper to enqueue `if_process_document(doc_id)` and `if_run_insight(insight_id)` jobs.

## 9. `worker.py`
**Redis Worker (ARQ)**
- Consumes Redis jobs and executes pipeline work.

## 10. `test_pipeline.py`
**Test Suite**
- Simulates a frontend client.
- Verifies full lifecycle: Upload -> Processing -> Insight -> Result.
- Tests concurrency and status polling.

## 11. `chunking.py`
**Document Chunking for RAG**
- Uses `unstructured.chunking.title.chunk_by_title()` for semantic chunking.
- Preserves document structure (chapters, sections, tables).
- Extracts metadata (page numbers, headings, element IDs).
- Configurable via environment variables.

## 12. `embeddings.py`
**Vector Embedding Generation**
- Async OpenAI embedding client.
- Batch processing with rate limit handling.
- Automatic retry with exponential backoff.
- Supports `text-embedding-3-small` (1536 dims) and `text-embedding-3-large` (3072 dims).

## 13. Database Schema
**Tables:**
- `if_documents`: Parsed documents with elements and context.
- `if_insights`: Generated insights (summary, Q&A, tips, slides).
- `if_document_chunks`: Chunked text with vector embeddings for semantic search.
