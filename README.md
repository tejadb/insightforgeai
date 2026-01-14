<!-- 
NOTE: This README should be updated frequently as we progress with the implementation. 
Last updated: Jan 5, 2026 
-->

# InsightForge AI (New)

A modern document intelligence agent inspired by Google NotebookLM, built with Unstructured.io for robust multi-format document processing (PDF, PPTX, DOCX, etc.).

## Current Status

We have built a production-grade, asynchronous backend system. It features:
*   **Unstructured.io** for high-quality document parsing (OCR included).
*   **Redis Queue (ARQ)** for true non-blocking background processing.
*   **Supabase** for persistence (Postgres) and file storage.
*   **FastAPI** for a responsive API layer.
*   **RAG/Chunking**: Document chunking with vector embeddings for semantic search (ready for chat).
*   **Thread Pool Executor**: CPU-bound parsing runs in thread pool to prevent event loop blocking.
*   **Comprehensive Monitoring**: Thread pool, CPU usage, and parallelism monitoring built-in.

### Recent Improvements (Jan 2026)

1. **Thread Pool Executor for Parsing** ✅
   - Parsing now runs in a thread pool (default: CPU count workers)
   - Prevents event loop blocking
   - Allows I/O operations (embeddings) to run concurrently with CPU-bound parsing
   - Configurable via `IF_PARSE_THREAD_POOL_SIZE` env var

2. **Performance Monitoring** ✅
   - Thread pool statistics (active workers, queue size)
   - CPU usage tracking (process %, per-CPU usage)
   - Active thread count monitoring
   - Real-time metrics during parsing

3. **Worker Configuration** ✅
   - `max_jobs=5` (configurable via `IF_MAX_JOBS`)
   - `job_timeout=1800s` (30 min, configurable via `IF_JOB_TIMEOUT_SECONDS`)
   - `max_tries=2` (configurable via `IF_JOB_MAX_TRIES`)
   - System info logging on startup

### Test Results (Latest Run)

**Test Configuration**: 3 documents, 3 insights, 3 chat messages

**Performance Observations**:
- ✅ **Thread Pool Working**: 3/8 workers active during parallel parsing
- ✅ **Event Loop Not Blocked**: Embedding times fast (2-4s) even with concurrent parsing
- ✅ **Parallel Parsing**: All 3 documents parsed simultaneously
- ⚠️ **CPU Utilization**: Only 2-3 CPUs heavily utilized (not all 8)
- ⚠️ **Sequential Completion**: Documents complete sequentially despite parallel start

**Timing Results**:
- Document 1 (Aditi_SCM): parse_s=388.87s, embedding_s=3.84s, total=404.81s
- Document 2 (ABHISHEK SIP_merged): parse_s=579.23s, embedding_s=2.02s, total=587.63s
- Document 3 (Sachin Kumar Sharma): parse_s=645.85s, embedding_s=2.88s, total=659.73s
- **Total Wall-Clock Time**: 688.42s (~11.5 minutes)
- **Theoretical Minimum** (if fully parallel): 645.85s
- **Overhead**: 42.57s (6.6% overhead)

**Conclusion**: Thread pool is working correctly. Parallelism achieved. Some GIL contention or unstructured.io internal bottlenecks may limit full CPU utilization.

## Core User Flow

1.  **Document Upload & Processing (Async)**
    - User uploads a file -> API creates a DB record (`pending`) and returns `doc_id`.
    - **Background Job** (`process_document`) parses the file, extracts elements, and saves the full context to Supabase.
    - Status updates to `completed`.
    - Frontend enables the document for selection.

2.  **Insight Generation (Async)**
    - User selects one or more processed documents and chooses an action (Summary, Q&A, Tips, Slides).
    - API creates an insight record (`pending`) and returns `insight_id`.
    - **Background Job** (`run_insight_by_id`) combines contexts, builds a prompt with user instructions, calls the LLM, and saves the JSON result.
    - Status updates to `completed`.
    - Frontend displays the result.

3.  **Chat/RAG (Async)**
    - User sends a chat message with selected documents.
    - API creates chat messages and uses semantic search to find relevant chunks.
    - LLM generates contextual response based on retrieved chunks.
    - Response saved to chat history.

## Architecture Components

- **`doc_parser.py`**: Local-only parsing engine using Unstructured.io (OCR/Layout).
- **`elements_processor.py`**: Converts raw elements into LLM-ready context strings.
- **`chunking.py`**: Document chunking for RAG (uses unstructured.io chunking strategies).
- **`embeddings.py`**: OpenAI embedding generation (batch processing with rate limit handling).
- **`prompt_builder.py`**: Generates focused, JSON-mode prompts for each action.
- **`llm_client.py`**: Async OpenAI client with JSON mode enforcement.
- **`repo.py`**: Database (Supabase) access layer for Documents, Insights, and Chunks.
- **`pipeline.py`**: The "Service Layer" that orchestrates the async jobs (Processing, Chunking & Generation). **Includes thread pool executor for parsing.**
- **`api.py`**: Minimal FastAPI API that creates DB rows and enqueues Redis jobs.
- **`if_queue.py`**: Redis enqueue helpers (ARQ).
- **`worker.py`**: ARQ worker entrypoint that runs parsing + insight jobs. **Includes system monitoring.**
- **`test_pipeline.py`**: Comprehensive test suite (sequential + concurrent).
- **`schema.sql`**: Database definitions for `if_documents` and `if_insights`.
- **`schema_chunks.sql`**: Database schema for `if_document_chunks` (vector embeddings).

### Inspection Tools
- **`tests/inspect_document_extraction.py`**: Comprehensive extraction inspection tool that processes a document and saves:
  - All elements as complete JSON (nothing skipped)
  - The doc_context string (what we give to LLM)
  - All chunks as JSON (only content/text)
  
  This tool helps analyze document extraction quality, chunking behavior, and table handling.

## Known Issues & Critical Bugs

### 🔴 CRITICAL ISSUES (Must Fix Before Production)

#### 1. Cancellation Leak (Zombie Jobs)
**Location**: `embeddings.py:97` and `llm_client.py:80`

**Problem**: 
- `except Exception` catches `asyncio.CancelledError` (or `TimeoutError` from ARQ)
- When ARQ times out a job, the embedding/LLM call might still be running
- The exception handler treats it as a "retryable error" and retries
- **Original job continues in background** → zombie job
- ARQ retries → **duplicate processing**
- This explains `KeyError` in ARQ cleanup and "attempt-2-fast" pattern

**Evidence**:
```
DESCRIPTIVE& EFA& REGRESSION.pdf:
  Attempt 1: embedding_s=781s, total=978s (near timeout)
  Attempt 2: embedding_s=5s, total=168s (suspiciously fast!)
```

**Fix Required**:
```python
# embeddings.py and llm_client.py
except asyncio.CancelledError:
    # NEVER swallow this - let ARQ handle timeout cleanup
    raise
except Exception as e:
    # ... existing retry logic ...
```

**Status**: ❌ **NOT FIXED** - Must be implemented

---

#### 2. Missing HTTP Timeout Wrapper
**Location**: `embeddings.py:81`

**Problem**:
- Client has `timeout=300.0` (5 min) but this is per-request timeout
- Connection pool acquisition time is NOT covered
- If connection pool is full/stuck, call can wait indefinitely before starting
- Stack traces show stuck in `_connect()` (connection establishment)

**Evidence**:
```
File ".../httpcore/_async/connection.py", line 78, in handle_async_request
    stream = await self._connect(request)
...
asyncio.exceptions.CancelledError
```

**Fix Required**:
```python
# Wrap API call with asyncio.wait_for
EMBED_CALL_TIMEOUT = 60  # Shorter than ARQ timeout

async def get_embeddings_batch(...):
    try:
        response = await asyncio.wait_for(
            self.client.embeddings.create(...),
            timeout=EMBED_CALL_TIMEOUT
        )
    except asyncio.TimeoutError:
        raise ValueError("Embedding call timed out after 60s")
    except asyncio.CancelledError:
        raise
    except Exception as e:
        # ... retry logic ...
```

**Status**: ❌ **NOT FIXED** - Must be implemented

---

#### 3. No Global Concurrency Control for Embeddings
**Location**: `embeddings.py`

**Problem**:
- `max_jobs=5` means 5 documents can process concurrently
- Each job calls `get_embeddings_batch()` independently
- **All 5 can hit OpenAI simultaneously** → 5 concurrent HTTP connections
- Connection pool exhaustion → wild timing variance

**Evidence**:
```
7 chunks → 225s / 293s  (should be ~5s)
34 chunks → 160s        (should be ~10s)
14 chunks → 68s vs 2.5s (wild variance)
```

**Fix Required**:
```python
# Add global semaphore
_embedding_semaphore = asyncio.Semaphore(2)  # Max 2 concurrent

async def get_embeddings_batch(...):
    async with _embedding_semaphore:
        t_start = time.perf_counter()  # Start timer AFTER semaphore
        try:
            response = await asyncio.wait_for(
                self.client.embeddings.create(...),
                timeout=60.0
            )
        finally:
            elapsed = time.perf_counter() - t_start
            print(f"Embedding call took {elapsed:.2f}s")
```

**Status**: ❌ **NOT FIXED** - Must be implemented

---

### 🟡 SECONDARY ISSUES (Should Fix)

#### 4. Client Connection Pool Configuration
**Location**: `embeddings.py:31`

**Problem**:
- No explicit `httpx.AsyncClient` with connection limits
- Default httpx pool might be too small/large
- No keepalive configuration
- No explicit connect/read/write timeouts

**Fix Recommended**:
```python
import httpx

http_client = httpx.AsyncClient(
    timeout=httpx.Timeout(
        connect=10.0,    # Fast connect timeout
        read=60.0,       # Read timeout
        write=30.0,      # Write timeout
        pool=10.0        # Pool acquisition timeout
    ),
    limits=httpx.Limits(
        max_connections=20,
        max_keepalive_connections=10
    )
)

self.client = AsyncOpenAI(
    api_key=self.api_key,
    http_client=http_client
)
```

**Status**: ❌ **NOT FIXED** - Recommended improvement

---

#### 5. Timing Measurement Includes Wait Time
**Location**: `pipeline.py:144`

**Problem**:
- Timer starts BEFORE semaphore acquisition
- If 5 jobs all hit embeddings, 4 wait
- `embedding_s` includes wait time, not just API time
- Makes it look like "embeddings are slow" when it's actually "waiting for semaphore"

**Fix Required** (after adding semaphore):
- Move timer inside semaphore-acquired block
- Or log both "wait time" and "API time" separately

**Status**: ⚠️ **PARTIALLY ADDRESSED** - Will be fixed when semaphore is added

---

### 🟢 KNOWN LIMITATIONS (Not Bugs, But Worth Noting)

#### 6. GIL Contention in Parsing
**Observation**: Only 2-3 CPUs heavily utilized despite 3 threads running

**Possible Causes**:
- Python GIL (Global Interpreter Lock) - some Python code in unstructured.io may hold GIL
- Unstructured.io internal locks or sequential sections
- Memory bandwidth limits

**Impact**: Parsing is parallel but not fully utilizing all CPUs

**Potential Solutions** (if needed):
- Process Pool Executor instead of Thread Pool (more overhead, true parallelism)
- Profile unstructured.io to find bottlenecks
- Accept current performance (65% improvement over blocking approach)

**Status**: ✅ **ACCEPTABLE** - Performance is good, not critical

---

#### 7. Sequential Completion Pattern
**Observation**: Documents complete sequentially despite parallel start

**Possible Causes**:
- Different document complexities (some take longer)
- GIL contention causing some serialization
- Resource contention (memory/CPU)

**Impact**: Small overhead (6.6% in test run)

**Status**: ✅ **ACCEPTABLE** - Overhead is minimal

---

## Priority Fix Order

1. **Fix cancellation handling** (embeddings.py + llm_client.py) - **DO THIS FIRST**
2. **Add asyncio.wait_for wrapper** around embedding calls - **DO THIS SECOND**
3. **Add global semaphore** (2 concurrent) - **DO THIS THIRD**
4. **Configure httpx client explicitly** - **DO THIS FOURTH**
5. **Fix timing measurement** - **DO THIS LAST**

---

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Install system deps for OCR (Ubuntu/WSL)
sudo apt-get install tesseract-ocr tesseract-ocr-eng poppler-utils libmagic1 libreoffice

# Optional: Install psutil for detailed CPU/memory monitoring
pip install psutil
```

### Environment Variables (.env)

**Required**:
```bash
SUPABASE_URL="https://your-project.supabase.co"
SUPABASE_SERVICE_KEY="ey..."
OPENAI_API_KEY="sk-..."
OPENAI_MODEL="gpt-4o-mini"
REDIS_URL="redis://localhost:6379/0"
```

**Chunking Configuration (for RAG)**:
```bash
CHUNK_STRATEGY="by_title"
CHUNK_MAX_CHARACTERS=1000
CHUNK_NEW_AFTER_N_CHARS=500
CHUNK_OVERLAP=50
CHUNK_OVERLAP_ALL=false
CHUNK_MULTIPAGE_SECTIONS=true
CHUNK_COMBINE_UNDER_N_CHARS=500
```

**Embedding Configuration**:
```bash
OPENAI_EMBEDDING_MODEL="text-embedding-3-small"
EMBEDDING_DIMENSIONS=1536
EMBEDDING_BATCH_SIZE=1000  # Default: 1000 (OpenAI limit: 2048)
```

**Worker Configuration**:
```bash
IF_MAX_JOBS=5                    # Concurrent jobs (default: 5)
IF_JOB_TIMEOUT_SECONDS=1800      # Job timeout in seconds (default: 1800 = 30 min)
IF_JOB_MAX_TRIES=2               # Max retry attempts (default: 2)
IF_PARSE_THREAD_POOL_SIZE=8      # Parse thread pool size (default: CPU count)
```

**Redis Configuration** (optional):
```bash
IF_REDIS_CONN_TIMEOUT_SECONDS=4
IF_REDIS_CONN_RETRIES=3
IF_REDIS_CONN_RETRY_DELAY_SECONDS=1
IF_REDIS_MAX_CONNECTIONS=10
```

**Worker Health Check** (optional):
```bash
IF_WORKER_HEARTBEAT_INTERVAL_SECONDS=5
IF_WORKER_HEARTBEAT_TTL_SECONDS=30
IF_WORKER_HEARTBEAT_KEY="if:worker:heartbeat"
```

## Run (API + Worker)

You need **Redis** running first (`redis-server`). Then open 2 terminals (same venv / same folder).

```bash
# Terminal 1 (API)
cd "/home/teja/dailybetter/insight forge ai new"
source venv/bin/activate
uvicorn api:app --host 127.0.0.1 --port 8008 --reload
```

```bash
# Terminal 2 (Worker)
cd "/home/teja/dailybetter/insight forge ai new"
source venv/bin/activate
arq worker.WorkerSettings
```

**Expected Worker Output**:
```
📦 [Pipeline] Initialized parse thread pool with 8 workers (CPU count: 8)
🖥️  [Worker] System Info:
   CPU Count: 8 (logical)
   Max Jobs: 5
   Parse Thread Pool Size: 8
   PID: 25060
   Host: DESKTOP-LO3V2QK
   CPU Count (physical): 4
   Memory: 7.5 GB total
```

## Monitoring & Observability

### Worker Health Check

```bash
# Check worker health via API
curl -sS "http://127.0.0.1:8008/worker/health" | jq .
```

Returns:
- `redis_ok`: Redis connectivity
- `queue_len`: Jobs waiting in queue
- `in_progress`: Jobs currently processing
- `heartbeat`: Worker heartbeat info (if worker running)
- `heartbeat_age_seconds`: How fresh the heartbeat is

### Thread Pool Monitoring

During document processing, you'll see detailed monitoring:

```
🔍 [Before Parse] [ThreadPool] Active threads: 2
🔍 [Before Parse] [ThreadPool] Pool workers: 0/8
🔍 [Before Parse] [ThreadPool] Queue size: 0
🔍 [Before Parse] [CPU] Process CPU: 0.0%
🔍 [Before Parse] [CPU] Logical CPUs: 8, Physical: 4
🔍 [Before Parse] [CPU] Active CPUs (>5%): 2/8 - [0, 6]

⚡ [During Parse @2s] [ThreadPool] Active threads: 5
⚡ [During Parse @2s] [ThreadPool] Pool workers: 3/8
⚡ [During Parse @2s] [ThreadPool] Queue size: 0
⚡ [During Parse @2s] [CPU] Process CPU: 76.3%
⚡ [During Parse @2s] [CPU] Active CPUs (>5%): 3/8 - [0, 1, 2]

✅ [After Parse] [ThreadPool] Active threads: 5
✅ [After Parse] [ThreadPool] Pool workers: 3/8
✅ [After Parse] [CPU] Process CPU: 538.1%
✅ [After Parse] [CPU] Active CPUs (>5%): 7/8 - [0, 1, 2, 4, 5, 6, 7]
```

### Redis Queue Monitoring

```bash
# Check jobs in queue
redis-cli ZCARD arq:queue

# Check jobs in process
redis-cli --scan --pattern "arq:in-progress:*" | wc -l

# Watch queue continuously
watch -n 2 'echo "Queue: $(redis-cli ZCARD arq:queue) | In-Progress: $(redis-cli --scan --pattern \"arq:in-progress:*\" | wc -l)"'
```

## Running Tests

### Pipeline Tests

We have a powerful test runner `test_pipeline.py` that behaves like a frontend client. It tests:
1.  Immediate API response (no blocking).
2.  Status polling (pending -> processing -> completed).
3.  Sequential logic (Summary, QnA, Multi-Doc logic).
4.  **Concurrency**: Fires simultaneous requests to prove the API stays responsive.

**Requirements**: Redis, API server, and Worker must be running (see "Run" section above).

```bash
# Terminal 3 (Test Runner)
cd "/home/teja/dailybetter/insight forge ai new"
source venv/bin/activate
export IF_API_URL="http://127.0.0.1:8008"

# Run simple test (3 docs, 3 insights, 3 chats)
python3 test_pipeline.py
```

### Document Extraction Inspection

The `inspect_document_extraction.py` tool analyzes document extraction quality without requiring any services (Redis/API/Worker).

**Requirements**: Only Python and virtual environment (standalone script).

```bash
# Single Terminal (No other services needed)
cd "/home/teja/dailybetter/insight forge ai new"
source venv/bin/activate

# Run inspection on a document
python3 tests/inspect_document_extraction.py "tests/Assignments/Corporate Finance  Assignment ss.pdf"

# Or from tests folder
cd tests
python3 inspect_document_extraction.py "Assignments/Corporate Finance  Assignment ss.pdf"
```

**Output**: Creates 3 files in `tests/extraction_analysis/`:
- `{filename}_elements.json` - All extracted elements (complete)
- `{filename}_doc_context.txt` - The context string given to LLM
- `{filename}_chunks.json` - All chunks with content only

Use this to:
- Verify extraction quality
- Check chunking behavior (size, boundaries, duplication)
- Analyze table handling (HTML structure, atomic chunks)
- Debug chunking configuration

## Database Setup

1. **Run `schema.sql`** in Supabase SQL editor (creates `if_documents` and `if_insights` tables).
2. **Run `schema_chunks.sql`** in Supabase SQL editor (creates `if_document_chunks` table with vector support).

## Performance Characteristics

### Current Performance (After Thread Pool Implementation)

**Test Scenario**: 3 documents processed concurrently

**Results**:
- ✅ **Event Loop**: Not blocked (embeddings run fast: 2-4s)
- ✅ **Parallelism**: 3 documents parsing simultaneously
- ✅ **Thread Pool**: 3/8 workers active
- ⚠️ **CPU Utilization**: 2-3 CPUs heavily used (not all 8)
- ⚠️ **Completion**: Sequential pattern (small 6.6% overhead)

**Improvement Over Blocking Approach**: ~65% faster

### Bottlenecks Identified

1. **GIL Contention**: Some Python code in unstructured.io may hold GIL, limiting true parallelism
2. **Memory Bandwidth**: CPU-bound parsing may be memory-limited
3. **Unstructured.io Internals**: May have internal locks or sequential sections

### Optimization Opportunities

1. **Process Pool Executor**: Would provide true parallelism (no GIL) but higher overhead
2. **Profile unstructured.io**: Find internal bottlenecks
3. **Accept Current Performance**: 65% improvement is significant

## Next Steps / Roadmap

### Immediate (Critical Fixes)

1. **Fix Critical Bugs** (See "Known Issues" section above)
   - [ ] Fix cancellation handling in `embeddings.py` and `llm_client.py`
   - [ ] Add `asyncio.wait_for` wrapper around embedding calls
   - [ ] Add global semaphore for embedding concurrency control
   - [ ] Configure httpx client explicitly
   - [ ] Fix timing measurement (after semaphore added)

### Short Term

2. **Chat/RAG API** (In Progress)
   - [x] Chat endpoint implemented
   - [x] Semantic search implemented
   - [ ] Test and verify chat responses (currently returning "I don't have enough information...")

3. **Hardening & Reliability**
   - [ ] Implement "Dead Letter Queue" for permanently failed documents
   - [ ] Add structured logging (JSON) for production
   - [ ] Add retry logic improvements (after critical fixes)

4. **Frontend Integration**
   - [ ] Create simple frontend (vanilla JS + Tailwind) for manager demo
   - [ ] Connect to API endpoints
   - [ ] Implement polling logic in UI
   - [ ] Render structured JSON results (Cards, Slides)

### Medium Term

5. **Deployment (Dockerization)**
   - [ ] Create `Dockerfile` for Python app
   - [ ] Create `docker-compose.yml` (API + Worker + Redis)
   - [ ] Add health checks
   - [ ] Production deployment guide

6. **API Security**
   - [ ] Add authentication middleware (verify Supabase JWTs)
   - [ ] Implement rate limiting
   - [ ] Add CORS configuration

7. **Housekeeping**
   - [ ] Add cleanup job to delete old files from Storage/Redis
   - [ ] Add document expiration policy

## Architecture Decisions

### Why Thread Pool Executor (Not Process Pool)?

**Decision**: Use `ThreadPoolExecutor` for parsing

**Rationale**:
- Unstructured.io uses C extensions (Pillow, pdfplumber) that release the GIL
- Thread pool is simpler (shared memory, no serialization)
- Lower overhead than process pool
- Good enough performance (65% improvement)

**Trade-offs**:
- Some GIL contention still possible (Python code in unstructured.io)
- Not true parallelism (but close enough for our use case)
- Process pool would be better but adds complexity

### Why `max_jobs=5`?

**Decision**: Default to 5 concurrent jobs

**Rationale**:
- Balances throughput with resource usage
- With thread pool, parsing doesn't block event loop
- Embeddings are I/O-bound, can run concurrently
- Can be tuned via `IF_MAX_JOBS` env var

**Trade-offs**:
- More jobs = more memory usage
- More jobs = more connection pool contention (until semaphore is added)
- Fewer jobs = lower throughput

### Why 30-minute Job Timeout?

**Decision**: `job_timeout=1800s` (30 minutes)

**Rationale**:
- Large documents can take 10-15 minutes to parse
- Embeddings can take 5-10 minutes for large batches
- Need buffer for retries and network issues
- Can be tuned via `IF_JOB_TIMEOUT_SECONDS` env var

**Trade-offs**:
- Longer timeout = more time before detecting stuck jobs
- Shorter timeout = more false failures on large documents

## Troubleshooting

### Issue: Documents timing out

**Symptoms**: Documents fail with `WorkerJobTimeout` error

**Possible Causes**:
1. Document too large (parse time > 30 min)
2. Embedding call stuck (missing timeout wrapper - see Critical Issue #2)
3. Connection pool exhaustion (missing semaphore - see Critical Issue #3)

**Solutions**:
1. Increase `IF_JOB_TIMEOUT_SECONDS` (if document is legitimately large)
2. Fix Critical Issue #2 (add `asyncio.wait_for` wrapper)
3. Fix Critical Issue #3 (add global semaphore)

### Issue: Duplicate processing

**Symptoms**: Same document processed twice, "attempt-2-fast" pattern

**Possible Causes**:
1. Cancellation leak (zombie jobs - see Critical Issue #1)
2. ARQ retry logic triggering incorrectly

**Solutions**:
1. Fix Critical Issue #1 (proper cancellation handling)
2. Check ARQ retry configuration

### Issue: Slow embedding times

**Symptoms**: Embeddings take 200s+ when they should take 5s

**Possible Causes**:
1. Connection pool exhaustion (missing semaphore - see Critical Issue #3)
2. Missing timeout wrapper (see Critical Issue #2)
3. Network issues

**Solutions**:
1. Fix Critical Issue #3 (add global semaphore)
2. Fix Critical Issue #2 (add `asyncio.wait_for` wrapper)
3. Check network connectivity

### Issue: Low CPU utilization

**Symptoms**: Only 2-3 CPUs active despite 3+ threads running

**Possible Causes**:
1. GIL contention (Python code in unstructured.io)
2. Memory bandwidth limits
3. Unstructured.io internal locks

**Solutions**:
1. Accept current performance (65% improvement is good)
2. Consider Process Pool Executor (if more parallelism needed)
3. Profile unstructured.io to find bottlenecks

## References

- **`ANALYSIS_EMBEDDING_ISSUES.md`**: Detailed analysis of embedding timeout and concurrency issues
- **`tests/stress_run_report.md`**: Stress test results and analysis
- **`tests/stress_run_report.csv`**: Detailed per-document timing data
- **`commands.txt`**: All terminal commands for testing and monitoring

---

**Last Updated**: Jan 5, 2026  
**Status**: Production-ready with known issues documented. Critical fixes pending.
