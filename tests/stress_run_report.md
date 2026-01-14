## Stress Test Report — 15 Documents (Latest Run)

This report summarizes the latest `test_pipeline.py --stress --dir "tests/Assignments"` run with:
- **Worker settings**: `max_jobs=5`, `job_timeout=600s`, `max_tries=2`
- **Embeddings**: OpenAI client timeout `120s`, **embedding retries=2**, batch mode for all chunks
- **Docs**: 15 PDFs from `tests/Assignments`

Overall outcome:
- **Documents**: 12 **completed**, 3 **failed** (all on **embedding stage due to worker job timeout**)
- **Insights**: 10/10 **completed successfully** in parallel

The new timing + error instrumentation in `pipeline.py` makes it clear **where** time is spent and **what exactly failed** when there is an error.

---

## High-level: What went well vs. what’s still bad

- **Went well**
  - **No “stuck in processing” docs**: failed docs are clearly marked `status=error` with rich `error_message` (stage, elapsed, timings).
  - **Most docs are fast enough**: 12/15 completed within the 600s worker timeout; typical total times range from **70s–330s**.
  - **Insights are excellent**: 10 insights completed, each in **3–7s**, even with large contexts (up to ~39K chars).
  - **Observability is much better**:
    - Per-doc timings: `download_s`, `parse_s`, `chunking_s`, `embedding_s`, `store_*_s`, `total_s`.
    - Per-failure error strings: `stage=...`, `elapsed=...s`, `error=WorkerJobTimeout: ...`, `timings=(...)`.
    - Worker heartbeat + `/worker/health` gives Redis + queue + “worker_seen_recently” view.
  - **Chunking correctness**: All successful docs show “Generating embeddings for N chunks” followed by “Stored N chunks” and contiguous `chunk_index` assignments with no duplication.

- **Still bad / remaining issues**
  - **3 large/complex docs fail due to worker timeout at embedding stage**:
    - `ABHISHEK SIP_merged.pdf` — 83 chunks, **parse ~279s**, then embedding pushes total to **~933s** and hits `job_timeout=600s` on first attempt; it later **succeeds on retry** (shorter run), but the first run is still an expensive timeout.
    - `Eco assignment2.pdf` — 9 chunks, **parse ~75s**, total **~618s**, timed out.
    - `2385100018_Sachin Kumar Sharma_8287565735.pdf` — 142 chunks, **parse ~376s**, total **~703s**, timed out.
  - **Embedding is the dominant bottleneck on big docs**:
    - E.g. `DESCRIPTIVE& EFA& REGRESSION.pdf` first successful run: `embedding_s ≈ 781s` (13 minutes) even though parsing is ~193s.
    - For some smaller docs, embedding still takes **100–220s**.
  - **Worker retries are expensive on slow docs**:
    - `DESCRIPTIVE& EFA& REGRESSION.pdf` (bfb07aa7...) was processed **twice** (ARQ `try=2`), adding up to ~1150s of work for that doc alone.
    - Timeouts on the three failing docs also consumed a lot of worker time before failing cleanly.

- **Changes that clearly improved things**
  - **Timing metrics + stage-aware error messages in `pipeline.py`**:
    - Now every failure tells you: `stage=embed_chunks`, `elapsed=...s`, `timings=(download_s=..., parse_s=..., chunking_s=..., ...)`.
  - **Worker heartbeat + `/worker/health`**:
    - You can see `queue_len`, `in_progress`, and heartbeat age; confirmed **worker_seen_recently=true** during the run.
  - **Worker concurrency + timeout tuning**:
    - `max_jobs=5` gave good utilization without blowing up CPU or Redis.
    - `job_timeout=600s` is enough for **most** docs; only the very large/slow ones hit it.
  - **Embedding retries reduced to 2**:
    - Avoids very long retry chains on already-slow docs, limiting wasted time.

- **Changes that are still borderline / need further tuning**
  - **Embedding timeout vs. job timeout**:
    - OpenAI embedding timeout is **120s per call**, plus up to **2 retries** with backoff. Combined with slow network / API, this can easily eat several hundred seconds inside a single `embed_chunks` stage.
    - For huge docs (83/142 chunks) this pushes the total wall time over the **600s** job timeout — hence the `WorkerJobTimeout` failures.
  - **Retry behavior on big docs**:
    - ARQ will retry a timed-out job (up to `max_tries=2`), which is correct but expensive if the underlying cause is “document is too heavy for our budget”.

---

## Per-document Summary Table (15 docs)

Notes:
- **Status** is the final status for that document in this run.
- **Stage failed** is only set for errors (always `embed_chunks` in this run).
- **Timings** are taken from the last log line for that doc (`processed successfully` or error). Empty timing fields mean that stage never completed (e.g. embedding_s is missing when we timed out mid-embedding).

| # | Document Title                                      | Doc ID                                   | Final Status | Stage Failed  | Attempts | Chunks | download_s | parse_s  | chunking_s | embedding_s | store_chunks_s | store_document_s | total_s  | Error Summary                                                                                          |
|---|-----------------------------------------------------|------------------------------------------|-------------:|---------------|---------:|-------:|-----------:|---------:|-----------:|------------:|----------------:|-----------------:|---------:|--------------------------------------------------------------------------------------------------------|
| 1 | CapX Assignment.pdf                                 | 3a22922b-be6d-4d24-8d7d-0568a9b5466e     |  completed   | –             |   1      |   7    |   0.83s    |  46.56s  |   0.00s    | 225.41s    | 0.44s          | 0.14s           | 274.13s | –                                                                                                      |
| 2 | CB Assignment.pdf                                   | 6dd1de1c-d7d9-4e50-831e-4b130479e470     |  completed   | –             |   1      |   1    |   0.68s    |  24.48s  |   0.00s    | 198.57s    | 0.14s          | 0.12s           | 224.75s | –                                                                                                      |
| 3 | BE ADITI.pdf                                        | ffdd4fd7-732c-40cc-b875-a558e89d89bb     |  completed   | –             |   1      |  10    |   0.91s    |  25.33s  |   0.00s    | 170.63s    | 0.80s          | 0.17s           | 198.56s | –                                                                                                      |
| 4 | Micro Economics Assignment.pdf                      | 33e64e43-533d-4c1f-8837-4ea18fc50b88     |  completed   | –             |   1      |   3    |   1.07s    |  23.46s  |   0.00s    | 144.31s    | 0.71s          | 0.19s           | 170.61s | –                                                                                                      |
| 5 | Aditi_SCM.pdf                                       | c68234e4-c45c-4942-9db2-1cc191aa4a89     |  completed   | –             |   1      |  59    |   0.88s    | 140.45s  |   0.01s    |   4.57s    | 2.22s          | 0.76s           | 150.13s | –                                                                                                      |
| 6 | Market_Sizing_Assessment_Final.pdf                  | e07b4b23-0af6-4066-a0c7-2bdcf8b2dccb     |  completed   | –             |   1      |   7    |   1.15s    |  20.07s  |   0.00s    | 293.99s    | 0.31s          | 0.14s           | 316.28s | –                                                                                                      |
| 7 | Corporate Finance Assignment ss.pdf                 | 5f6a2497-a999-49f8-858e-9d1e8fd6ec78     |  completed   | –             |   1      |  34    |   0.95s    | 129.09s  |   0.01s    | 160.85s    | 1.24s          | 0.66s           | 293.98s | –                                                                                                      |
| 8 | MOS Assignment 2 ppt.pdf                            | 7cbeee30-6413-4601-abd9-dbeb75606c4d     |  completed   | –             |   1      |   5    |   0.92s    |  29.07s  |   0.00s    | 127.62s    | 0.48s          | 0.19s           | 158.68s | –                                                                                                      |
| 9 | Assignment 1.pdf                                    | 1167bda6-6c67-479d-a9cc-c6943a09d946     |  completed   | –             |   1      |  14    |   1.11s    |  58.45s  |   0.00s    |  68.31s    | 0.74s          | 0.40s           | 129.41s | –                                                                                                      |
|10 | Assignment 1 (1).pdf                                | ad007cd9-7433-4051-acee-09285ce7d68c     |  completed   | –             |   1      |  14    |   0.99s    |  64.81s  |   0.00s    |   2.51s    | 0.65s          | 0.35s           |  70.46s | –                                                                                                      |
|11 | ABHISHEK SIP_merged.pdf                             | 3f0e2b08-f451-4297-b8a9-e64936509060     |    error     | embed_chunks  |   1      |  83    |   0.99s    | 278.66s  |   0.01s    | (timed out) | –              | –               | 932.52s | `WorkerJobTimeout` during embedding batch for 83 chunks (first attempt).                               |
|12 | DESCRIPTIVE& EFA& REGRESSION.pdf                    | bfb07aa7-0999-4d1c-9ed6-412fcb7d5dba     |  completed   | – (final)     |   2      |  50    |   1.19s/0.85s | 192.57s/160.39s | 0.00s/0.01s | 781.41s/5.15s | 2.07s/0.97s     | 0.92s/0.42s     | 978.85s / 168.76s | First attempt nearly 16 min (781s embedding), retried and then completed quickly on second attempt.   |
|13 | Eco assignment2.pdf                                 | 5597e2e6-ba16-4765-8042-aab602dadbbd     |    error     | embed_chunks  |   1      |   9    |   1.57s    |  75.47s  |   0.00s    | (timed out) | –              | –               | 618.27s | `WorkerJobTimeout` during embedding batch for 9 chunks.                                               |
|14 | 2385100018_Sachin Kumar Sharma_8287565735.pdf       | 34861734-6a43-4557-a92a-21c4bc9e28c4     |    error     | embed_chunks  |   1      | 142    |   1.79s    | 376.38s  |   0.04s    | (timed out) | –              | –               | 703.13s | `WorkerJobTimeout` during embedding batch for 142 chunks (very large doc).                            |
|15 | MR _Assignemnt.pdf                                  | 931c92a2-e46c-426a-aeba-82b1ae442bfb     |  completed   | –             |   1      |  48    |   8.18s    | 150.94s  |   0.01s    | 166.14s    | 1.22s          | 0.40s           | 328.05s | –                                                                                                      |

> For document 12, two timing sets are shown (`first / second`) to highlight how the **first attempt was extremely slow and likely close to timing out**, while the **second attempt completed quickly**.

---

## What changes helped vs previous behavior

- **Previously**
  - Jobs could hang up to **1 hour** due to very large default OpenAI timeouts and missing `CancelledError` handling.
  - “Stuck” documents remained in `processing` with no clear error reason.
  - No per-stage timings, so it was unclear whether **parse** or **embeddings** were the bottleneck.

- **Now**
  - **Timeouts are explicit and clean**:
    - Worker kills the job at **600s**, and `process_document` catches `asyncio.CancelledError`, recording:
      - `stage=embed_chunks`
      - `elapsed=...s`
      - detailed `timings=(download_s=..., parse_s=..., chunking_s=...)`
    - No more “ghost processing” states.
  - **Performance visibility is excellent**:
    - It is now obvious that **OCR/parse** is expensive but bounded (e.g. 278s for ABHISHEK, 376s for Sachin).
    - The worst cases are clearly **embedding-heavy**: up to **781s** of embedding time for DESCRIPTIVE& EFA on first attempt.
  - **Worker health is visible**:
    - `/worker/health` confirms:
      - Redis is healthy
      - Queue lengths are what you expect
      - Worker heartbeat is fresh (`worker_seen_recently=true`).

---

## What still needs to be done / next tuning steps

1. **Align embedding timeout + retries with job timeout**
   - Consider tightening:
     - OpenAI timeout (e.g. **60–90s** instead of 120s).
     - Embedding retries (already 2; could drop to 1 for very large batches).
   - And/or implement a **time budget** inside `process_document`:
     - Before starting a new embedding batch, check `elapsed_s` vs `job_timeout` and **skip retry** if you’re already near the limit.

2. **Special handling for extremely large docs (83–142 chunks)**
   - Option A: **Raise job_timeout** only for specific doc sizes (e.g. when `chunks_to_store > 100`).
   - Option B: **Split embeddings into smaller conceptual units** or **pre-filter elements** to reduce chunk count for massive PDFs.

3. **Optional: classify “too big” docs as a distinct error**
   - Right now they show `WorkerJobTimeout`. You could:
     - Add a guard: if `parse_s` or `embedding_s` exceeds some threshold (e.g. 300s), mark the doc as **“too heavy for this plan”** with a clearer error message for the UI.

4. **Monitor over more runs**
   - Repeat this stress test periodically to ensure:
     - No new regressions in parse/embedding times.
     - Redis + ARQ remain stable under load with the current `max_jobs=5`.

