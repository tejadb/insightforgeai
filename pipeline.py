"""
InsightForge Pipeline.

Connects the Repository, Parser, and LLM Client to execute document workflows.
Functions here are designed to be run asynchronously (e.g. background tasks).
"""

import asyncio
import json
import time
from typing import List, Dict, Any
import concurrent.futures
from functools import partial
import threading
import os

# Try to import psutil for CPU monitoring (optional)
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("⚠️ [Pipeline] psutil not available - CPU monitoring will be limited")

from repo import InsightRepo
from doc_parser import UnstructuredDocParser, ParseOptions
from elements_processor import doc_context
from prompt_builder import PromptBuilder
from llm_client import LLMClient
from chunking import chunk_document_elements, extract_chunk_metadata, chunk_to_text, get_chunking_config
from embeddings import get_embedding, get_embeddings_batch

# Initialize singletons (stateless)
repo = InsightRepo()
parser = UnstructuredDocParser(options=ParseOptions(strategy="hi_res"))
llm = LLMClient()

# Thread pool executor for CPU-bound parsing tasks
# This prevents blocking the event loop while allowing parallel parsing across CPUs
# Pool size defaults to CPU count, but can be overridden via IF_PARSE_THREAD_POOL_SIZE env var
_parse_pool_size = int(os.getenv("IF_PARSE_THREAD_POOL_SIZE", str(os.cpu_count() or 1)))
_parse_executor: concurrent.futures.ThreadPoolExecutor = concurrent.futures.ThreadPoolExecutor(
    max_workers=_parse_pool_size,
    thread_name_prefix="parse-worker"
)
print(f"📦 [Pipeline] Initialized parse thread pool with {_parse_pool_size} workers (CPU count: {os.cpu_count()})")


def _get_thread_pool_stats() -> dict[str, Any]:
    """Get current thread pool statistics for monitoring."""
    stats = {
        "active_threads": threading.active_count(),
        "pool_max_workers": _parse_pool_size,
        "pool_workers": len(_parse_executor._threads) if hasattr(_parse_executor, '_threads') else 0,
    }
    
    # Try to get queue size
    if hasattr(_parse_executor, '_work_queue'):
        try:
            stats["queue_size"] = _parse_executor._work_queue.qsize()
        except:
            stats["queue_size"] = "N/A"
    else:
        stats["queue_size"] = "N/A"
    
    # Get CPU usage (if psutil is available)
    if PSUTIL_AVAILABLE:
        try:
            process = psutil.Process(os.getpid())
            stats["cpu_percent"] = process.cpu_percent(interval=0.1)
            stats["cpu_count_logical"] = psutil.cpu_count(logical=True)
            stats["cpu_count_physical"] = psutil.cpu_count(logical=False)
            # Get per-CPU usage
            stats["cpu_per_cpu"] = psutil.cpu_percent(interval=0.1, percpu=True)
        except Exception as e:
            stats["cpu_percent"] = f"Error: {e}"
            stats["cpu_count_logical"] = "N/A"
            stats["cpu_count_physical"] = "N/A"
            stats["cpu_per_cpu"] = "N/A"
    else:
        stats["cpu_percent"] = "psutil not available"
        stats["cpu_count_logical"] = os.cpu_count() or "N/A"
        stats["cpu_count_physical"] = "N/A"
        stats["cpu_per_cpu"] = "N/A"
    
    return stats


def _format_thread_pool_stats(stats: dict[str, Any], prefix: str = "🔍") -> str:
    """Format thread pool statistics for logging."""
    lines = [
        f"{prefix} [ThreadPool] Active threads: {stats['active_threads']}",
        f"{prefix} [ThreadPool] Pool workers: {stats['pool_workers']}/{stats['pool_max_workers']}",
        f"{prefix} [ThreadPool] Queue size: {stats['queue_size']}",
    ]
    
    if isinstance(stats.get('cpu_percent'), (int, float)):
        lines.append(f"{prefix} [CPU] Process CPU: {stats['cpu_percent']:.1f}%")
        if isinstance(stats.get('cpu_count_logical'), int):
            lines.append(f"{prefix} [CPU] Logical CPUs: {stats['cpu_count_logical']}, Physical: {stats.get('cpu_count_physical', 'N/A')}")
        if isinstance(stats.get('cpu_per_cpu'), list):
            active_cpus = [i for i, pct in enumerate(stats['cpu_per_cpu']) if pct > 5.0]
            lines.append(f"{prefix} [CPU] Active CPUs (>5%): {len(active_cpus)}/{len(stats['cpu_per_cpu'])} - {active_cpus}")
    else:
        lines.append(f"{prefix} [CPU] {stats.get('cpu_percent', 'N/A')}")
    
    return "\n".join(lines)


def shutdown_parse_executor(wait: bool = True) -> None:
    """
    Gracefully shutdown the parse thread pool executor.
    
    Should be called during worker shutdown for clean resource cleanup.
    
    Args:
        wait: If True, wait for all pending tasks to complete. If False, cancel pending tasks.
    """
    global _parse_executor
    if _parse_executor is not None:
        print(f"🛑 [Pipeline] Shutting down parse thread pool executor (wait={wait})...")
        _parse_executor.shutdown(wait=wait)
        print(f"✅ [Pipeline] Parse thread pool executor shutdown complete")

def _timings_str(t: dict[str, float]) -> str:
    if not t:
        return ""
    parts = [f"{k}={v:.2f}s" for k, v in t.items()]
    return ", ".join(parts)

async def process_document(doc_id: str):
    """
    Background Task: Parse a document and save its context.
    
    1. Fetch doc metadata (storage path).
    2. Download file from Supabase storage.
    3. Parse with Unstructured (OCR/layout analysis).
    4. Save elements and combined context to DB.
    5. Update status ('completed' or 'error').
    """
    print(f"⚙️ [Pipeline] Processing document: {doc_id}")
    
    stage = "init"
    timings: dict[str, float] = {}
    t_total_start = time.perf_counter()
    title: str = ""
    storage_path: str = ""

    try:
        # 1. Update status to processing
        stage = "set_status_processing"
        repo.update_document_status(doc_id, "processing")
        
        # 2. Get document metadata
        stage = "get_document_metadata"
        doc = repo.get_document(doc_id)
        storage_path = doc["storage_path"]
        title = doc.get("title") or ""
        
        # 3. Download file
        stage = "download_file"
        print(f"📥 [Pipeline] Downloading {title} ({storage_path})...")
        t0 = time.perf_counter()
        file_bytes = repo.download_file(storage_path)
        timings["download_s"] = time.perf_counter() - t0
        
        # 4. Parse
        stage = "parse_document"
        print(f"🧠 [Pipeline] Parsing content for {title!r}...")
        # Rely on storage_path (filename) for detection; don't force filetype manually
        # to avoid conflicts in unstructured.partition
        # Run parsing in thread pool to avoid blocking the event loop
        # This allows I/O operations (embeddings) to run concurrently with CPU-bound parsing
        
        # Monitor thread pool before parsing
        stats_before = _get_thread_pool_stats()
        print(_format_thread_pool_stats(stats_before, prefix="🔍 [Before Parse]"))
        
        t0 = time.perf_counter()
        loop = asyncio.get_event_loop()
        # Use partial to pass filename as keyword argument
        parse_func = partial(parser.parse_bytes, file_bytes, filename=storage_path)
        
        # Submit to thread pool
        print(f"🚀 [Pipeline] Submitting parse task to thread pool (doc_id={doc_id[:8]}...)")
        future = loop.run_in_executor(_parse_executor, parse_func)
        
        # Start a background task to monitor during parsing (non-blocking)
        async def monitor_during_parse():
            await asyncio.sleep(2.0)  # Wait 2 seconds after start
            if not future.done():
                stats_during = _get_thread_pool_stats()
                print(_format_thread_pool_stats(stats_during, prefix="⚡ [During Parse @2s]"))
        
        monitor_task = asyncio.create_task(monitor_during_parse())
        
        # Wait for completion
        elements = await future
        timings["parse_s"] = time.perf_counter() - t0
        
        # Cancel monitor task if still running
        if not monitor_task.done():
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass
        
        # Monitor thread pool after parsing
        stats_after = _get_thread_pool_stats()
        print(_format_thread_pool_stats(stats_after, prefix="✅ [After Parse]"))
        print(f"⏱️  [Pipeline] Parse completed in {timings['parse_s']:.2f}s for {title!r}")
        
        # 5. Check if elements are empty
        if not elements:
            print(f"⚠️ [Pipeline] No elements extracted from document")
            stage = "store_empty_document"
            repo.update_document_content(doc_id, [], "")
            repo.update_document_status(doc_id, "completed")
            timings["total_s"] = time.perf_counter() - t_total_start
            print(f"✅ [Pipeline] Document {doc_id} processed (empty).")
            return
        
        # 6. Build context string (for existing insights flow)
        stage = "build_context"
        t0 = time.perf_counter()
        context_text = doc_context(elements, filename=title)
        timings["context_s"] = time.perf_counter() - t0
        
        # 7. Chunk document for RAG
        stage = "chunk_document"
        print(f"📦 [Pipeline] Chunking document for RAG...")
        try:
            # Delete existing chunks if document is being re-processed
            # If chunks don't exist, this is fine - just continue
            try:
                repo.delete_document_chunks(doc_id)
            except Exception as delete_error:
                # Log but don't fail - chunks might not exist yet
                print(f"⚠️ [Pipeline] Could not delete existing chunks (may not exist): {str(delete_error)}")
            
            chunking_config = get_chunking_config()
            t0 = time.perf_counter()
            chunked_elements = chunk_document_elements(elements, **chunking_config)
            timings["chunking_s"] = time.perf_counter() - t0
            
            if not chunked_elements:
                print(f"⚠️ [Pipeline] No chunks generated from document")
            else:
                # Prepare chunks for storage
                # Use separate counter for chunk_index to handle skipped empty chunks correctly
                chunks_to_store = []
                chunk_index = 0
                for chunk_elem in chunked_elements:
                    chunk_text = chunk_to_text(chunk_elem)
                    if not chunk_text.strip():
                        continue  # Skip empty chunks (don't increment chunk_index)
                    
                    metadata = extract_chunk_metadata(chunk_elem, elements)
                    
                    chunks_to_store.append({
                        "content": chunk_text,
                        "chunk_index": chunk_index,  # Sequential index, no gaps
                        "metadata": metadata
                    })
                    chunk_index += 1  # Only increment when chunk is actually stored
                
                # Generate embeddings in batches
                if chunks_to_store:
                    print(f"🔢 [Pipeline] Generating embeddings for {len(chunks_to_store)} chunks (batch mode)...")
                    
                    # Process in batches
                    # Default to 1000 to maximize throughput (OpenAI limit is 2048 inputs)
                    # This ensures virtually all documents are processed in 1 single API call.
                    batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", "1000"))
                    chunks_content = [c["content"] for c in chunks_to_store]
                    
                    total_chunks = len(chunks_content)
                    t_embed_start = time.perf_counter()
                    
                    for i in range(0, total_chunks, batch_size):
                        batch_slice = chunks_content[i : i + batch_size]
                        print(f"   ... processing batch {i // batch_size + 1}/{(total_chunks + batch_size - 1) // batch_size} ({len(batch_slice)} chunks)")
                        
                        try:
                            stage = "embed_chunks"
                            batch_embeddings = await get_embeddings_batch(batch_slice)
                            
                            # Assign embeddings back to the original chunk dictionaries
                            for j, embedding in enumerate(batch_embeddings):
                                chunks_to_store[i + j]["embedding"] = embedding
                                
                        except Exception as emb_err:
                            raise ValueError(
                                f"Embedding batch failed (error_type={type(emb_err).__name__}, doc_title={title!r}, "
                                f"batch_start={i}, batch_size={len(batch_slice)}, total_chunks={total_chunks}): {emb_err}"
                            )
                    timings["embedding_s"] = time.perf_counter() - t_embed_start
                    
                    # Store chunks in database
                    stage = "store_chunks"
                    t0 = time.perf_counter()
                    repo.store_document_chunks(doc_id, chunks_to_store)
                    timings["store_chunks_s"] = time.perf_counter() - t0
                    # Note: repo.store_document_chunks also prints a success line. This line is the pipeline-level summary.
                    print(f"✅ [Pipeline] Stored {len(chunks_to_store)} chunks with embeddings")
                else:
                    print(f"⚠️ [Pipeline] No valid chunks to store")
                
        except Exception as chunk_error:
            # Chunking/embedding failures are now fatal - document processing fails
            raise ValueError(f"stage={stage} chunking/embedding failed: {str(chunk_error)}")
        
        # 8. Save to DB (existing flow)
        stage = "store_document_content"
        t0 = time.perf_counter()
        repo.update_document_content(doc_id, elements, context_text)
        timings["store_document_s"] = time.perf_counter() - t0
        timings["total_s"] = time.perf_counter() - t_total_start
        print(f"✅ [Pipeline] Document {doc_id} processed successfully. ({_timings_str(timings)})")
        
    except Exception as e:
        elapsed_s = time.perf_counter() - t_total_start
        error_msg = (
            f"stage={stage} elapsed={elapsed_s:.2f}s doc_title={title!r} storage_path={storage_path!r} "
            f"error={type(e).__name__}: {str(e)} timings=({_timings_str(timings)})"
        )
        print(f"❌ [Pipeline] Processing failed for {doc_id}: {error_msg}")
        repo.update_document_status(doc_id, "error", error_message=error_msg)
        raise
    except asyncio.CancelledError:
        elapsed_s = time.perf_counter() - t_total_start
        error_msg = (
            f"stage={stage} elapsed={elapsed_s:.2f}s doc_title={title!r} storage_path={storage_path!r} "
            f"error=WorkerJobTimeout: Job timed out or cancelled by worker timings=({_timings_str(timings)})"
        )
        print(f"❌ [Pipeline] Processing cancelled for {doc_id}: {error_msg}")
        try:
            repo.update_document_status(doc_id, "error", error_message=error_msg)
        except:
            pass
        raise


async def run_insight(
    user_id: str,
    action_type: str,
    doc_ids: List[str],
    instructions: str
) -> Dict[str, Any]:
    """
    Legacy convenience method (synchronous orchestration style).

    NOTE: With Redis queue + API, the industrial flow is:
    - API creates an `if_insights` row (status='pending') and returns insight_id immediately
    - Worker runs `run_insight_by_id(insight_id)`

    This function remains for local/manual usage, but the queue worker should use
    `run_insight_by_id`.
    """
    print(
        f"🚀 [Pipeline] (legacy) Starting Insight Job: {action_type} for {len(doc_ids)} docs"
    )
    insight_id = repo.create_insight(
        user_id=user_id,
        action_type=action_type,
        doc_ids=doc_ids,
        instructions=instructions,
    )
    await run_insight_by_id(insight_id)
    insight = repo.get_insight(insight_id)
    return {"insight_id": insight_id, "status": insight["status"], "result": insight.get("result", {})}


def _normalize_action_type(action_type: str) -> str:
    # Back-compat alias
    if action_type == "tips":
        return "interview_tips"
    return action_type


def _build_prompt_messages(action_type: str, full_context: str, instructions: str) -> list[dict[str, str]]:
    at = _normalize_action_type(action_type)
    if at == "summary":
        return PromptBuilder.get_summary_prompt(full_context, instructions)
    if at == "qna":
        return PromptBuilder.get_qna_prompt(full_context, user_instructions=instructions)
    if at == "interview_tips":
        return PromptBuilder.get_interview_tips_prompt(full_context, instructions)
    if at == "slides":
        return PromptBuilder.get_slides_prompt(full_context, user_instructions=instructions)
    raise ValueError(f"Unknown action type: {action_type}")


async def run_insight_by_id(insight_id: str) -> Dict[str, Any]:
    """
    Background Job (Redis worker): generate an insight for a pre-created `if_insights` row.

    Flow:
      1) load insight row (user_id, action_type, document_ids, instructions)
      2) mark status='processing'
      3) load document contexts (must be completed)
      4) build prompt, call LLM (json_mode)
      5) save result, status='completed' (or 'error')
    """
    insight = repo.get_insight(insight_id)
    action_type = insight["action_type"]
    doc_ids = insight["document_ids"]
    instructions = insight.get("instructions") or ""

    print(
        f"🚀 [Pipeline] Starting Insight Job: {action_type} for {len(doc_ids)} docs (insight_id={insight_id})"
    )

    try:
        stage = "set_status_processing"
        t_total_start = time.perf_counter()
        timings: dict[str, float] = {}
        repo.update_insight_status(insight_id, "processing")

        stage = "load_contexts"
        t0 = time.perf_counter()
        contexts = repo.get_documents_context(doc_ids)
        timings["load_contexts_s"] = time.perf_counter() - t0
        if not contexts:
            raise ValueError("No processed document contexts found. Ensure documents are processed.")

        full_context = "\n\n" + ("=" * 20) + "\n\n".join(contexts)

        stage = "build_prompt"
        t0 = time.perf_counter()
        messages = _build_prompt_messages(action_type, full_context, instructions)
        timings["build_prompt_s"] = time.perf_counter() - t0

        print(f"🤖 [Pipeline] Sending to LLM ({len(full_context)} chars context)...")
        stage = "llm_call"
        t0 = time.perf_counter()
        response_str = await llm.get_completion(messages, json_mode=True)
        timings["llm_s"] = time.perf_counter() - t0
        
        # Parse JSON with error handling for malformed responses
        try:
            stage = "json_parse"
            # Try to extract JSON from markdown code fences if present
            import re
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_str, re.DOTALL)
            if json_match:
                response_str = json_match.group(1)
            
            t0 = time.perf_counter()
            result_json = json.loads(response_str)
            timings["json_parse_s"] = time.perf_counter() - t0
        except json.JSONDecodeError as json_error:
            # Log the raw response for debugging
            print(f"❌ [Pipeline] Invalid JSON response from LLM: {str(json_error)}")
            print(f"Raw response (first 500 chars): {response_str[:500]}")
            raise ValueError(f"LLM returned invalid JSON: {str(json_error)}")

        stage = "store_result"
        t0 = time.perf_counter()
        repo.update_insight_result(
            insight_id=insight_id,
            result=result_json,
            status="completed",
        )
        timings["store_result_s"] = time.perf_counter() - t0
        timings["total_s"] = time.perf_counter() - t_total_start
        print(f"✅ [Pipeline] Insight {insight_id} completed.")

        return {"insight_id": insight_id, "status": "completed", "result": result_json}

    except Exception as e:
        elapsed_s = time.perf_counter() - t_total_start if "t_total_start" in locals() else None
        timing_str = _timings_str(timings) if "timings" in locals() else ""
        err = f"stage={locals().get('stage','unknown')} elapsed={elapsed_s:.2f}s error={str(e)} timings=({timing_str})" if elapsed_s is not None else str(e)
        print(f"❌ [Pipeline] Insight failed: {err}")
        repo.update_insight_result(
            insight_id=insight_id,
            result={},
            status="error",
            error_message=err,
        )
        raise

# Helpers for specific actions (wrappers around run_insight)

async def generate_summary(user_id: str, doc_ids: List[str], instructions: str = ""):
    return await run_insight(user_id, "summary", doc_ids, instructions)

async def generate_qna(user_id: str, doc_ids: List[str], instructions: str = ""):
    return await run_insight(user_id, "qna", doc_ids, instructions)

async def generate_interview_tips(user_id: str, doc_ids: List[str], instructions: str = ""):
    return await run_insight(user_id, "interview_tips", doc_ids, instructions)

async def generate_slides(user_id: str, doc_ids: List[str], instructions: str = ""):
    return await run_insight(user_id, "slides", doc_ids, instructions)

