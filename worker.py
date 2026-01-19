"""
ARQ worker entrypoint.

Run:
  arq worker.WorkerSettings
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import socket
import time
from typing import Any

from arq.connections import RedisSettings

import pipeline


async def if_process_document(ctx: dict[str, Any], doc_id: str) -> None:
    await pipeline.process_document(doc_id)


async def if_run_insight(ctx: dict[str, Any], insight_id: str) -> None:
    await pipeline.run_insight_by_id(insight_id)


async def _heartbeat_loop(*, redis: Any, key: str, payload_base: dict[str, Any], interval_s: float, ttl_s: int) -> None:
    while True:
        payload = dict(payload_base)
        payload["ts"] = time.time()
        try:
            # arq uses redis-py asyncio client under the hood; `set(..., ex=ttl)` is supported.
            await redis.set(key, json.dumps(payload), ex=ttl_s)
        except Exception as e:
            # Best effort only; never crash worker because heartbeat failed.
            print(f"⚠️ [Worker] Heartbeat update failed: {e}")
        try:
            await asyncio.sleep(interval_s)
        except asyncio.CancelledError:
            # Expected on shutdown (Ctrl+C). Exit quietly.
            return


async def _on_startup(ctx: dict[str, Any]) -> None:
    """
    Record a lightweight heartbeat in Redis so the API can report if a worker is alive.
    """
    redis = ctx.get("redis")
    if redis is None:
        return

    # Log system information on startup
    cpu_count = os.cpu_count() or 1
    # Default to 1 for debugging; override via IF_MAX_JOBS env var.
    # max_jobs = int(os.getenv("IF_MAX_JOBS", "1"))
    # parse_pool_size = int(os.getenv("IF_PARSE_THREAD_POOL_SIZE", str(cpu_count)))
    
    # Reverting to default behavior (or comment out explicit overrides for now)
    max_jobs = int(os.getenv("IF_MAX_JOBS", str(cpu_count))) # Default to CPU count if not set
    # parse_pool_size = ... (let pipeline handle its own default)

    print(f"🖥️  [Worker] System Info:")
    print(f"   CPU Count: {cpu_count} (logical)")
    print(f"   Max Jobs: {max_jobs}")
    # print(f"   Parse Thread Pool Size: {parse_pool_size}")
    print(f"   PID: {os.getpid()}")
    print(f"   Host: {socket.gethostname()}")
    
    # Try to get CPU info if psutil is available
    try:
        import psutil
        cpu_physical = psutil.cpu_count(logical=False) or cpu_count
        print(f"   CPU Count (physical): {cpu_physical}")
        print(f"   Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB total")
    except ImportError:
        print(f"   (Install psutil for detailed CPU/memory info)")

    interval_s = float(os.getenv("IF_WORKER_HEARTBEAT_INTERVAL_SECONDS", "5"))
    ttl_s = int(os.getenv("IF_WORKER_HEARTBEAT_TTL_SECONDS", "30"))
    key = os.getenv("IF_WORKER_HEARTBEAT_KEY", "if:worker:heartbeat")

    payload_base = {
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "max_jobs": max_jobs,
        "job_timeout_seconds": int(os.getenv("IF_JOB_TIMEOUT_SECONDS", "1800")),
        "max_tries": int(os.getenv("IF_JOB_MAX_TRIES", "2")),
        "started_at": time.time(),
    }

    ctx["if_heartbeat_task"] = asyncio.create_task(
        _heartbeat_loop(redis=redis, key=key, payload_base=payload_base, interval_s=interval_s, ttl_s=ttl_s)
    )


async def _on_shutdown(ctx: dict[str, Any]) -> None:
    task = ctx.get("if_heartbeat_task")
    if task:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


class WorkerSettings:
    _redis_settings = RedisSettings.from_dsn(os.getenv("REDIS_URL", "redis://localhost:6379/0"))
    # Same knobs as API enqueue side; defaults are ARQ defaults.
    _redis_settings.conn_timeout = int(os.getenv("IF_REDIS_CONN_TIMEOUT_SECONDS", "4"))
    _redis_settings.conn_retries = int(os.getenv("IF_REDIS_CONN_RETRIES", str(_redis_settings.conn_retries)))
    _redis_settings.conn_retry_delay = int(os.getenv("IF_REDIS_CONN_RETRY_DELAY_SECONDS", str(_redis_settings.conn_retry_delay)))
    _max_conns_raw = (os.getenv("IF_REDIS_MAX_CONNECTIONS") or "").strip()
    if _max_conns_raw:
        _redis_settings.max_connections = int(_max_conns_raw)
    redis_settings = _redis_settings
    functions = [if_process_document, if_run_insight]

    # Concurrency control:
    # - `max_jobs` is the maximum number of jobs this worker process will run concurrently.
    # - With thread pool executor for parsing, multiple jobs can run without blocking the event loop.
    # Default to 1 for debugging; override via IF_MAX_JOBS env var.
    max_jobs = int(os.getenv("IF_MAX_JOBS", "1"))

    # Timeouts/retries tuned for OCR + LLM calls (can adjust later)
    # Increased to 30 min to handle very large documents with many chunks
    job_timeout = int(os.getenv("IF_JOB_TIMEOUT_SECONDS", "1800"))  # 30 min
    max_tries = int(os.getenv("IF_JOB_MAX_TRIES", "2"))

    on_startup = _on_startup
    on_shutdown = _on_shutdown


