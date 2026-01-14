"""
Redis queue helpers for InsightForge (ARQ).

Why this file is named `if_queue.py`:
- Avoids collision with Python stdlib `queue`.

Responsibilities:
- API enqueues jobs by name with arguments (doc_id / insight_id).
- Worker consumes and runs pipeline functions.
"""

from __future__ import annotations

import os
import asyncio
from typing import Any, Optional

from arq import create_pool
from arq.connections import RedisSettings


def get_redis_settings() -> RedisSettings:
    dsn = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    settings = RedisSettings.from_dsn(dsn)
    # ARQ defaults: conn_timeout=1s, conn_retries=5, conn_retry_delay=1s.
    # Expose knobs so high parallelism doesn't create spurious connect failures.
    settings.conn_timeout = int(os.getenv("IF_REDIS_CONN_TIMEOUT_SECONDS", "4"))
    settings.conn_retries = int(os.getenv("IF_REDIS_CONN_RETRIES", str(settings.conn_retries)))
    settings.conn_retry_delay = int(os.getenv("IF_REDIS_CONN_RETRY_DELAY_SECONDS", str(settings.conn_retry_delay)))
    max_conns_raw = (os.getenv("IF_REDIS_MAX_CONNECTIONS") or "").strip()
    if max_conns_raw:
        settings.max_connections = int(max_conns_raw)
    return settings


_redis_pool = None
_redis_pool_lock = asyncio.Lock()


async def _get_pool():
    """
    Create (once) and reuse a single ARQ redis pool per API process.

    This avoids opening/closing a new redis connection pool for every request,
    which can overload redis during parallel uploads.
    """
    global _redis_pool
    if _redis_pool is not None:
        return _redis_pool
    async with _redis_pool_lock:
        if _redis_pool is None:
            _redis_pool = await create_pool(get_redis_settings())
    return _redis_pool


async def get_pool():
    """
    Public wrapper for obtaining the shared ARQ redis pool.
    Used by observability endpoints (e.g. worker health).
    """
    return await _get_pool()


async def enqueue_job(job_name: str, *args: Any, **kwargs: Any) -> Optional[str]:
    """
    Enqueue a job onto Redis via ARQ.

    Returns:
        ARQ job_id (string) if available. This is mostly for debugging; the main
        IDs the frontend should track are doc_id / insight_id stored in Supabase.
    """
    try:
        redis = await _get_pool()
        job = await redis.enqueue_job(job_name, *args, **kwargs)
        return job.job_id if job else None
    except Exception as e:
        # Handle Redis connection failures gracefully
        print(f"❌ Failed to enqueue job {job_name}: {str(e)}")
        raise ValueError(f"Failed to enqueue job: Redis connection error - {str(e)}")


async def enqueue_process_document(doc_id: str) -> Optional[str]:
    return await enqueue_job("if_process_document", doc_id)


async def enqueue_run_insight(insight_id: str) -> Optional[str]:
    return await enqueue_job("if_run_insight", insight_id)


