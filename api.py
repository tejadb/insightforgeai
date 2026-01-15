"""
InsightForge API (minimal).

This API is intentionally small:
- creates DB rows immediately (pending)
- enqueues Redis jobs (ARQ)
- provides status polling endpoints
"""

from __future__ import annotations

import uuid
import os
import time
import json
from typing import Any, List, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel, Field

from repo import InsightRepo
from if_queue import enqueue_process_document, enqueue_run_insight, get_pool as get_arq_pool
from embeddings import get_embedding
from llm_client import LLMClient
from prompt_builder import PromptBuilder


app = FastAPI(title="InsightForge AI (New)")
repo = InsightRepo()
llm = LLMClient()


# Valid action types for insights
VALID_ACTION_TYPES = ["summary", "qna", "interview_tips", "tips", "slides"]


class CreateInsightRequest(BaseModel):
    user_id: str
    action_type: str = Field(..., description="summary | qna | tips | interview_tips | slides")
    document_ids: List[str]
    instructions: str = ""
    title: Optional[str] = None


class ChatMessageRequest(BaseModel):
    user_id: str
    message: str
    document_ids: List[str]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/worker/health")
async def worker_health() -> dict[str, Any]:
    """
    Lightweight worker/queue observability endpoint.

    Notes:
    - This does NOT guarantee a worker will pick up jobs, but it provides:
      - Redis connectivity
      - queue length / in-progress count
      - worker heartbeat freshness (if worker.py is running with heartbeat enabled)
    """
    try:
        redis = await get_arq_pool()

        queue_len = await redis.zcard("arq:queue")

        # Count in-progress jobs via SCAN (avoids blocking Redis with KEYS).
        in_progress = 0
        try:
            async for _ in redis.scan_iter(match="arq:in-progress:*"):
                in_progress += 1
        except Exception:
            # Fallback for older redis client wrappers
            keys = await redis.keys("arq:in-progress:*")
            in_progress = len(keys or [])

        hb_key = os.getenv("IF_WORKER_HEARTBEAT_KEY", "if:worker:heartbeat")
        hb_raw = await redis.get(hb_key)
        heartbeat: Optional[dict[str, Any]] = None
        heartbeat_age_s: Optional[float] = None
        if hb_raw:
            if isinstance(hb_raw, (bytes, bytearray)):
                hb_raw = hb_raw.decode("utf-8", errors="replace")
            try:
                heartbeat = json.loads(hb_raw)
                ts = heartbeat.get("ts")
                if isinstance(ts, (int, float)):
                    heartbeat_age_s = max(0.0, time.time() - float(ts))
            except Exception:
                heartbeat = {"raw": hb_raw}

        ttl_s = int(os.getenv("IF_WORKER_HEARTBEAT_TTL_SECONDS", "30"))
        worker_seen_recently = heartbeat_age_s is not None and heartbeat_age_s <= float(ttl_s)

        return {
            "redis_ok": True,
            "queue_len": int(queue_len),
            "in_progress": int(in_progress),
            "heartbeat": heartbeat,
            "heartbeat_age_seconds": heartbeat_age_s,
            "worker_seen_recently": worker_seen_recently,
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Worker health check failed: {str(e)}")


@app.post("/documents")
async def create_document(
    user_id: str = Form(...),
    file: UploadFile = File(...),
) -> dict[str, Any]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="File must have a filename.")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file.")

    # Sanitize user_id and filename to prevent path injection
    import os
    safe_user_id = os.path.basename(user_id.replace("..", "").replace("/", "_").replace("\\", "_"))
    safe_filename = os.path.basename(file.filename.replace("..", "").replace("/", "_").replace("\\", "_")) if file.filename else "file"
    
    storage_path = f"if_uploads/{safe_user_id}/{uuid.uuid4()}_{safe_filename}"
    
    # Create document record first
    doc_id = repo.create_document_record(
        user_id=user_id,
        title=file.filename,
        storage_path=storage_path,
    )
    
    # Upload file after record is created (so we can clean up on failure)
    try:
        repo.upload_bytes(
            file_bytes,
            storage_path=storage_path,
            content_type=file.content_type or "application/octet-stream",
        )
    except Exception as upload_error:
        # If upload fails, try to clean up the document record
        try:
            repo.delete_document(doc_id)
        except:
            pass  # Best effort cleanup
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(upload_error)}")

    await enqueue_process_document(doc_id)

    return {
        "doc_id": doc_id,
        "status": "pending",
        "storage_path": storage_path,
        "title": file.filename,
    }


@app.get("/documents/{doc_id}")
def get_document(doc_id: str) -> dict[str, Any]:
    return repo.get_document(doc_id)


@app.post("/insights")
async def create_insight(req: CreateInsightRequest) -> dict[str, Any]:
    # Validate action_type
    if req.action_type not in VALID_ACTION_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid action_type: {req.action_type}. Must be one of: {', '.join(VALID_ACTION_TYPES)}"
        )
    
    insight_id = repo.create_insight(
        user_id=req.user_id,
        action_type=req.action_type,
        doc_ids=req.document_ids,
        instructions=req.instructions,
        title=req.title or "",
    )

    await enqueue_run_insight(insight_id)

    return {"insight_id": insight_id, "status": "pending"}


@app.get("/insights/{insight_id}")
def get_insight(insight_id: str) -> dict[str, Any]:
    return repo.get_insight(insight_id)


# ---- CHAT (single chat per user) ------------------------------------

CHAT_HISTORY_LIMIT = int(os.getenv("CHAT_HISTORY_LIMIT", "30"))
CHAT_TOP_K_CHUNKS = int(os.getenv("CHAT_TOP_K_CHUNKS", "5"))
CHAT_CHUNK_CHAR_LIMIT = int(os.getenv("CHAT_CHUNK_CHAR_LIMIT", "2000"))
CHAT_MAX_TOKENS = int(os.getenv("CHAT_MAX_TOKENS", "800"))


@app.post("/chat/message")
async def chat_message(req: ChatMessageRequest) -> dict[str, Any]:
    """
    Single-chat-per-user endpoint.

    Request includes:
    - user_id
    - message
    - document_ids (can vary per message)

    Behavior:
    - Store all messages in DB (no session table)
    - Include ONLY the last CHAT_HISTORY_LIMIT messages (excluding current user message)
      in the LLM prompt for continuity.
    """
    user_id = (req.user_id or "").strip()
    user_message = (req.message or "").strip()
    document_ids = req.document_ids or []

    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required")
    if not user_message:
        raise HTTPException(status_code=400, detail="message is required")
    if not document_ids:
        raise HTTPException(status_code=400, detail="document_ids is required")

    # Validate documents belong to user and are processed (completed)
    try:
        repo.validate_documents_for_user(
            user_id=user_id,
            document_ids=document_ids,
            require_completed=True,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Store user message (we store everything; UI can load full history later)
    try:
        user_msg_id = repo.create_chat_message(
            user_id=user_id,
            role="user",
            content=user_message,
            document_ids=document_ids,
            metadata={"message_length": len(user_message)},
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to store chat message: {str(e)}")

    # Build conversation history for LLM: last N messages excluding current user message
    history_rows = repo.get_chat_history(user_id=user_id, limit=CHAT_HISTORY_LIMIT + 1)
    # Most recent first; after we inserted above, the newest message should be the current user message.
    history_rows_excluding_current = history_rows[1:] if len(history_rows) > 0 else []
    # Convert to chronological order for prompt readability
    history_rows_excluding_current.reverse()
    conversation_history = [
        {"role": r.get("role", "user"), "content": r.get("content", "")}
        for r in history_rows_excluding_current
        if (r.get("content") or "").strip()
    ]

    # Retrieve relevant chunks (vector search)
    started = time.time()
    try:
        query_embedding = await get_embedding(user_message)
        similar_chunks = repo.query_similar_chunks(
            query_embedding=query_embedding,
            doc_ids=document_ids,
            top_k=CHAT_TOP_K_CHUNKS,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve context chunks: {str(e)}")

    retrieved_chunks_for_prompt: List[Dict[str, str]] = []
    retrieved_chunks_for_metadata: List[Dict[str, Any]] = []
    for c in (similar_chunks or []):
        chunk_id = c.get("id")
        doc_id = c.get("document_id")
        chunk_index = c.get("chunk_index")
        similarity = c.get("similarity")
        page_start = c.get("page_start")
        page_end = c.get("page_end")
        section_heading = c.get("section_heading")
        content = (c.get("content") or "").strip()
        if not content:
            continue

        content_for_prompt = content[:CHAT_CHUNK_CHAR_LIMIT]
        page_part = ""
        if page_start is not None and page_end is not None:
            page_part = f" p{page_start}-{page_end}"
        ref = f"[doc {doc_id} chunk {chunk_index}{page_part}]"

        retrieved_chunks_for_prompt.append({"ref": ref, "content": content_for_prompt})
        retrieved_chunks_for_metadata.append(
            {
                "chunk_id": str(chunk_id) if chunk_id is not None else None,
                "document_id": str(doc_id) if doc_id is not None else None,
                "chunk_index": chunk_index,
                "similarity": similarity,
                "page_start": page_start,
                "page_end": page_end,
                "section_heading": section_heading,
            }
        )

    # Prompt + LLM call
    messages = PromptBuilder.get_chat_prompt(
        user_message=user_message,
        conversation_history=conversation_history,
        retrieved_chunks=retrieved_chunks_for_prompt,
    )

    try:
        assistant_text = await llm.get_completion(
            messages,
            json_mode=False,
            max_tokens=CHAT_MAX_TOKENS,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM chat failed: {str(e)}")

    response_time_ms = int((time.time() - started) * 1000)

    # Store assistant message
    try:
        assistant_msg_id = repo.create_chat_message(
            user_id=user_id,
            role="assistant",
            content=assistant_text,
            document_ids=document_ids,
            metadata={
                "retrieved_chunks": retrieved_chunks_for_metadata,
                "chunks_count": len(retrieved_chunks_for_metadata),
                "history_messages_used": len(conversation_history),
                "history_limit": CHAT_HISTORY_LIMIT,
                "model_used": getattr(llm, "default_model", None),
                "response_time_ms": response_time_ms,
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to store assistant message: {str(e)}")

    return {
        "user_message_id": user_msg_id,
        "assistant_message_id": assistant_msg_id,
        "content": assistant_text,
        "metadata": {
            "retrieved_chunks": retrieved_chunks_for_metadata,
            "chunks_count": len(retrieved_chunks_for_metadata),
            "history_messages_used": len(conversation_history),
            "history_limit": CHAT_HISTORY_LIMIT,
            "response_time_ms": response_time_ms,
        },
    }


@app.get("/chat/history")
def chat_history(user_id: str, limit: int = 200) -> dict[str, Any]:
    """
    Returns chat history for a user.
    Frontend can paginate later; for now, you can request a larger limit.
    """
    user_id = (user_id or "").strip()
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required")

    # Clamp limit to avoid accidental huge payloads
    safe_limit = max(1, min(int(limit), 1000))
    try:
        rows = repo.get_chat_history(user_id=user_id, limit=safe_limit)
        # Return chronological order to UI
        rows.reverse()
        return {"messages": rows, "count": len(rows)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load chat history: {str(e)}")


