"""
Agent-style chat for InsightForge AI (tool-using).

Goal:
- Use the existing `if_chat_messages` table (store only user/assistant messages).
- Allow the model to call tools:
  1) retrieve_chunks(query, top_k)
  2) get_full_context(reason, max_chars)
  3) create_insight_job(action_type, title, instructions)

Design notes:
- We do NOT store intermediate tool calls as chat messages (keeps DB clean).
- We return tool usage details as metadata for observability/UI.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional, Tuple

from embeddings import get_embedding
from if_queue import enqueue_run_insight
from repo import InsightRepo


AGENT_MAX_STEPS = int(__import__("os").getenv("IF_AGENT_MAX_STEPS", "6"))
AGENT_MAX_TOOL_OUTPUT_CHARS = int(__import__("os").getenv("IF_AGENT_MAX_TOOL_OUTPUT_CHARS", "120000"))
AGENT_TOP_K_DEFAULT = int(__import__("os").getenv("IF_AGENT_TOP_K_CHUNKS", "5"))
AGENT_CHUNK_CHAR_LIMIT = int(__import__("os").getenv("IF_AGENT_CHUNK_CHAR_LIMIT", "2000"))
AGENT_FULL_CONTEXT_CHAR_LIMIT = int(__import__("os").getenv("IF_AGENT_FULL_CONTEXT_CHAR_LIMIT", "80000"))
AGENT_MAX_RETRIEVE_CALLS = int(__import__("os").getenv("IF_AGENT_MAX_RETRIEVE_CALLS", "5"))


def _json_dumps_safe(obj: Any) -> str:
    s = json.dumps(obj, ensure_ascii=False)
    if len(s) > AGENT_MAX_TOOL_OUTPUT_CHARS:
        return s[:AGENT_MAX_TOOL_OUTPUT_CHARS] + "…(truncated)"
    return s


def build_agent_system_prompt() -> str:
    return (
        "You are InsightForge AI Agent.\n"
        "You have tools to retrieve document context and to create background insight jobs.\n"
        "\n"
        "Rules:\n"
        "- Use ONLY information from the provided documents when answering questions.\n"
        "- For normal questions: call retrieve_chunks (you may call it up to 5 times total), then answer with citations like [Doc Title pX-Y].\n"
        "- Use get_full_context ONLY if retrieve_chunks is insufficient AND you truly need broad context (it is expensive).\n"
        "- For insight requests: if the user wants an artifact (summary/qna/interview_tips/slides), create an insight job.\n"
        "  - You may optionally call retrieve_chunks first to craft better job instructions aligned with the user's intent.\n"
        "  - After creating a job, clearly tell the user what you created and include the returned job id.\n"
        "- Do NOT claim something appears in multiple documents unless you have supporting citations from each document.\n"
        "- If information is missing, say so.\n"
    )


def build_agent_tools() -> List[Dict[str, Any]]:
    # OpenAI "tools" schema for chat.completions
    return [
        {
            "type": "function",
            "function": {
                "name": "retrieve_chunks",
                "description": "Retrieve the most relevant document chunks for a query from selected documents (or a subset).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "User question rewritten as a search query."},
                        "doc_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional subset of document IDs to search. If omitted, search all selected documents.",
                        },
                        "top_k": {"type": "integer", "description": "How many chunks to retrieve.", "default": AGENT_TOP_K_DEFAULT},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_full_context",
                "description": "Fetch the full precomputed document context for the selected documents (or a subset). Use only if necessary.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reason": {"type": "string", "description": "Why full context is required vs. chunks."},
                        "doc_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional subset of document IDs. If omitted, use all selected documents.",
                        },
                        "max_chars": {"type": "integer", "description": "Max characters of context to return.", "default": AGENT_FULL_CONTEXT_CHAR_LIMIT},
                    },
                    "required": ["reason"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "create_insight_job",
                "description": "Create an insight job (summary/qna/interview_tips/slides) for the selected documents.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action_type": {
                            "type": "string",
                            "enum": ["summary", "qna", "interview_tips", "slides"],
                            "description": "Which insight to create.",
                        },
                        "doc_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional subset of document IDs for the job. If omitted, use all selected documents.",
                        },
                        "title": {"type": "string", "description": "A short title for the job."},
                        "instructions": {"type": "string", "description": "Optional instructions based on user request (can be empty)."},
                    },
                    "required": ["action_type"],
                },
            },
        },
    ]


def _as_chat_messages(history: List[Dict[str, str]]) -> List[Dict[str, str]]:
    msgs: List[Dict[str, str]] = []
    for m in history:
        role = (m.get("role") or "user").strip().lower()
        if role not in {"user", "assistant"}:
            role = "user"
        content = (m.get("content") or "").strip()
        if not content:
            continue
        msgs.append({"role": role, "content": content})
    return msgs


async def run_agent_chat(
    *,
    llm: Any,
    repo: InsightRepo,
    user_id: str,
    user_message: str,
    document_ids: List[str],
    conversation_history: List[Dict[str, str]],
) -> Tuple[str, Dict[str, Any]]:
    """
    Runs a tool-using chat loop and returns (assistant_text, metadata).
    """
    started = time.time()
    tools = build_agent_tools()

    # Preload doc titles so the model can choose doc_ids and cite cleanly.
    try:
        doc_titles_by_id = repo.get_document_titles(document_ids)
    except Exception:
        doc_titles_by_id = {str(d): str(d) for d in (document_ids or [])}

    messages: List[Dict[str, Any]] = [{"role": "system", "content": build_agent_system_prompt()}]
    messages.extend(_as_chat_messages(conversation_history))
    messages.append(
        {
            "role": "user",
            "content": (
                f"USER_MESSAGE:\n{user_message.strip()}\n\n"
                "SELECTED_DOCUMENTS (id -> title):\n"
                + "\n".join([f"- {did}: {doc_titles_by_id.get(str(did), str(did))}" for did in document_ids])
                + "\n"
            ),
        }
    )

    tool_events: List[Dict[str, Any]] = []
    created_insights: List[Dict[str, Any]] = []
    retrieve_calls_used = 0

    # We keep looping while the model requests tool calls.
    for step in range(AGENT_MAX_STEPS):
        resp = await llm.client.chat.completions.create(
            model=getattr(llm, "default_model", None) or None,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.0,
            max_tokens=800,
        )
        if not resp.choices:
            raise ValueError("Agent LLM returned no choices")

        msg = resp.choices[0].message

        # Tool calls?
        tool_calls = getattr(msg, "tool_calls", None) or []
        if tool_calls:
            # Add the assistant message with tool calls so the model can continue the trace.
            messages.append(
                {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                        }
                        for tc in tool_calls
                    ],
                }
            )

            for tc in tool_calls:
                name = tc.function.name
                raw_args = tc.function.arguments or "{}"
                try:
                    args = json.loads(raw_args)
                except Exception:
                    args = {}

                if name == "retrieve_chunks":
                    if retrieve_calls_used >= AGENT_MAX_RETRIEVE_CALLS:
                        tool_out = {"error": f"retrieve_chunks call limit reached ({AGENT_MAX_RETRIEVE_CALLS})"}
                        tool_events.append({"tool": "retrieve_chunks", "error": "limit_reached"})
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": _json_dumps_safe(tool_out),
                            }
                        )
                        continue

                    q = str(args.get("query") or "").strip()
                    requested_doc_ids = args.get("doc_ids")
                    top_k = int(args.get("top_k") or AGENT_TOP_K_DEFAULT)
                    if not q:
                        tool_out = {"error": "query is required"}
                    else:
                        # Restrict tool access to selected documents only.
                        effective_doc_ids = list(document_ids)
                        if isinstance(requested_doc_ids, list) and requested_doc_ids:
                            requested = {str(d).strip() for d in requested_doc_ids if str(d).strip()}
                            effective_doc_ids = [d for d in document_ids if str(d) in requested]
                            if not effective_doc_ids:
                                effective_doc_ids = list(document_ids)

                        query_embedding = await get_embedding(q)
                        rows = repo.query_similar_chunks(query_embedding=query_embedding, doc_ids=effective_doc_ids, top_k=top_k)
                        retrieved_chunks_for_prompt: List[Dict[str, str]] = []
                        retrieved_chunks_for_metadata: List[Dict[str, Any]] = []
                        for c in (rows or []):
                            content = (c.get("content") or "").strip()
                            if not content:
                                continue
                            content_for_prompt = content[:AGENT_CHUNK_CHAR_LIMIT]
                            chunk_index = c.get("chunk_index")
                            doc_id = c.get("document_id")
                            page_start = c.get("page_start")
                            page_end = c.get("page_end")
                            page_part = ""
                            if page_start is not None and page_end is not None:
                                page_part = f" p{page_start}-{page_end}"
                            doc_id_str = str(doc_id) if doc_id is not None else ""
                            doc_title = (doc_titles_by_id.get(doc_id_str) or doc_id_str or "Document").strip()
                            # User-facing citations should be stable and friendly (no UUIDs/chunk indices)
                            ref = f"[{doc_title}{page_part}]"
                            retrieved_chunks_for_prompt.append({"ref": ref, "content": content_for_prompt})
                            retrieved_chunks_for_metadata.append(
                                {
                                    "chunk_id": str(c.get("id")) if c.get("id") is not None else None,
                                    "document_id": doc_id_str or None,
                                    "chunk_index": chunk_index,
                                    "similarity": c.get("similarity"),
                                    "page_start": page_start,
                                    "page_end": page_end,
                                    "section_heading": c.get("section_heading"),
                                    "doc_title": doc_title,
                                }
                            )
                        tool_out = {
                            "query": q,
                            "doc_ids": effective_doc_ids,
                            "top_k": top_k,
                            "chunks": retrieved_chunks_for_prompt,
                            "chunks_meta": retrieved_chunks_for_metadata,
                        }
                        retrieve_calls_used += 1
                        tool_events.append(
                            {
                                "tool": "retrieve_chunks",
                                "query": q,
                                "doc_ids": effective_doc_ids,
                                "top_k": top_k,
                                "returned": len(retrieved_chunks_for_prompt),
                                "retrieve_calls_used": retrieve_calls_used,
                            }
                        )

                elif name == "get_full_context":
                    reason = str(args.get("reason") or "").strip()
                    requested_doc_ids = args.get("doc_ids")
                    max_chars = int(args.get("max_chars") or AGENT_FULL_CONTEXT_CHAR_LIMIT)
                    effective_doc_ids = list(document_ids)
                    if isinstance(requested_doc_ids, list) and requested_doc_ids:
                        requested = {str(d).strip() for d in requested_doc_ids if str(d).strip()}
                        effective_doc_ids = [d for d in document_ids if str(d) in requested]
                        if not effective_doc_ids:
                            effective_doc_ids = list(document_ids)

                    contexts = repo.get_documents_context(effective_doc_ids)
                    # Match pipeline formatting: separator between docs.
                    joined = ("\n\n" + ("=" * 20) + "\n\n").join(contexts or [])
                    tool_out = {
                        "reason": reason,
                        "context_chars": len(joined),
                        "context": joined[:max_chars],
                        "truncated": len(joined) > max_chars,
                    }
                    tool_events.append({"tool": "get_full_context", "reason": reason, "doc_ids": effective_doc_ids, "max_chars": max_chars, "chars": len(joined)})

                elif name == "create_insight_job":
                    action_type = str(args.get("action_type") or "").strip()
                    requested_doc_ids = args.get("doc_ids")
                    title = str(args.get("title") or "").strip()
                    instructions = str(args.get("instructions") or "").strip()
                    if action_type not in {"summary", "qna", "interview_tips", "slides"}:
                        tool_out = {"error": f"Invalid action_type: {action_type}"}
                    else:
                        effective_doc_ids = list(document_ids)
                        if isinstance(requested_doc_ids, list) and requested_doc_ids:
                            requested = {str(d).strip() for d in requested_doc_ids if str(d).strip()}
                            effective_doc_ids = [d for d in document_ids if str(d) in requested]
                            if not effective_doc_ids:
                                effective_doc_ids = list(document_ids)

                        insight_id = repo.create_insight(
                            user_id=user_id,
                            action_type=action_type,
                            doc_ids=effective_doc_ids,
                            instructions=instructions,
                            title=title,
                        )
                        arq_job_id = await enqueue_run_insight(insight_id)
                        tool_out = {
                            "insight_id": insight_id,
                            "job_id": insight_id,
                            "status": "pending",
                            "arq_job_id": arq_job_id,
                            "action_type": action_type,
                            "doc_ids": effective_doc_ids,
                        }
                        created_insights.append(tool_out)
                        tool_events.append({"tool": "create_insight_job", "action_type": action_type, "insight_id": insight_id, "doc_ids": effective_doc_ids})

                else:
                    tool_out = {"error": f"Unknown tool: {name}"}

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": _json_dumps_safe(tool_out),
                    }
                )

            # Continue loop for next model step.
            continue

        # No tool calls: final assistant content
        assistant_text = (msg.content or "").strip()
        if not assistant_text:
            assistant_text = "I’m not sure how to respond based on the available information."

        elapsed_ms = int((time.time() - started) * 1000)
        job_id = None
        if created_insights:
            job_id = created_insights[0].get("insight_id") or created_insights[0].get("job_id")
        metadata = {
            "response_type": "insight_job" if created_insights else "chat",
            "job_id": job_id,
            "agent": {
                "steps": step + 1,
                "tool_events": tool_events,
                "created_insights": created_insights,
                "retrieve_calls_used": retrieve_calls_used,
            },
            "model_used": getattr(llm, "default_model", None),
            "response_time_ms": elapsed_ms,
        }
        return assistant_text, metadata

    # If we exit due to max steps, return a safe message.
    elapsed_ms = int((time.time() - started) * 1000)
    job_id = None
    if created_insights:
        job_id = created_insights[0].get("insight_id") or created_insights[0].get("job_id")
    metadata = {
        "response_type": "insight_job" if created_insights else "chat",
        "job_id": job_id,
        "agent": {
            "steps": AGENT_MAX_STEPS,
            "tool_events": tool_events,
            "created_insights": created_insights,
            "retrieve_calls_used": retrieve_calls_used,
            "stopped_reason": "max_steps",
        },
        "model_used": getattr(llm, "default_model", None),
        "response_time_ms": elapsed_ms,
    }
    return (
        "I started working on that, but hit my internal step limit. Please rephrase your request more specifically.",
        metadata,
    )

