"""
Prompt Builder for InsightForge AI.

Short, focused prompts for each action (summary, Q&A, tips, slides).
Each builder accepts:
- document context (already pre-processed)
- optional user instructions (free text)

All prompts are designed to work well with JSON mode for reliable parsing.
"""

import os
from typing import List, Dict

# Context truncation limit (configurable via env var, default 80000)
CONTEXT_TRUNCATION_LIMIT = int(os.getenv("CONTEXT_TRUNCATION_LIMIT", "80000"))


class PromptBuilder:
    """
    Generates chat messages for specific analysis tasks.
    Returns OpenAI-compatible message lists: [{"role": "system", "content": ...}, ...]
    """

    @staticmethod
    def _system_prompt() -> str:
        """Base system persona shared by all actions."""
        return (
            "You are InsightForge AI, an expert document analysis assistant. "
            "You must use ONLY the provided document context. "
            "If information is missing, say so instead of guessing."
        )

    # ---- SUMMARY -----------------------------------------------------

    @staticmethod
    def get_summary_prompt(
        doc_context: str,
        user_instructions: str = "",
    ) -> List[Dict[str, str]]:
        """
        Prompt for generating a concise structured summary.

        JSON output schema:
        {
          "overview": "2–3 sentence overview",
          "key_points": ["short bullet 1", "short bullet 2", ...]
        }
        """
        context_snippet = doc_context[:CONTEXT_TRUNCATION_LIMIT]
        if len(doc_context) > CONTEXT_TRUNCATION_LIMIT:
            print(f"⚠️ Context truncated from {len(doc_context)} to {CONTEXT_TRUNCATION_LIMIT} characters")

        user_content = f"""
You will receive content from one or more documents.
Create a concise summary as JSON with very short, high-signal text.

USER_INSTRUCTIONS (optional, may be empty):
{user_instructions.strip() or "None"}

DOCUMENT_CONTEXT:
{context_snippet}

RESPONSE FORMAT (valid JSON only):
{{
  "overview": "2-3 sentence plain-language overview of the document(s).",
  "key_points": [
    "short key point 1 (max 2 lines)",
    "short key point 2",
    "short key point 3"
  ]
}}

Rules:
- Keep each string short (max 2-3 lines).
- Do NOT add fields other than overview and key_points.
- Do NOT wrap the JSON in markdown fences.
"""
        return [
            {"role": "system", "content": PromptBuilder._system_prompt()},
            {"role": "user", "content": user_content},
        ]

    # ---- Q&A ---------------------------------------------------------

    @staticmethod
    def get_qna_prompt(
        doc_context: str,
        num_questions: int = 5,
        user_instructions: str = "",
    ) -> List[Dict[str, str]]:
        """
        Prompt for generating practice Q&A.

        JSON output schema:
        {
          "qna": [
            {"question": "...", "answer": "..."}
          ]
        }
        """
        context_snippet = doc_context[:CONTEXT_TRUNCATION_LIMIT]
        if len(doc_context) > CONTEXT_TRUNCATION_LIMIT:
            print(f"⚠️ Context truncated from {len(doc_context)} to {CONTEXT_TRUNCATION_LIMIT} characters")

        user_content = f"""
You will create {num_questions} interview-style Q&A pairs about this document.

USER_INSTRUCTIONS (optional, may be empty):
{user_instructions.strip() or "None"}

DOCUMENT_CONTEXT:
{context_snippet}

RESPONSE FORMAT (valid JSON only):
{{
  "qna": [
    {{
      "question": "short question text (1 line)",
      "answer": "short but complete answer (2-3 lines)"
    }}
  ]
}}

Rules:
- Return EXACTLY {num_questions} items in the qna array where possible.
- Each string must be short (max 2-3 lines).
- Do NOT add extra top-level fields.
- Do NOT wrap the JSON in markdown fences.
"""
        return [
            {"role": "system", "content": PromptBuilder._system_prompt()},
            {"role": "user", "content": user_content},
        ]

    # ---- INTERVIEW TIPS ----------------------------------------------

    @staticmethod
    def get_interview_tips_prompt(
        doc_context: str,
        user_instructions: str = "",
    ) -> List[Dict[str, str]]:
        """
        Prompt for generating short interview tips.

        JSON output schema:
        {
          "tips": ["short tip 1", "short tip 2", ...]
        }
        """
        context_snippet = doc_context[:CONTEXT_TRUNCATION_LIMIT]
        if len(doc_context) > CONTEXT_TRUNCATION_LIMIT:
            print(f"⚠️ Context truncated from {len(doc_context)} to {CONTEXT_TRUNCATION_LIMIT} characters")

        user_content = f"""
Based on this document, generate practical interview tips as JSON.
Assume the user needs to explain or defend this work in an interview.

USER_INSTRUCTIONS (optional, may be empty):
{user_instructions.strip() or "None"}

DOCUMENT_CONTEXT:
{context_snippet}

RESPONSE FORMAT (valid JSON only):
{{
  "tips": [
    "short, actionable tip 1 (max 2 lines)",
    "short, actionable tip 2",
    "short, actionable tip 3"
  ]
}}

Rules:
- Each tip should be concrete and specific to the document.
- 3–7 tips is ideal.
- Do NOT add extra fields.
- Do NOT wrap the JSON in markdown fences.
"""
        return [
            {"role": "system", "content": PromptBuilder._system_prompt()},
            {"role": "user", "content": user_content},
        ]

    # ---- SLIDE DECK --------------------------------------------------

    @staticmethod
    def get_slides_prompt(
        doc_context: str,
        num_slides: int = 5,
        user_instructions: str = "",
    ) -> List[Dict[str, str]]:
        """
        Prompt for generating a slide deck outline.

        JSON output schema:
        {
          "slides": [
            {"title": "...", "bullets": ["...", "..."]}
          ]
        }
        """
        context_snippet = doc_context[:CONTEXT_TRUNCATION_LIMIT]
        if len(doc_context) > CONTEXT_TRUNCATION_LIMIT:
            print(f"⚠️ Context truncated from {len(doc_context)} to {CONTEXT_TRUNCATION_LIMIT} characters")

        user_content = f"""
Create an outline for a {num_slides}-slide presentation based on this document.
Each slide should have a short title and 3-5 bullet points.

USER_INSTRUCTIONS (optional, may be empty):
{user_instructions.strip() or "None"}

DOCUMENT_CONTEXT:
{context_snippet}

RESPONSE FORMAT (valid JSON only):
{{
  "slides": [
    {{
      "title": "very short slide title (max 1 line)",
      "bullets": [
        "concise bullet 1 (max 1 line)",
        "concise bullet 2",
        "concise bullet 3"
      ]
    }}
  ]
}}

Rules:
- Aim for {num_slides} slides if there is enough content.
- Keep titles and bullets very short and presentation-ready.
- Do NOT add extra top-level fields.
- Do NOT wrap the JSON in markdown fences.
"""
        return [
            {"role": "system", "content": PromptBuilder._system_prompt()},
            {"role": "user", "content": user_content},
        ]

    # ---- CHAT --------------------------------------------------------

    @staticmethod
    def get_chat_prompt(
        *,
        user_message: str,
        conversation_history: List[Dict[str, str]],
        retrieved_chunks: List[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        """
        Prompt for chat (RAG) using:
        - conversation_history: last N messages (chronological) as [{"role": "user"/"assistant", "content": "..."}]
        - retrieved_chunks: chunk snippets formatted as [{"ref": "...", "content": "..."}]

        Notes:
        - No query rewriting.
        - Not JSON mode.
        """
        # Build history block (already capped by API)
        history_lines: List[str] = []
        for m in conversation_history:
            role = (m.get("role") or "").strip().lower()
            content = (m.get("content") or "").strip()
            if not content:
                continue
            if role not in {"user", "assistant"}:
                role = "user"
            prefix = "User" if role == "user" else "Assistant"
            history_lines.append(f"{prefix}: {content}")

        chunks_lines: List[str] = []
        for c in retrieved_chunks:
            ref = (c.get("ref") or "").strip()
            content = (c.get("content") or "").strip()
            if not content:
                continue
            if ref:
                chunks_lines.append(f"{ref}\n{content}")
            else:
                chunks_lines.append(content)

        history_block = "\n".join(history_lines) if history_lines else "None"
        chunks_block = "\n\n".join(chunks_lines) if chunks_lines else "None"

        user_content = f"""
You are a document Q&A chat assistant.
You must answer using ONLY the provided DOCUMENT CONTEXT.
If the answer is not in the context, say you don't have enough information.

CONVERSATION HISTORY (most recent 30 messages max; may be empty):
{history_block}

DOCUMENT CONTEXT (retrieved chunks):
{chunks_block}

CURRENT USER MESSAGE:
{user_message.strip()}
"""

        return [
            {"role": "system", "content": PromptBuilder._system_prompt()},
            {"role": "user", "content": user_content},
        ]

