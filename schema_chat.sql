-- ===============================================================
-- InsightForge AI (New) - Chat History Schema (Single chat per user)
-- ===============================================================
--
-- Goals:
-- - Store ALL chat messages for a user (frontend can paginate later)
-- - Include per-message document_ids (each user message can use different docs)
-- - Keep schema minimal and traceable (metadata jsonb)
--
-- Notes:
-- - This schema does NOT create Supabase Storage buckets.
-- - RLS policies are included as OPTIONAL; if you only use the Service Key
--   from your backend, RLS is bypassed anyway.
--

CREATE TABLE IF NOT EXISTS public.if_chat_messages (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),

  -- Owner of the chat (Supabase auth user id)
  user_id uuid NOT NULL,

  -- 'user' or 'assistant'
  role text NOT NULL CHECK (role IN ('user', 'assistant')),

  -- Raw message text
  content text NOT NULL,

  -- Document IDs used for this message (can vary per message)
  document_ids uuid[] NOT NULL,

  -- Flexible metadata for traceability:
  -- - assistant: retrieved_chunks[{chunk_id, document_id, similarity, chunk_index, page_start, page_end, section_heading}], model_used, etc.
  -- - user: client info, message_length, etc.
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb,

  created_at timestamptz NOT NULL DEFAULT now()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_if_chat_messages_user_id
  ON public.if_chat_messages(user_id);

CREATE INDEX IF NOT EXISTS idx_if_chat_messages_user_created_at
  ON public.if_chat_messages(user_id, created_at DESC);

-- Helpful if you ever filter by doc_ids
CREATE INDEX IF NOT EXISTS idx_if_chat_messages_doc_ids_gin
  ON public.if_chat_messages USING GIN(document_ids);

-- ===============================================================
-- OPTIONAL: Row Level Security (RLS)
-- ===============================================================
-- If you only use Supabase Service Key from backend, RLS doesn't matter.
-- If you ever expose chat reads/writes via Supabase anon/auth keys, enable RLS.
--
-- ALTER TABLE public.if_chat_messages ENABLE ROW LEVEL SECURITY;
--
-- CREATE POLICY "if_chat_messages_select_own"
--   ON public.if_chat_messages
--   FOR SELECT
--   USING (auth.uid() = user_id);
--
-- CREATE POLICY "if_chat_messages_insert_own"
--   ON public.if_chat_messages
--   FOR INSERT
--   WITH CHECK (auth.uid() = user_id);


