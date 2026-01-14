-- InsightForge AI (New) - Database Schema
-- NOTE: Apply these statements in Supabase (SQL editor or migration).
-- This file is the single source of truth for schema related to the new InsightForge pipeline.

-- ===============================================================
-- 1) Parsed Documents Table
--    Stores parsed elements + combined context per document.
--    We are NOT reusing db9Documents; this is a fresh table.
-- ===============================================================

CREATE TABLE IF NOT EXISTS public.if_documents (
  -- Primary key for this table (separate from any old db9 IDs)
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),

  -- Owner of the document (Supabase auth user id)
  user_id uuid NOT NULL,

  -- Human-friendly title (can come from upload metadata or parsed content)
  title text NOT NULL,

  -- Path/key in Supabase storage (e.g. "folder/file.pdf").
  -- We can reuse an existing bucket like `dbdocs` or create a new one
  -- dedicated to InsightForge; this column just stores the relative path.
  storage_path text NOT NULL,

  -- Raw parsed elements from Unstructured.io.
  -- This is the direct output of `partition(...)` serialized as JSON.
  elements jsonb NOT NULL,

  -- Combined text context we actually feed into the LLM (output of doc_context()).
  -- Keeping this here avoids recomputing on every run.
  context text NOT NULL,

  -- Processing status: 'pending', 'processing', 'completed', 'error'
  -- Allows frontend to show loading state and prevents using unparsed docs.
  status text NOT NULL DEFAULT 'pending',
  error_message text,

  -- Minimal timestamps (very useful for debugging and future cleanup).
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

-- Optional: basic index to quickly list documents per user
CREATE INDEX IF NOT EXISTS idx_if_documents_user_id
  ON public.if_documents(user_id);


-- ===============================================================
-- 2) Generated Insights Table
--    One row per AI run (summary / qna / interview_tips / slides).
-- ===============================================================

CREATE TABLE IF NOT EXISTS public.if_insights (
  -- Primary key for the insight record
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),

  -- Owner of the insight (Supabase auth user id)
  user_id uuid NOT NULL,

  -- Type of insight: 'summary', 'qna', 'interview_tips', 'slides', etc.
  action_type text NOT NULL,

  -- Documents this insight is based on (references if_documents.id).
  -- We store an array because a single insight may use multiple docs.
  document_ids uuid[] NOT NULL,

  -- Optional human-friendly title shown in UI, e.g.
  -- "Summary: Corporate Finance Assignment + Market Sizing Deck"
  title text,

  -- Raw user instructions text that guided this run.
  instructions text,

  -- Full JSON result from the LLM (summary/qna/tips/slides structures).
  result jsonb NOT NULL,

  -- Status of the run (for async processing / retries in future).
  -- For now, everything is 'completed', but this gives us room to grow.
  status text NOT NULL DEFAULT 'completed',

  -- Error message for failed runs (if any).
  error_message text,

  -- Optional LLM metadata: model, provider, token usage, cost, etc.
  llm_metadata jsonb,

  -- Timestamps
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

-- Helpful indexes for querying user history and by type
CREATE INDEX IF NOT EXISTS idx_if_insights_user_created_at
  ON public.if_insights(user_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_if_insights_user_type
  ON public.if_insights(user_id, action_type, created_at DESC);


-- ===============================================================
-- 3) Notes on Storage Buckets
-- ===============================================================
-- Existing buckets used in the old InsightForge agent:
--   - db7InsightForgeDocs
--   - dbdocs
--   - db1UserResume
--   - db4Company-Ext-Docs
--   - db39generatedresume
--
-- For the new pipeline you can:
--   A) Reuse `dbdocs` for all new InsightForge uploads (simpler), OR
--   B) Create a dedicated bucket like `dbInsightForgeDocsV2` to keep
--      these files isolated from other product features.
--
-- This SQL file does NOT create buckets; configure them in Supabase
-- Storage settings. The `storage_path` column above should store
-- only the path/key inside whichever bucket you choose.



