-- ===============================================================
-- Document Chunks Table (for RAG/Vector Search)
-- Stores chunked text with vector embeddings for semantic search.
-- ===============================================================

-- Enable pgvector extension (required for vector column)
CREATE EXTENSION IF NOT EXISTS vector;

-- Chunks table
CREATE TABLE IF NOT EXISTS public.if_document_chunks (
  -- Primary key
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  
  -- Foreign key to parent document
  document_id uuid NOT NULL REFERENCES public.if_documents(id) ON DELETE CASCADE,
  
  -- The actual chunk text (what we embed and search)
  content text NOT NULL,
  
    -- Vector embedding (dimensions configurable via EMBEDDING_DIMENSIONS env var, default 1536)
  -- Note: If changing dimensions, update EMBEDDING_DIMENSIONS env var and recreate table
  embedding vector(1536) NOT NULL,
  
  -- Metadata for traceability and ordering
  chunk_index integer NOT NULL,  -- Order within document (0, 1, 2, ...)
  page_start integer,             -- First page this chunk appears on
  page_end integer,               -- Last page this chunk appears on
  section_heading text,           -- e.g. "CHAPTER-1", "Introduction"
  element_ids uuid[],             -- Array of element IDs that contributed to this chunk
  
  -- Chunk metadata (JSONB for flexibility)
  -- Can store: element_types, table_html, etc.
  metadata jsonb,
  
  -- Timestamps
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

-- Indexes for performance

-- For filtering by document (btree index)
CREATE INDEX IF NOT EXISTS idx_if_chunks_document_id 
  ON public.if_document_chunks(document_id);

-- For ordering chunks within a document (btree index)
CREATE INDEX IF NOT EXISTS idx_if_chunks_doc_index 
  ON public.if_document_chunks(document_id, chunk_index);

-- Vector similarity search index (ivfflat for fast approximate search)
-- Note: This index requires at least some data. If you get an error, insert a few rows first.
-- The lists parameter should be approximately rows/1000 for best performance
CREATE INDEX IF NOT EXISTS idx_if_chunks_embedding 
  ON public.if_document_chunks 
  USING ivfflat (embedding vector_cosine_ops)
  WITH (lists = 100);

-- ===============================================================
-- RPC Function for Efficient Vector Similarity Search
-- ===============================================================

-- Function to search similar chunks using pgvector cosine similarity
-- This is much more efficient than fetching all chunks and computing similarity in Python
CREATE OR REPLACE FUNCTION public.search_similar_chunks(
  query_embedding vector(1536),
  doc_ids uuid[],
  top_k int DEFAULT 5
)
RETURNS TABLE (
  id uuid,
  document_id uuid,
  content text,
  chunk_index integer,
  page_start integer,
  page_end integer,
  section_heading text,
  element_ids uuid[],
  metadata jsonb,
  similarity float
)
LANGUAGE plpgsql
AS $$
BEGIN
  RETURN QUERY
  SELECT 
    c.id,
    c.document_id,
    c.content,
    c.chunk_index,
    c.page_start,
    c.page_end,
    c.section_heading,
    c.element_ids,
    c.metadata,
    -- Cosine similarity: 1 - cosine_distance (higher = more similar)
    1 - (c.embedding <=> query_embedding) AS similarity
  FROM public.if_document_chunks c
  WHERE c.document_id = ANY(doc_ids)
  ORDER BY c.embedding <=> query_embedding  -- Order by cosine distance (ascending = most similar first)
  LIMIT top_k;
END;
$$;

