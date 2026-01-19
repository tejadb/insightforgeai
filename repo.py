"""
InsightForge Repository.

Handles all interactions with Supabase (Database + Storage).
Tables: if_documents, if_insights
Bucket: dbdocs (shared)
"""

import os
import json
from typing import List, Dict, Any, Optional, Tuple
from uuid import UUID
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Embedding dimension (must match schema and embedding model)
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "1536"))

class InsightRepo:
    def __init__(self):
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_KEY")
        
        if not url or not key:
            raise ValueError("Missing SUPABASE_URL or SUPABASE_SERVICE_KEY in environment.")
            
        self.supabase: Client = create_client(url, key)
        self.bucket_name = "dbdocs"  # Can be configurable

    # ---- DOCUMENTS ---------------------------------------------------

    def create_document_record(self, user_id: str, title: str, storage_path: str) -> str:
        """
        Create a new document record (status='pending').
        Returns the new document ID.
        """
        data = {
            "user_id": user_id,
            "title": title,
            "storage_path": storage_path,
            "status": "pending",
            "elements": [], # Empty init
            "context": "", # Empty init
            "created_at": "now()",
            "updated_at": "now()"
        }
        response = self.supabase.table("if_documents").insert(data).execute()
        if response.data:
            return response.data[0]["id"]
        raise ValueError("Failed to create document record")

    def get_document(self, doc_id: str) -> Dict[str, Any]:
        """Fetch document metadata/status by ID."""
        response = self.supabase.table("if_documents").select("*").eq("id", doc_id).single().execute()
        if not response.data:
            raise ValueError(f"Document {doc_id} not found.")
        return response.data

    def update_document_status(self, doc_id: str, status: str, error_message: Optional[str] = None):
        """Update processing status (e.g., 'processing', 'error')."""
        data = {"status": status, "updated_at": "now()"}
        if error_message:
            data["error_message"] = error_message
            
        self.supabase.table("if_documents").update(data).eq("id", doc_id).execute()

    def update_document_content(self, doc_id: str, elements: List[Any], context: str):
        """Save parsed content and mark as completed."""
        # Convert elements to JSON-serializable dicts if they aren't already
        # (The parser returns Element objects, we need lists of dicts)
        from unstructured.staging.base import elements_to_dicts
        
        # Check if elements are already dicts or objects
        elements_json = elements
        if elements and not isinstance(elements[0], dict):
            try:
                elements_json = elements_to_dicts(elements)
            except Exception as e:
                # Fallback: convert to basic dict representation
                print(f"⚠️ Warning: elements_to_dicts failed, using fallback: {str(e)}")
                elements_json = [{"text": str(elem), "type": getattr(elem, "type", "Unknown")} for elem in elements]

        data = {
            "elements": elements_json,
            "context": context,
            "status": "completed",
            "error_message": None,
            "updated_at": "now()"
        }
        self.supabase.table("if_documents").update(data).eq("id", doc_id).execute()

    def get_documents_context(self, doc_ids: List[str]) -> List[str]:
        """
        Retrieve context text for multiple documents.
        Ensures they are all 'completed'.
        """
        if not doc_ids:
            return []
            
        response = self.supabase.table("if_documents")\
            .select("id, title, context, status")\
            .in_("id", doc_ids)\
            .execute()
            
        contexts = []
        for doc in response.data:
            if doc["status"] != "completed":
                print(f"⚠️ Warning: Document {doc['title']} ({doc['id']}) is not processed. Status: {doc['status']}")
                continue
            contexts.append(doc["context"])
            
        return contexts

    def get_document_titles(self, doc_ids: List[str]) -> Dict[str, str]:
        """
        Fetch {document_id: title} for the provided doc_ids.

        Used to generate user-friendly citations (title + page range) without exposing UUIDs.
        """
        if not doc_ids:
            return {}
        resp = (
            self.supabase.table("if_documents")
            .select("id, title")
            .in_("id", doc_ids)
            .execute()
        )
        rows = resp.data or []
        out: Dict[str, str] = {}
        for r in rows:
            did = str(r.get("id"))
            title = (r.get("title") or "").strip()
            if did:
                out[did] = title or did
        return out

    # ---- CHAT (single chat per user) ---------------------------------

    def create_chat_message(
        self,
        *,
        user_id: str,
        role: str,
        content: str,
        document_ids: List[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Insert a chat message into `if_chat_messages`.

        Notes:
        - We store ALL messages; the API decides how many to include in LLM context.
        - `document_ids` is required and may vary per message.
        """
        if role not in {"user", "assistant"}:
            raise ValueError("role must be 'user' or 'assistant'")
        if not content or not content.strip():
            raise ValueError("content cannot be empty")
        if not document_ids:
            raise ValueError("document_ids cannot be empty")

        data = {
            "user_id": user_id,
            "role": role,
            "content": content,
            "document_ids": document_ids,
            "metadata": metadata or {},
        }
        resp = self.supabase.table("if_chat_messages").insert(data).execute()
        if resp.data:
            return resp.data[0]["id"]
        raise ValueError("Failed to create chat message")

    def get_chat_history(
        self,
        *,
        user_id: str,
        limit: int = 30,
    ) -> List[Dict[str, Any]]:
        """
        Get latest chat messages for a user (most recent first).
        """
        if limit <= 0:
            return []
        resp = (
            self.supabase.table("if_chat_messages")
            .select("*")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return resp.data or []

    def validate_documents_for_user(
        self,
        *,
        user_id: str,
        document_ids: List[str],
        require_completed: bool = True,
    ) -> None:
        """
        Validate that:
        - all provided document_ids exist
        - they belong to user_id
        - optionally they are status='completed'
        """
        if not document_ids:
            raise ValueError("document_ids cannot be empty")

        # Validate UUID format early (prevents noisy Postgres errors)
        try:
            UUID(user_id)
        except Exception:
            raise ValueError("user_id must be a valid UUID")
        for d in document_ids:
            try:
                UUID(d)
            except Exception:
                raise ValueError(f"Invalid document_id UUID: {d}")

        resp = (
            self.supabase.table("if_documents")
            .select("id, user_id, status, title")
            .in_("id", document_ids)
            .execute()
        )
        rows = resp.data or []
        found_ids = {r["id"] for r in rows}
        missing = [d for d in document_ids if d not in found_ids]
        if missing:
            raise ValueError(f"Some documents not found: {missing}")

        not_owned = [r["id"] for r in rows if str(r["user_id"]) != str(user_id)]
        if not_owned:
            raise ValueError(f"Some documents do not belong to user: {not_owned}")

        if require_completed:
            not_ready = [r["id"] for r in rows if r.get("status") != "completed"]
            if not_ready:
                raise ValueError(f"Some documents are not processed yet: {not_ready}")

    # ---- INSIGHTS ----------------------------------------------------

    def create_insight(
        self, 
        user_id: str, 
        action_type: str, 
        doc_ids: List[str], 
        instructions: str,
        title: str = ""
    ) -> str:
        """
        Create a new insight record (status='pending').
        Returns the new insight ID.
        """
        data = {
            "user_id": user_id,
            "action_type": action_type,
            "document_ids": doc_ids,
            "instructions": instructions,
            "title": title or f"{action_type.title()} generation",
            "status": "pending",
            "result": {}, # Placeholder until complete
            "created_at": "now()",
            "updated_at": "now()"
        }
        
        response = self.supabase.table("if_insights").insert(data).execute()
        if response.data:
            return response.data[0]["id"]
        raise ValueError("Failed to create insight record")

    def update_insight_result(
        self, 
        insight_id: str, 
        result: Dict[str, Any], 
        status: str = "completed",
        error_message: Optional[str] = None,
        llm_metadata: Optional[Dict[str, Any]] = None
    ):
        """Save the LLM result."""
        data = {
            "result": result,
            "status": status,
            "updated_at": "now()"
        }
        if error_message:
            data["error_message"] = error_message
        if llm_metadata:
            data["llm_metadata"] = llm_metadata
            
        self.supabase.table("if_insights").update(data).eq("id", insight_id).execute()

    def update_insight_status(self, insight_id: str, status: str, error_message: Optional[str] = None):
        """Update insight processing status (e.g., 'processing', 'error')."""
        data = {"status": status, "updated_at": "now()"}
        if error_message:
            data["error_message"] = error_message
        self.supabase.table("if_insights").update(data).eq("id", insight_id).execute()

    def get_insight(self, insight_id: str) -> Dict[str, Any]:
        """Fetch insight result/status."""
        response = self.supabase.table("if_insights").select("*").eq("id", insight_id).single().execute()
        if not response.data:
            raise ValueError(f"Insight {insight_id} not found.")
        return response.data

    # ---- STORAGE -----------------------------------------------------

    def upload_file(self, local_path: str, storage_path: str) -> str:
        """Upload a local file to Supabase storage."""
        try:
            with open(local_path, 'rb') as f:
                print(f"📤 Uploading to '{self.bucket_name}': {storage_path}")
                self.supabase.storage.from_(self.bucket_name).upload(
                    path=storage_path,
                    file=f,
                    file_options={"content-type": "application/octet-stream"}
                )
            return storage_path
        except Exception as e:
            # Check if file exists (error 409ish), if so, return path
            if "Duplicate" in str(e) or "already exists" in str(e):
                print(f"⚠️ File {storage_path} already exists, using existing.")
                return storage_path
            raise ValueError(f"Failed to upload file: {str(e)}")

    def upload_bytes(self, file_bytes: bytes, storage_path: str, content_type: str = "application/octet-stream") -> str:
        """Upload raw bytes to Supabase storage."""
        try:
            print(f"📤 Uploading bytes to '{self.bucket_name}': {storage_path}")
            self.supabase.storage.from_(self.bucket_name).upload(
                path=storage_path,
                file=file_bytes,
                file_options={"content-type": content_type},
            )
            return storage_path
        except Exception as e:
            if "Duplicate" in str(e) or "already exists" in str(e):
                print(f"⚠️ File {storage_path} already exists, using existing.")
                return storage_path
            raise ValueError(f"Failed to upload bytes: {str(e)}")

    def download_file(self, storage_path: str) -> bytes:
        """Download file bytes from Supabase storage bucket."""
        try:
            # storage_path typically looks like "folder/filename.pdf"
            # bucket is defined in self.bucket_name (dbdocs)
            print(f"📥 Downloading from bucket '{self.bucket_name}': {storage_path}")
            data = self.supabase.storage.from_(self.bucket_name).download(storage_path)
            return data
        except Exception as e:
            raise ValueError(f"Failed to download file '{storage_path}': {str(e)}")

    # ---- DOCUMENT CHUNKS (RAG) ----------------------------------------

    def store_document_chunks(
        self,
        doc_id: str,
        chunks: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Store document chunks with embeddings in if_document_chunks table.
        
        Args:
            doc_id: Document ID
            chunks: List of chunk dicts, each with:
                - content: str (chunk text)
                - embedding: List[float] (vector embedding)
                - chunk_index: int
                - metadata: Dict (page_start, page_end, section_heading, element_ids, etc.)
        
        Returns:
            List of chunk IDs (UUIDs)
        """
        if not chunks:
            return []
        
        # Prepare data for bulk insert with validation
        rows = []
        for chunk in chunks:
            # Validate embedding dimensions
            embedding = chunk.get("embedding")
            if not embedding:
                raise ValueError(f"Chunk {chunk.get('chunk_index', 'unknown')} missing embedding")
            
            if len(embedding) != EMBEDDING_DIMENSIONS:
                raise ValueError(
                    f"Chunk {chunk.get('chunk_index', 'unknown')} has wrong embedding dimension: "
                    f"expected {EMBEDDING_DIMENSIONS}, got {len(embedding)}"
                )
            
            row = {
                "document_id": doc_id,
                "content": chunk["content"],
                "embedding": embedding,  # Supabase handles vector type conversion
                "chunk_index": chunk["chunk_index"],
                "metadata": chunk.get("metadata", {}),
            }
            
            # Extract metadata fields
            metadata = chunk.get("metadata", {})
            if "page_start" in metadata:
                row["page_start"] = metadata["page_start"]
            if "page_end" in metadata:
                row["page_end"] = metadata["page_end"]
            if "section_heading" in metadata:
                row["section_heading"] = metadata["section_heading"]
            if "element_ids" in metadata:
                row["element_ids"] = metadata["element_ids"]
            
            rows.append(row)
        
        # Bulk insert (Supabase handles transaction automatically)
        try:
            response = self.supabase.table("if_document_chunks").insert(rows).execute()
            chunk_ids = [row["id"] for row in response.data]
            print(f"✅ Stored {len(chunk_ids)} chunks for document {doc_id}")
            return chunk_ids
        except Exception as e:
            print(f"❌ Failed to store chunks: {str(e)}")
            raise ValueError(f"Failed to store document chunks: {str(e)}")

    def query_similar_chunks(
        self,
        query_embedding: List[float],
        doc_ids: List[str],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Query similar chunks using vector similarity search.
        
        Uses pgvector cosine similarity via RPC function for efficient database-side search.
        
        Requirements:
        - Supabase RPC function 'search_similar_chunks' must exist (see schema_chunks.sql)
        - Function signature: search_similar_chunks(query_embedding vector, doc_ids uuid[], top_k int)
        
        Args:
            query_embedding: Query vector (must match EMBEDDING_DIMENSIONS)
            doc_ids: List of document IDs to search within
            top_k: Number of top results to return
        
        Returns:
            List of chunk dicts with similarity scores, ordered by similarity (highest first)
        """
        if not doc_ids or not query_embedding:
            return []
        
        # Validate embedding dimensions
        if len(query_embedding) != EMBEDDING_DIMENSIONS:
            raise ValueError(
                f"Query embedding has wrong dimension: expected {EMBEDDING_DIMENSIONS}, "
                f"got {len(query_embedding)}"
            )
        
        try:
            # Use RPC function for efficient database-side vector search
            # This avoids fetching all chunks and computing similarity in Python
            response = self.supabase.rpc(
                "search_similar_chunks",
                {
                    "query_embedding": query_embedding,
                    "doc_ids": doc_ids,
                    "top_k": top_k
                }
            ).execute()
            
            if not response.data:
                return []
            
            return response.data
            
        except Exception as e:
            error_str = str(e).lower()
            # If RPC function doesn't exist, fall back to Python computation (less efficient)
            if "function" in error_str and "does not exist" in error_str:
                print(f"⚠️ RPC function not found, falling back to Python computation")
                return self._query_similar_chunks_fallback(query_embedding, doc_ids, top_k)
            
            print(f"❌ Failed to query similar chunks: {str(e)}")
            raise ValueError(f"Failed to query similar chunks: {str(e)}")
    
    def _query_similar_chunks_fallback(
        self,
        query_embedding: List[float],
        doc_ids: List[str],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Fallback method: compute similarity in Python (less efficient).
        Used when RPC function is not available.
        """
        try:
            import numpy as np
        except ImportError:
            raise ValueError("numpy is required for vector similarity search. Install with: pip install numpy")
        
        # Fetch all chunks from selected docs
        response = self.supabase.table("if_document_chunks")\
            .select("*")\
            .in_("document_id", doc_ids)\
            .execute()
        
        if not response.data:
            return []
        
        # Compute cosine similarity in Python
        query_vec = np.array(query_embedding)
        results = []
        
        for chunk in response.data:
            chunk_embedding = chunk.get("embedding")
            if not chunk_embedding:
                continue
            
            chunk_vec = np.array(chunk_embedding)
            
            # Cosine similarity
            dot_product = np.dot(query_vec, chunk_vec)
            norm_query = np.linalg.norm(query_vec)
            norm_chunk = np.linalg.norm(chunk_vec)
            
            if norm_query > 0 and norm_chunk > 0:
                similarity = dot_product / (norm_query * norm_chunk)
            else:
                similarity = 0.0
            
            chunk["similarity"] = float(similarity)
            results.append(chunk)
        
        # Sort by similarity (highest first) and return top_k
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]

    def get_chunks_by_document(self, doc_id: str) -> List[Dict[str, Any]]:
        """
        Get all chunks for a document, ordered by chunk_index.
        
        Args:
            doc_id: Document ID
        
        Returns:
            List of chunk dicts, ordered by chunk_index
        """
        try:
            response = self.supabase.table("if_document_chunks")\
                .select("*")\
                .eq("document_id", doc_id)\
                .order("chunk_index")\
                .execute()
            
            return response.data or []
        except Exception as e:
            print(f"❌ Failed to get chunks: {str(e)}")
            raise ValueError(f"Failed to get chunks for document {doc_id}: {str(e)}")

    def delete_document_chunks(self, doc_id: str):
        """
        Delete all chunks for a document (cleanup).
        
        Args:
            doc_id: Document ID
        """
        try:
            self.supabase.table("if_document_chunks")\
                .delete()\
                .eq("document_id", doc_id)\
                .execute()
            print(f"✅ Deleted chunks for document {doc_id}")
        except Exception as e:
            print(f"❌ Failed to delete chunks: {str(e)}")
            raise ValueError(f"Failed to delete chunks for document {doc_id}: {str(e)}")
    
    def delete_document(self, doc_id: str):
        """
        Delete a document and all its chunks.
        
        Chunks are automatically deleted via CASCADE, but this method provides
        explicit cleanup and error handling.
        
        Args:
            doc_id: Document ID
        """
        try:
            # Delete chunks first (explicit cleanup, though CASCADE handles it)
            self.delete_document_chunks(doc_id)
            
            # Delete document record
            self.supabase.table("if_documents")\
                .delete()\
                .eq("id", doc_id)\
                .execute()
            print(f"✅ Deleted document {doc_id}")
        except Exception as e:
            print(f"❌ Failed to delete document: {str(e)}")
            raise ValueError(f"Failed to delete document {doc_id}: {str(e)}")
