"""
Embedding generation for RAG.

Uses OpenAI's embedding API to generate vector embeddings for chunks and queries.
"""

import os
import asyncio
from typing import List, Optional
from openai import AsyncOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Embedding dimension (configurable via env var, default 1536 for text-embedding-3-small)
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "1536"))


class EmbeddingClient:
    """
    Async client for OpenAI embedding API.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables.")
        
        # Increased timeout to 5 minutes to handle large batch embedding requests
        self.client = AsyncOpenAI(api_key=self.api_key, timeout=300.0)  # 5 min
        self.default_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        self.expected_dimensions = EMBEDDING_DIMENSIONS
    
    async def get_embedding(self, text: str, model: Optional[str] = None, max_retries: int = 2) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            model: Override default embedding model
            max_retries: Maximum retry attempts for errors
        
        Returns:
            List of floats (embedding vector)
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty for embedding")
        
        # Reuse batch logic for single text
        embeddings = await self.get_embeddings_batch([text], model, max_retries)
        return embeddings[0]

    async def get_embeddings_batch(self, texts: List[str], model: Optional[str] = None, max_retries: int = 2) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts: List of strings to embed
            model: Override default embedding model
            max_retries: Maximum retry attempts for errors
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
            
        # Validate inputs
        clean_texts = [t.strip() for t in texts]
        if any(not t for t in clean_texts):
             # For batch, we'll allow it but OpenAI might reject empty strings. 
             # Ideally pipeline filters them out.
             # If strictly required, raise error or filter. 
             # Current pipeline filters empty chunks.
             pass

        retries = 0
        while retries < max_retries:
            try:
                response = await self.client.embeddings.create(
                    model=model or self.default_model,
                    input=clean_texts,
                )
                
                embeddings = [d.embedding for d in response.data]
                
                # Validate dimensions of the first one (assume all match)
                if embeddings and len(embeddings[0]) != self.expected_dimensions:
                    raise ValueError(
                        f"Expected {self.expected_dimensions} dimensions, got {len(embeddings[0])}. "
                        f"Check EMBEDDING_DIMENSIONS env var matches your model."
                    )
                
                return embeddings
                
            except Exception as e:
                error_str = str(e).lower()
                error_type = type(e).__name__
                
                # Check for rate limit, transient errors, or explicit asyncio timeouts
                # Explicitly catch ConnectTimeout which often happens with high concurrency
                is_timeout = (
                    "timeout" in error_str 
                    or "timeout" in error_type.lower()
                    or isinstance(e, asyncio.TimeoutError)
                )
                
                if "rate limit" in error_str or "429" in error_str or is_timeout:
                    retries += 1
                    if retries < max_retries:
                        wait_time = 2 ** retries
                        print(f"⚠️ Embedding batch error ({error_type}), retrying in {wait_time}s... (attempt {retries}/{max_retries})")
                        await asyncio.sleep(wait_time)
                        continue
                
                # Other errors: log and raise
                print(f"❌ Embedding Batch Error: {str(e)}")
                raise
        
        raise ValueError(f"Failed to generate embeddings after {max_retries} retries")


# Singleton instance (can be reused)
_embedding_client: Optional[EmbeddingClient] = None


def get_embedding_client() -> EmbeddingClient:
    """Get or create singleton EmbeddingClient instance."""
    global _embedding_client
    if _embedding_client is None:
        _embedding_client = EmbeddingClient()
    return _embedding_client


# Convenience function
async def get_embedding(text: str) -> List[float]:
    """Convenience function to get a single embedding."""
    client = get_embedding_client()
    return await client.get_embedding(text)


async def get_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """Convenience function to get embeddings for a batch of texts."""
    client = get_embedding_client()
    return await client.get_embeddings_batch(texts)

