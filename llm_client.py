"""
LLM Client for InsightForge AI.

Wraps OpenAI API interactions with support for:
- Standard chat completions
- JSON mode (for structured outputs)
- Error handling
"""

import os
import asyncio
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class LLMClient:
    """
    Async client for OpenAI-compatible APIs.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables.")
        
        self.client = AsyncOpenAI(api_key=self.api_key)
        self.default_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    async def get_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.0,
        json_mode: bool = False,
        max_tokens: int = 4000,
        max_retries: int = 3
    ) -> str:
        """
        Get chat completion from LLM with retry logic for transient errors.
        
        Args:
            messages: List of message dicts [{"role": "user", "content": "..."}]
            model: Override default model
            temperature: Creativity (0.0 = deterministic)
            json_mode: If True, enforces valid JSON output
            max_tokens: Response length limit
            max_retries: Maximum retry attempts for transient errors
            
        Returns:
            String content of the response.
        """
        retries = 0
        while retries < max_retries:
            try:
                params: Dict[str, Any] = {
                    "model": model or self.default_model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }

                if json_mode:
                    params["response_format"] = {"type": "json_object"}

                response = await self.client.chat.completions.create(**params)
                
                # Check if response has choices
                if not response.choices or len(response.choices) == 0:
                    raise ValueError("LLM returned empty choices list")
                
                content = response.choices[0].message.content
                if not content:
                    raise ValueError("LLM returned empty response")
                    
                return content

            except Exception as e:
                error_str = str(e).lower()
                error_type = type(e).__name__
                
                # Check for rate limit or transient errors
                if "rate limit" in error_str or "429" in error_str or "timeout" in error_str or "503" in error_str:
                    retries += 1
                    if retries < max_retries:
                        wait_time = 2 ** retries
                        print(f"⚠️ LLM error ({error_type}), retrying in {wait_time}s... (attempt {retries}/{max_retries})")
                        await asyncio.sleep(wait_time)
                        continue
                
                # Other errors: log and raise
                print(f"❌ LLM Error ({error_type}): {str(e)}")
                raise
        
        raise ValueError(f"Failed to get LLM completion after {max_retries} retries")

