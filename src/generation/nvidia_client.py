"""NVIDIA API client using Llama models."""

import asyncio
import time
from typing import AsyncIterator, List, Optional
from dataclasses import dataclass

from openai import OpenAI, AsyncOpenAI

from ..config import get_settings
from ..retrieval import SearchResult


@dataclass
class GenerationChunk:
    """A chunk of generated response."""
    text: str
    is_complete: bool = False
    latency_ms: int = 0


# Optimized system prompt for low-latency voice output
VOICE_SYSTEM_PROMPT = """You're a Dell technical support assistant. Answer concisely using documentation context. Keep sentences under 15 words. Be direct and helpful."""


class NvidiaClient:
    """
    LLM client using NVIDIA API with Llama models.
    
    Supports various Llama models through NVIDIA's inference endpoint.
    """
    
    def __init__(self, model: str = "meta/llama-3.1-8b-instruct"):
        """Initialize NVIDIA client."""
        self.settings = get_settings()
        self.model_name = model
        self.base_url = "https://integrate.api.nvidia.com/v1"
        self._client = None
        self._async_client = None
    
    @property
    def api_key(self) -> str:
        """Get NVIDIA API key."""
        return self.settings.nvidia_api_key
    
    @property
    def client(self) -> OpenAI:
        """Lazy load synchronous OpenAI client."""
        if self._client is None:
            self._client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key
            )
        return self._client
    
    @property
    def async_client(self) -> AsyncOpenAI:
        """Lazy load asynchronous OpenAI client."""
        if self._async_client is None:
            self._async_client = AsyncOpenAI(
                base_url=self.base_url,
                api_key=self.api_key
            )
        return self._async_client
    
    async def generate_streaming(
        self,
        query: str,
        context_docs: List[SearchResult],
        max_tokens: int = 500
    ) -> AsyncIterator[GenerationChunk]:
        """
        Generate streaming response from query and context.
        
        Args:
            query: User query
            context_docs: Retrieved documents for context
            max_tokens: Maximum tokens to generate
            
        Yields:
            GenerationChunk with text fragments
        """
        context = self._build_context(context_docs)
        
        messages = [
            {"role": "system", "content": VOICE_SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQ: {query}\nA:"}
        ]
        
        start_time = time.time()
        
        try:
            response = await self.async_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.2,  # Lower temperature for consistency
                stream=True
            )
            
            async for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    elapsed_ms = int((time.time() - start_time) * 1000)
                    yield GenerationChunk(
                        text=chunk.choices[0].delta.content,
                        is_complete=False,
                        latency_ms=elapsed_ms
                    )
            
            # Final chunk
            elapsed_ms = int((time.time() - start_time) * 1000)
            yield GenerationChunk(
                text="",
                is_complete=True,
                latency_ms=elapsed_ms
            )
            
        except Exception as e:
            yield GenerationChunk(
                text=f"I apologize, but I encountered an issue: {str(e)[:100]}",
                is_complete=True
            )
    
    async def generate(
        self,
        query: str,
        context_docs: List[SearchResult],
        max_tokens: int = 500
    ) -> str:
        """
        Generate complete response (non-streaming).
        """
        context = self._build_context(context_docs)
        
        messages = [
            {"role": "system", "content": VOICE_SYSTEM_PROMPT},
            {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {query}"}
        ]
        
        try:
            response = await self.async_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.2
            )
            
            return response.choices[0].message.content
                    
        except Exception as e:
            return f"I apologize, but I encountered an issue: {str(e)[:100]}"
    
    def _build_context(
        self,
        docs: List[SearchResult],
        max_chars: int = 2000  # Reduced from 3000 for faster processing
    ) -> str:
        """Build concise context from retrieved documents."""
        if not docs:
            return "No docs found."
        
        context_parts = []
        total_chars = 0
        
        # Only use top 3 docs (reduced from 5) for faster TTFB
        for doc in docs[:3]:
            content = doc.content.strip()
            
            if total_chars + len(content) > max_chars:
                remaining = max_chars - total_chars
                if remaining > 100:
                    content = content[:remaining] + "..."
                else:
                    break
            
            # Remove document numbers - simpler format
            context_parts.append(content)
            total_chars += len(content)
        
        return "\n\n".join(context_parts)


# Alias for compatibility
LLMClient = NvidiaClient
