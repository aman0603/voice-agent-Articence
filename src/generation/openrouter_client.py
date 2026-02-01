"""OpenRouter LLM client using Trinity model."""

import asyncio
import time
from typing import AsyncIterator, List, Optional
from dataclasses import dataclass

import httpx

from ..config import get_settings
from ..retrieval import SearchResult


@dataclass
class GenerationChunk:
    """A chunk of generated response."""
    text: str
    is_complete: bool = False
    latency_ms: int = 0


# System prompt optimized for voice output
VOICE_SYSTEM_PROMPT = """You are a technical support voice assistant helping customers troubleshoot Dell PowerEdge hardware issues.

CRITICAL RULES FOR VOICE OUTPUT:
1. Keep sentences SHORT (max 15 words each)
2. Use simple, conversational language
3. Avoid parenthetical references like "(see Table 7-4)"
4. Spell out abbreviations when first used
5. Use natural transitions: "First...", "Next...", "Finally..."
6. Break complex steps into simple bullet points
7. Be direct and helpful"""


class OpenRouterClient:
    """
    LLM client using OpenRouter API with Trinity model.
    
    Free tier with no rate limits for testing.
    """
    
    def __init__(self, model: str = "arcee-ai/trinity-large-preview:free"):
        """Initialize OpenRouter client."""
        self.settings = get_settings()
        self.model_name = model
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self._client = None
    
    @property
    def api_key(self) -> str:
        """Get OpenRouter API key."""
        return self.settings.open_router_key
    
    @property
    def headers(self) -> dict:
        """Get API headers."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "Voice RAG Agent"
        }
    
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
            {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {query}"}
        ]
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.3,
            "stream": True
        }
        
        start_time = time.time()
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream(
                    "POST", 
                    self.api_url, 
                    headers=self.headers, 
                    json=payload
                ) as response:
                    if response.status_code != 200:
                        error_text = await response.aread()
                        yield GenerationChunk(
                            text=f"Error: {error_text.decode()[:100]}",
                            is_complete=True
                        )
                        return
                    
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data = line[6:]
                            if data == "[DONE]":
                                break
                            
                            try:
                                import json
                                chunk_data = json.loads(data)
                                delta = chunk_data.get("choices", [{}])[0].get("delta", {})
                                content = delta.get("content", "")
                                
                                if content:
                                    elapsed_ms = int((time.time() - start_time) * 1000)
                                    yield GenerationChunk(
                                        text=content,
                                        is_complete=False,
                                        latency_ms=elapsed_ms
                                    )
                            except (json.JSONDecodeError, IndexError, KeyError):
                                continue
            
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
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.3
        }
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return data["choices"][0]["message"]["content"]
                else:
                    return f"Error: {response.status_code} - {response.text[:100]}"
                    
        except Exception as e:
            return f"I apologize, but I encountered an issue: {str(e)[:100]}"
    
    def _build_context(
        self,
        docs: List[SearchResult],
        max_chars: int = 3000
    ) -> str:
        """Build context string from retrieved documents."""
        if not docs:
            return "No relevant documentation found."
        
        context_parts = []
        total_chars = 0
        
        for i, doc in enumerate(docs[:5], 1):
            content = doc.content.strip()
            
            if total_chars + len(content) > max_chars:
                remaining = max_chars - total_chars
                if remaining > 100:
                    content = content[:remaining] + "..."
                else:
                    break
            
            context_parts.append(f"[Document {i}]\n{content}")
            total_chars += len(content)
        
        return "\n\n".join(context_parts)


# Alias for compatibility
LLMClient = OpenRouterClient
