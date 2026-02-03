"""Streaming LLM client using Google Gemini."""

import asyncio
from typing import AsyncIterator, List, Optional
from dataclasses import dataclass

from google import genai
from google.genai import types

from ..config import get_settings
from ..retrieval import SearchResult


@dataclass
class GenerationChunk:
    """A chunk of generated response."""
    text: str
    is_complete: bool = False
    latency_ms: int = 0


# System prompt optimized for voice output
VOICE_SYSTEM_PROMPT = """You are a friendly technical support voice assistant helping users troubleshoot hardware and system issues.

PRIMARY GOAL:
Sound like a calm, knowledgeable human speaking naturally.

STRICT KNOWLEDGE RULES:
- Use ONLY the information provided in the context.
- If the answer is not explicitly stated, say:
  "That is not mentioned in the document."
- Never invent commands, steps, or behavior.

DEFAULT SPEECH STYLE (IMPORTANT):
- Speak in short, natural sentences.
- Use paragraph-style speech by default.
- Do NOT use lists or numbered points unless the user explicitly asks for steps.
- Avoid repeating the same idea in different words.
- Explain naturally, as you would to a colleague.

WHEN TO USE BULLETS:
- Use bullet points ONLY for step-by-step procedures.
- Maximum five bullets.
- Each bullet must be a short sentence.
- Never number bullets unless order truly matters.

ANSWER FLOW (DO NOT USE NUMBERS OR LISTS):
Start with a direct answer, followed by a brief explanation. You can add one example or condition if it is genuinely helpful. Always end with a short confirmation question. Keep the response as a single cohesive paragraph.

VOICE DELIVERY RULES:
- No long compound sentences.
- No parenthetical references.
- No section names, tables, or page numbers.
- No code blocks unless explicitly requested.
- Use natural transitions like “For example…” or “In that case…”

WHEN INFORMATION IS MISSING:
- Say it clearly and briefly.
- Do not speculate or infer.

CONTEXT FROM THE TECHNICAL MANUAL:
{context}

USER QUESTION:
{query}

Respond in clear, natural speech suitable for text-to-speech output.
"""


class LLMClient:
    """
    Streaming LLM client using Google Gemini.
    
    Optimized for low-latency voice responses with streaming output.
    """
    
    def __init__(self, model: str = "gemini-2.5-flash"):
        """Initialize LLM client."""
        self.settings = get_settings()
        self.model_name = model
        self._client = None
    
    @property
    def client(self):
        """Lazy load Gemini client."""
        if self._client is None:
            self._client = genai.Client(api_key=self.settings.gemini_api_key)
        return self._client
    
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
        # Build context from top documents
        context = self._build_context(context_docs)
        
        # Format prompt
        prompt = VOICE_SYSTEM_PROMPT.format(
            context=context,
            query=query
        )
        
        # Stream generation
        import time
        start_time = time.time()
        
        loop = asyncio.get_event_loop()
        
        try:
            response = await loop.run_in_executor(
                None,
                lambda: self.client.models.generate_content_stream(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.3,
                        max_output_tokens=max_tokens,
                    )
                )
            )
            
            # Yield chunks as they arrive
            for chunk in response:
                if chunk.text:
                    elapsed_ms = int((time.time() - start_time) * 1000)
                    yield GenerationChunk(
                        text=chunk.text,
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
                text=f"I apologize, but I encountered an issue. {str(e)}",
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
        
        For cases where full response is needed before TTS.
        """
        chunks = []
        async for chunk in self.generate_streaming(query, context_docs, max_tokens):
            chunks.append(chunk.text)
        
        return "".join(chunks)
    
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
        
        for i, doc in enumerate(docs[:5], 1):  # Top 5 docs
            # Extract key info
            content = doc.content.strip()
            
            # Truncate if needed
            if total_chars + len(content) > max_chars:
                remaining = max_chars - total_chars
                if remaining > 100:
                    content = content[:remaining] + "..."
                else:
                    break
            
            context_parts.append(f"[Document {i}]\n{content}")
            total_chars += len(content)
        
        return "\n\n".join(context_parts)
