"""
NVIDIA API client using Llama models.
Optimized for low-latency, voice-friendly RAG.
"""

import time
from typing import AsyncIterator, List
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


# =========================
# System Prompt (VOICE + RAG SAFE)
# =========================

VOICE_SYSTEM_PROMPT = """You are a technical support voice assistant helping customers troubleshoot hardware issues.

PRIMARY GOAL:
Sound like a calm, knowledgeable human — not a manual, not a robot.

STRICT KNOWLEDGE RULES:
- Use ONLY the information provided in the context.
- If the answer is not explicitly stated, say:
  "That is not mentioned in the document."
- Never invent commands, steps, or behavior.

SPEECH STYLE RULES (VERY IMPORTANT):
- One idea per sentence.
- Keep sentences short and natural.
- Avoid repeating the same idea in different words.
- Explain, do not define.
- Prefer conversational phrasing over technical phrasing.
- Do not read like documentation.

STRUCTURE EVERY ANSWER LIKE THIS:
1. Direct answer in one sentence.
2. Short explanation in one or two sentences.
3. One practical condition or example, if helpful.
4. End with a brief confirmation question.

VOICE DELIVERY RULES:
- No long compound sentences.
- No parenthetical references.
- No section names, tables, or page numbers.
- No code blocks unless the user explicitly asks.
- Use natural transitions like “For example…” or “In that case…”
- Maximum six short bullet points if bullets are needed.

WHEN INFORMATION IS MISSING:
- Say it clearly and briefly.
- Do not speculate or infer.
- Offer nearby helpful information only if it is explicitly stated.

CONTEXT FROM THE TECHNICAL MANUAL:
{context}

USER QUESTION:
{query}

Respond in clear, natural speech suitable for text-to-speech output.
"""

class NvidiaClient:
    """
    LLM client using NVIDIA API with Llama models.
    Optimized for voice-based RAG systems.
    """

    def __init__(self, model: str = "meta/llama-3.1-8b-instruct"):
        self.settings = get_settings()
        self.model_name = model
        self.base_url = "https://integrate.api.nvidia.com/v1"
        self._client = None
        self._async_client = None
    @property
    def api_key(self) -> str:
        return self.settings.nvidia_api_key

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key
            )
        return self._client

    @property
    def async_client(self) -> AsyncOpenAI:
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
        max_tokens: int = 300
    ) -> AsyncIterator[GenerationChunk]:

        context = self._build_context(context_docs)

        # If retrieval failed, fail safely
        if context == "No docs found.":
            yield GenerationChunk(
                text="I could not find this information in the manual. Would you like me to check something else?",
                is_complete=True,
                latency_ms=0
            )
            return

        system_prompt = VOICE_SYSTEM_PROMPT.format(
            context=context,
            query=query
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]

        start_time = time.time()
        sentence_buffer = ""

        try:
            response = await self.async_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=min(max_tokens, 300),
                temperature=0.2,
                stream=True
            )

            async for chunk in response:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if not delta:
                    continue

                sentence_buffer += delta

                # Emit only full sentences for clean TTS
                if sentence_buffer.strip().endswith((".", "?", "!")):
                    elapsed_ms = int((time.time() - start_time) * 1000)
                    yield GenerationChunk(
                        text=sentence_buffer.strip(),
                        is_complete=False,
                        latency_ms=elapsed_ms
                    )
                    sentence_buffer = ""

            # Flush remaining text
            if sentence_buffer.strip():
                elapsed_ms = int((time.time() - start_time) * 1000)
                yield GenerationChunk(
                    text=sentence_buffer.strip(),
                    is_complete=False,
                    latency_ms=elapsed_ms
                )

            elapsed_ms = int((time.time() - start_time) * 1000)
            yield GenerationChunk(
                text="",
                is_complete=True,
                latency_ms=elapsed_ms
            )

        except Exception:
            yield GenerationChunk(
                text="Sorry, I had trouble generating the answer. Please try again.",
                is_complete=True,
                latency_ms=0
            )

    async def generate(
        self,
        query: str,
        context_docs: List[SearchResult],
        max_tokens: int = 300
    ) -> str:

        context = self._build_context(context_docs)

        if context == "No docs found.":
            return "I could not find this information in the manual. Would you like me to check something else?"

        system_prompt = VOICE_SYSTEM_PROMPT.format(
            context=context,
            query=query
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]

        try:
            response = await self.async_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=min(max_tokens, 300),
                temperature=0.2
            )
            return response.choices[0].message.content.strip()

        except Exception:
            return "Sorry, I had trouble generating the answer. Please try again."

    def _build_context(
        self,
        docs: List[SearchResult],
        max_chars: int = 2000
    ) -> str:

        if not docs:
            return "No docs found."

        context_parts = []
        total_chars = 0

        # Use top 3 docs only (latency optimized)
        for doc in docs[:5]:
            content = doc.content.strip()

            if total_chars + len(content) > max_chars:
                remaining = max_chars - total_chars
                if remaining > 100:
                    content = content[:remaining] + "..."
                else:
                    break

            context_parts.append(content)
            total_chars += len(content)

        return "\n\n".join(context_parts)

LLMClient = NvidiaClient
