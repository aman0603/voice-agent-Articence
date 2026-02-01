"""Query rewriter for resolving referential and conversational queries."""

import asyncio
from typing import Optional
from dataclasses import dataclass
import re

from google import genai
from google.genai import types

from ..config import get_settings
from .context_manager import ContextManager


@dataclass
class RewrittenQuery:
    """Result of query rewriting."""
    original: str
    rewritten: str
    confidence: float
    needs_context: bool
    reasoning: Optional[str] = None


# Patterns that indicate a referential query
REFERENTIAL_PATTERNS = [
    r'\b(it|this|that|these|those)\b',
    r'\b(the first|the second|the third|the last|the other)\b',
    r'\b(one|ones)\b(?!\s+(?:hundred|thousand|million))',
    r'\b(same|another|next|previous)\b',
    r'\b(here|there)\b',
    r'^\s*(and|but|also|what about|how about)\b',
]


REWRITE_PROMPT = """You are a query rewriter for a technical support system. Your task is to rewrite user queries to be self-contained for retrieval from a technical manual database.

CONVERSATION CONTEXT:
{context}

CURRENT USER QUERY: "{query}"

INSTRUCTIONS:
1. If the query contains references (it, this, that, the second one, etc.), resolve them using conversation context
2. If the query is already self-contained, return it unchanged
3. Always produce a clear, retrieval-optimized query
4. Keep technical terms intact
5. Do NOT add information not implied by the context

OUTPUT FORMAT:
Return ONLY the rewritten query, nothing else. No quotes, no explanation.

REWRITTEN QUERY:"""


class QueryRewriter:
    """
    Rewrites conversational queries into standalone retrieval queries.
    
    Handles:
    - Pronoun resolution (it, this, that)
    - Ordinal references (the second one, the first option)
    - Elliptical queries (And what about X?)
    """
    
    def __init__(self, context_manager: Optional[ContextManager] = None):
        """Initialize query rewriter."""
        self.settings = get_settings()
        self.context_manager = context_manager or ContextManager()
        self._client = None
    
    @property
    def client(self):
        """Lazy load Gemini client."""
        if self._client is None:
            self._client = genai.Client(api_key=self.settings.gemini_api_key)
        return self._client
    
    def needs_rewriting(self, query: str) -> bool:
        """Check if query needs rewriting based on referential patterns."""
        query_lower = query.lower()
        return any(
            re.search(pattern, query_lower) 
            for pattern in REFERENTIAL_PATTERNS
        )
    
    async def rewrite(
        self, 
        query: str, 
        session_id: str,
        use_llm: bool = True
    ) -> RewrittenQuery:
        """
        Rewrite a query to be self-contained.
        
        Args:
            query: The user's query
            session_id: Session ID for context lookup
            use_llm: Whether to use LLM for complex rewrites
            
        Returns:
            RewrittenQuery with original and rewritten forms
        """
        # Check if rewriting is needed
        needs_rewrite = self.needs_rewriting(query)
        
        if not needs_rewrite:
            return RewrittenQuery(
                original=query,
                rewritten=query,
                confidence=1.0,
                needs_context=False
            )
        
        # Get context
        context_data = self.context_manager.get_context_for_rewriting(session_id)
        
        # Try rule-based rewriting first (faster)
        rule_result = self._rule_based_rewrite(query, context_data)
        if rule_result and rule_result.confidence > 0.8:
            return rule_result
        
        # Fall back to LLM rewriting
        if use_llm and self.settings.gemini_api_key:
            return await self._llm_rewrite(query, context_data)
        
        # If no LLM, return original
        return RewrittenQuery(
            original=query,
            rewritten=query,
            confidence=0.5,
            needs_context=True,
            reasoning="LLM rewriting unavailable"
        )
    
    def _rule_based_rewrite(
        self, 
        query: str, 
        context: dict
    ) -> Optional[RewrittenQuery]:
        """Apply rule-based query rewriting."""
        query_lower = query.lower()
        recent_queries = context.get("recent_queries", [])
        active_topic = context.get("active_topic")
        
        # Handle "what about X" pattern
        what_about = re.match(r'^(?:and\s+)?(?:what|how)\s+about\s+(.+)', query_lower)
        if what_about and recent_queries:
            last_query = recent_queries[-1]
            # Extract the subject from last query
            subject_match = re.search(r'(?:explain|describe|what is|how does?)\s+(.+?)(?:\?|$)', last_query.lower())
            if subject_match:
                subject = subject_match.group(1)
                new_query = f"{subject} {what_about.group(1)}"
                return RewrittenQuery(
                    original=query,
                    rewritten=new_query,
                    confidence=0.85,
                    needs_context=True,
                    reasoning="Combined with previous query subject"
                )
        
        # Handle "the second/first/third one"
        ordinal_match = re.search(r'the\s+(first|second|third|last|other)\s+(?:one|option|mode|type)', query_lower)
        if ordinal_match and active_topic:
            ordinal = ordinal_match.group(1)
            new_query = f"{ordinal} {active_topic}"
            return RewrittenQuery(
                original=query,
                rewritten=new_query,
                confidence=0.8,
                needs_context=True,
                reasoning=f"Resolved ordinal reference with topic: {active_topic}"
            )
        
        return None
    
    async def _llm_rewrite(
        self, 
        query: str, 
        context: dict
    ) -> RewrittenQuery:
        """Use LLM to rewrite the query."""
        context_summary = context.get("context_summary", "No previous context.")
        
        prompt = REWRITE_PROMPT.format(
            context=context_summary,
            query=query
        )
        
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.1,
                        max_output_tokens=100,
                    )
                )
            )
            
            rewritten = response.text.strip().strip('"\'')
            
            return RewrittenQuery(
                original=query,
                rewritten=rewritten,
                confidence=0.9,
                needs_context=True,
                reasoning="LLM rewriting"
            )
            
        except Exception as e:
            # Fallback to original on error
            return RewrittenQuery(
                original=query,
                rewritten=query,
                confidence=0.5,
                needs_context=True,
                reasoning=f"LLM error: {str(e)}"
            )
