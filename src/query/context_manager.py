"""Conversation context manager for multi-turn interactions."""

from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime
from collections import deque


@dataclass
class ContextEntry:
    """A single turn in the conversation."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    query_rewritten: Optional[str] = None  # Rewritten query if applicable
    retrieved_topics: List[str] = field(default_factory=list)


@dataclass
class ConversationContext:
    """Current conversation state."""
    session_id: str
    entries: List[ContextEntry]
    active_topic: Optional[str] = None
    mentioned_entities: List[str] = field(default_factory=list)
    
    def get_recent_queries(self, n: int = 3) -> List[str]:
        """Get the n most recent user queries."""
        user_entries = [e for e in self.entries if e.role == "user"]
        return [e.content for e in user_entries[-n:]]
    
    def get_context_summary(self) -> str:
        """Get a summary of conversation context for LLM."""
        recent = self.entries[-5:] if len(self.entries) > 5 else self.entries
        lines = []
        for entry in recent:
            role = "User" if entry.role == "user" else "Assistant"
            # Truncate long responses
            content = entry.content[:200] + "..." if len(entry.content) > 200 else entry.content
            lines.append(f"{role}: {content}")
        return "\n".join(lines)


class ContextManager:
    """
    Manages conversation context for query rewriting.
    
    Maintains a sliding window of conversation history to resolve
    referential queries like "What about the second one?"
    """
    
    def __init__(self, max_history: int = 10):
        """Initialize context manager."""
        self.max_history = max_history
        self.sessions: dict[str, ConversationContext] = {}
    
    def get_or_create_session(self, session_id: str) -> ConversationContext:
        """Get existing session or create new one."""
        if session_id not in self.sessions:
            self.sessions[session_id] = ConversationContext(
                session_id=session_id,
                entries=[]
            )
        return self.sessions[session_id]
    
    def add_user_message(
        self, 
        session_id: str, 
        content: str,
        rewritten_query: Optional[str] = None
    ) -> None:
        """Add a user message to the session."""
        session = self.get_or_create_session(session_id)
        entry = ContextEntry(
            role="user",
            content=content,
            query_rewritten=rewritten_query
        )
        session.entries.append(entry)
        self._trim_history(session)
        self._extract_entities(session, content)
    
    def add_assistant_message(
        self, 
        session_id: str, 
        content: str,
        topics: Optional[List[str]] = None
    ) -> None:
        """Add an assistant message to the session."""
        session = self.get_or_create_session(session_id)
        entry = ContextEntry(
            role="assistant",
            content=content,
            retrieved_topics=topics or []
        )
        session.entries.append(entry)
        self._trim_history(session)
        
        # Update active topic if topics were discussed
        if topics:
            session.active_topic = topics[0]
    
    def get_context_for_rewriting(self, session_id: str) -> dict:
        """Get context needed for query rewriting."""
        session = self.get_or_create_session(session_id)
        
        return {
            "recent_queries": session.get_recent_queries(3),
            "context_summary": session.get_context_summary(),
            "active_topic": session.active_topic,
            "mentioned_entities": session.mentioned_entities[-10:],
            "turn_count": len([e for e in session.entries if e.role == "user"])
        }
    
    def _trim_history(self, session: ConversationContext) -> None:
        """Trim history to max size."""
        if len(session.entries) > self.max_history:
            session.entries = session.entries[-self.max_history:]
    
    def _extract_entities(self, session: ConversationContext, text: str) -> None:
        """Extract and track mentioned entities."""
        # Simple entity extraction - could be enhanced with NER
        import re
        
        # Look for technical terms, numbers, codes
        patterns = [
            r'\b(RAID\s*\d+)\b',
            r'\b(PSU\s*\d+)\b',
            r'\b(port\s*\d+)\b',
            r'\b(slot\s*\d+)\b',
            r'\b(error\s*\w+)\b',
            r'\b(table\s*\d+-\d+)\b',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            session.mentioned_entities.extend(matches)
        
        # Keep unique and recent
        session.mentioned_entities = list(dict.fromkeys(session.mentioned_entities[-20:]))
    
    def clear_session(self, session_id: str) -> None:
        """Clear a session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
