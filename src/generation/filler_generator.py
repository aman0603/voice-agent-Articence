"""Filler response generator for reducing perceived latency."""

from typing import Optional
import random


# Contextual filler phrases
FILLERS = {
    "troubleshooting": [
        "Let me look that up for you.",
        "Checking the troubleshooting guide now.",
        "I'll find the solution for that.",
        "Let me check what the manual says.",
    ],
    "configuration": [
        "Looking up those configuration steps.",
        "Let me find the setup instructions.",
        "Checking the configuration guide.",
        "I'll get those settings for you.",
    ],
    "explanation": [
        "Let me explain that for you.",
        "Good question. One moment.",
        "I'll break that down for you.",
        "Let me give you the details.",
    ],
    "general": [
        "One moment please.",
        "Let me check on that.",
        "I'm looking that up now.",
        "Just a moment.",
    ],
    "power": [
        "Checking the power supply documentation.",
        "Let me look up the power configuration.",
    ],
    "storage": [
        "Looking at the storage section.",
        "Checking the RAID configuration guide.",
    ],
    "network": [
        "Checking the network documentation.",
        "Let me find the networking section.",
    ],
    "error": [
        "Let me look up that error.",
        "Checking the error codes now.",
        "I'll find what that error means.",
    ],
}

# Transition phrases to connect filler with actual response
TRANSITIONS = [
    "Okay, so",
    "Alright.",
    "Here's what I found.",
    "So,",
    "Okay.",
]


class FillerGenerator:
    """
    Generates contextual filler responses to eliminate perceived silence.
    
    Filler is played while retrieval and LLM generation are in progress,
    then seamlessly transitions to the actual response.
    """
    
    def __init__(self):
        """Initialize filler generator."""
        self.fillers = FILLERS
        self.transitions = TRANSITIONS
        self._last_filler = None
    
    def get_filler(
        self,
        query_type: str = "general",
        domain: Optional[str] = None
    ) -> str:
        """
        Get a contextual filler phrase.
        
        Args:
            query_type: Type of query (troubleshooting, configuration, etc.)
            domain: Optional domain (power, storage, network)
            
        Returns:
            Filler phrase for TTS
        """
        # Try domain-specific filler first
        if domain and domain in self.fillers:
            candidates = self.fillers[domain]
        elif query_type in self.fillers:
            candidates = self.fillers[query_type]
        else:
            candidates = self.fillers["general"]
        
        # Avoid repeating the same filler
        available = [f for f in candidates if f != self._last_filler]
        if not available:
            available = candidates
        
        filler = random.choice(available)
        self._last_filler = filler
        
        return filler
    
    def get_transition(self) -> str:
        """Get a transition phrase to connect filler with response."""
        return random.choice(self.transitions)
    
    def format_with_transition(
        self,
        filler: str,
        response_start: str
    ) -> str:
        """
        Format filler with smooth transition to response.
        
        Ensures natural flow from filler to actual content.
        """
        transition = self.get_transition()
        
        # Check if response already has a natural start
        lower_start = response_start.lower().strip()
        natural_starts = ["first", "to", "the", "you", "let's", "okay", "so"]
        
        if any(lower_start.startswith(s) for s in natural_starts):
            return f"{filler} {transition} {response_start}"
        else:
            return f"{filler} {response_start}"
    
    def should_use_filler(self, estimated_latency_ms: int) -> bool:
        """
        Determine if filler should be used based on estimated latency.
        
        Only use fillers if delay would be noticeable (>300ms).
        """
        return estimated_latency_ms > 300
