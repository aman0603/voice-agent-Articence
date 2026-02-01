"""Intent detection from partial ASR transcripts for speculative retrieval."""

from dataclasses import dataclass
from typing import List, Optional, Set
import re


@dataclass
class Intent:
    """Detected intent from partial transcript."""
    domain: str  # e.g., "power", "storage", "network"
    keywords: List[str]
    confidence: float
    is_question: bool = False
    query_type: str = "general"  # troubleshooting, configuration, explanation


# Domain-specific keywords for hardware troubleshooting
DOMAIN_KEYWORDS = {
    "power": {
        "power", "psu", "supply", "redundant", "redundancy", "voltage", 
        "watt", "ac", "dc", "ups", "battery", "outlet", "cord"
    },
    "storage": {
        "disk", "drive", "raid", "ssd", "hdd", "storage", "volume",
        "controller", "array", "backup", "partition", "format"
    },
    "network": {
        "network", "ethernet", "port", "switch", "router", "ip", 
        "dhcp", "dns", "vlan", "nic", "cable", "connection"
    },
    "memory": {
        "memory", "ram", "dimm", "ecc", "slot", "module", "gb"
    },
    "processor": {
        "cpu", "processor", "core", "thermal", "heatsink", "fan"
    },
    "boot": {
        "boot", "bios", "uefi", "startup", "post", "firmware"
    },
    "led": {
        "led", "light", "indicator", "blink", "amber", "green", "status"
    },
    "error": {
        "error", "fault", "fail", "warning", "critical", "issue", "problem"
    }
}

QUESTION_PATTERNS = [
    r"\b(what|how|where|when|why|which|can|could|should|is|are|does|do)\b",
    r"\?$",
    r"\b(explain|tell|describe|show)\b"
]

QUERY_TYPE_PATTERNS = {
    "troubleshooting": [
        r"\b(error|fail|not working|issue|problem|fix|resolve|troubleshoot)\b",
        r"\b(doesn't|won't|can't|cannot)\b"
    ],
    "configuration": [
        r"\b(configure|setup|set up|install|enable|disable|change|modify)\b",
        r"\b(settings?|options?|parameters?)\b"
    ],
    "explanation": [
        r"\b(what is|explain|describe|tell me about|how does)\b",
        r"\b(mean|purpose|function|work)\b"
    ]
}


class IntentDetector:
    """
    Detects intent from partial ASR transcripts for speculative retrieval.
    
    Designed for low-latency operation on incomplete utterances.
    """
    
    def __init__(self, confidence_threshold: float = 0.6):
        """Initialize intent detector."""
        self.confidence_threshold = confidence_threshold
        self.domain_keywords = DOMAIN_KEYWORDS
        
    def detect(self, text: str) -> Optional[Intent]:
        """
        Detect intent from text (partial or final transcript).
        
        Args:
            text: Transcript text to analyze
            
        Returns:
            Intent if confidence exceeds threshold, else None
        """
        if not text or len(text.split()) < 2:
            return None
            
        text_lower = text.lower()
        words = set(text_lower.split())
        
        # Detect domain
        domain_scores = {}
        matched_keywords = []
        
        for domain, keywords in self.domain_keywords.items():
            matches = words & keywords
            if matches:
                domain_scores[domain] = len(matches)
                matched_keywords.extend(matches)
        
        if not domain_scores:
            return None
            
        # Get top domain
        top_domain = max(domain_scores.items(), key=lambda x: x[1])
        domain = top_domain[0]
        match_count = top_domain[1]
        
        # Calculate confidence based on keyword density
        confidence = min(0.95, 0.3 + (match_count * 0.2))
        
        # Check if it's a question
        is_question = any(
            re.search(pattern, text_lower) 
            for pattern in QUESTION_PATTERNS
        )
        
        # Detect query type
        query_type = "general"
        for qtype, patterns in QUERY_TYPE_PATTERNS.items():
            if any(re.search(p, text_lower) for p in patterns):
                query_type = qtype
                break
        
        if confidence >= self.confidence_threshold:
            return Intent(
                domain=domain,
                keywords=matched_keywords,
                confidence=confidence,
                is_question=is_question,
                query_type=query_type
            )
        
        return None
    
    def should_trigger_prefetch(self, intent: Optional[Intent]) -> bool:
        """Determine if we should trigger speculative retrieval."""
        if intent is None:
            return False
        return intent.confidence >= self.confidence_threshold
    
    def get_prefetch_query(self, intent: Intent) -> str:
        """Generate a query for speculative retrieval."""
        keywords = " ".join(intent.keywords[:3])
        if intent.query_type == "troubleshooting":
            return f"{intent.domain} {keywords} troubleshooting"
        elif intent.query_type == "configuration":
            return f"{intent.domain} {keywords} configuration setup"
        else:
            return f"{intent.domain} {keywords}"
