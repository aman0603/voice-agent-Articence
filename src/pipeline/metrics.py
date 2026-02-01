"""Latency metrics tracking for pipeline optimization."""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime
import statistics


@dataclass
class LatencyMetrics:
    """Metrics for a single pipeline execution."""
    request_id: str
    start_time: float = field(default_factory=time.time)
    
    # Component timings (ms)
    asr_start: Optional[float] = None
    asr_first_partial: Optional[float] = None
    asr_end: Optional[float] = None
    
    intent_detected: Optional[float] = None
    prefetch_started: Optional[float] = None
    prefetch_complete: Optional[float] = None
    
    query_rewrite_start: Optional[float] = None
    query_rewrite_end: Optional[float] = None
    
    retrieval_start: Optional[float] = None
    dense_complete: Optional[float] = None
    sparse_complete: Optional[float] = None
    hybrid_complete: Optional[float] = None
    
    rerank_start: Optional[float] = None
    rerank_end: Optional[float] = None
    
    llm_start: Optional[float] = None
    llm_first_token: Optional[float] = None
    llm_complete: Optional[float] = None
    
    filler_start: Optional[float] = None
    filler_tts_ready: Optional[float] = None
    
    tts_start: Optional[float] = None
    tts_first_byte: Optional[float] = None
    tts_complete: Optional[float] = None
    
    def mark(self, event: str) -> None:
        """Mark an event time."""
        elapsed = (time.time() - self.start_time) * 1000  # ms
        setattr(self, event, elapsed)
    
    @property
    def ttfb(self) -> Optional[float]:
        """Time to first byte of audio."""
        if self.tts_first_byte is not None:
            return self.tts_first_byte
        if self.filler_tts_ready is not None:
            return self.filler_tts_ready
        return None
    
    @property
    def total_latency(self) -> Optional[float]:
        """Total pipeline latency."""
        if self.tts_complete is not None:
            return self.tts_complete
        return None
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for logging."""
        return {
            k: v for k, v in self.__dict__.items()
            if v is not None and k != 'start_time'
        }
    
    def get_breakdown(self) -> Dict[str, float]:
        """Get latency breakdown by component."""
        breakdown = {}
        
        # ASR latency
        if self.asr_end and self.asr_start:
            breakdown['asr'] = self.asr_end - (self.asr_start or 0)
        
        # Query rewriting
        if self.query_rewrite_end and self.query_rewrite_start:
            breakdown['query_rewrite'] = self.query_rewrite_end - self.query_rewrite_start
        
        # Retrieval
        if self.hybrid_complete and self.retrieval_start:
            breakdown['retrieval'] = self.hybrid_complete - self.retrieval_start
        
        # Reranking
        if self.rerank_end and self.rerank_start:
            breakdown['rerank'] = self.rerank_end - self.rerank_start
        
        # LLM
        if self.llm_first_token and self.llm_start:
            breakdown['llm_to_first_token'] = self.llm_first_token - self.llm_start
        
        # TTS
        if self.tts_first_byte and self.tts_start:
            breakdown['tts'] = self.tts_first_byte - self.tts_start
        
        return breakdown


class MetricsTracker:
    """
    Tracks latency metrics across requests for monitoring and optimization.
    """
    
    def __init__(self, max_history: int = 1000):
        """Initialize metrics tracker."""
        self.max_history = max_history
        self.metrics: List[LatencyMetrics] = []
        self.target_ttfb_ms = 800
    
    def create_metrics(self, request_id: str) -> LatencyMetrics:
        """Create new metrics for a request."""
        metrics = LatencyMetrics(request_id=request_id)
        self.metrics.append(metrics)
        
        # Trim history
        if len(self.metrics) > self.max_history:
            self.metrics = self.metrics[-self.max_history:]
        
        return metrics
    
    def get_stats(self) -> Dict:
        """Get aggregate statistics."""
        if not self.metrics:
            return {}
        
        ttfb_values = [m.ttfb for m in self.metrics if m.ttfb is not None]
        
        if not ttfb_values:
            return {"total_requests": len(self.metrics)}
        
        return {
            "total_requests": len(self.metrics),
            "ttfb_avg_ms": statistics.mean(ttfb_values),
            "ttfb_p50_ms": statistics.median(ttfb_values),
            "ttfb_p95_ms": self._percentile(ttfb_values, 95),
            "ttfb_p99_ms": self._percentile(ttfb_values, 99),
            "ttfb_min_ms": min(ttfb_values),
            "ttfb_max_ms": max(ttfb_values),
            "target_met_pct": sum(1 for v in ttfb_values if v < self.target_ttfb_ms) / len(ttfb_values) * 100
        }
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile value."""
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def get_component_breakdown(self) -> Dict[str, Dict]:
        """Get breakdown by component across all requests."""
        components = ['asr', 'query_rewrite', 'retrieval', 'rerank', 'llm_to_first_token', 'tts']
        breakdown = {}
        
        for component in components:
            values = []
            for m in self.metrics:
                b = m.get_breakdown()
                if component in b:
                    values.append(b[component])
            
            if values:
                breakdown[component] = {
                    "avg_ms": statistics.mean(values),
                    "p95_ms": self._percentile(values, 95),
                    "max_ms": max(values)
                }
        
        return breakdown
