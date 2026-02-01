"""Cross-encoder reranker with early exit optimization."""

import asyncio
import time
from typing import List, Optional

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

from .hybrid_merger import SearchResult
from ..config import get_settings


class CrossEncoderReranker:
    """
    Cross-encoder reranker for final relevance scoring.
    
    Features:
    - Early exit when high-confidence result found
    - Timeout enforcement more latency control
    - Batched inference for efficiency
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        early_exit_threshold: float = 0.9,
        timeout_ms: Optional[int] = None
    ):
        """Initialize reranker."""
        self.settings = get_settings()
        self.model_name = model_name or self.settings.reranker_model
        self.early_exit_threshold = early_exit_threshold
        self.timeout_ms = timeout_ms or self.settings.rerank_timeout
        
        self._model = None
        self._tokenizer = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def load_model(self) -> None:
        """Load model and tokenizer."""
        if self._model is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name
            )
            self._model.to(self._device)
            self._model.eval()
    
    async def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int = 5
    ) -> List[SearchResult]:
        """
        Rerank results using cross-encoder.
        
        Args:
            query: Search query
            results: List of SearchResult to rerank
            top_k: Number of top results to return
            
        Returns:
            Top-k results sorted by rerank score
        """
        if not results:
            return []
        
        self.load_model()
        
        start_time = time.time()
        timeout_sec = self.timeout_ms / 1000.0
        
        # Limit candidates to rerank (for latency)
        candidates = results[:20]  # Max 20 candidates
        
        # Run inference
        loop = asyncio.get_event_loop()
        scored_results = await loop.run_in_executor(
            None,
            lambda: self._score_batch(query, candidates, timeout_sec, start_time)
        )
        
        # Sort by rerank score
        scored_results.sort(key=lambda x: x.rerank_score or 0, reverse=True)
        
        return scored_results[:top_k]
    
    def _score_batch(
        self,
        query: str,
        results: List[SearchResult],
        timeout_sec: float,
        start_time: float
    ) -> List[SearchResult]:
        """Score a batch of results with timeout and early exit."""
        pairs = [(query, r.content) for r in results]
        
        # Tokenize
        features = self._tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        features = {k: v.to(self._device) for k, v in features.items()}
        
        # Check timeout before inference
        elapsed = time.time() - start_time
        if elapsed >= timeout_sec:
            # Return with hybrid scores if timeout
            return results
        
        # Inference
        with torch.no_grad():
            scores = self._model(**features).logits.squeeze(-1)
            if len(scores.shape) == 0:
                scores = scores.unsqueeze(0)
            scores = torch.sigmoid(scores).cpu().numpy()
        
        # Assign scores
        for i, result in enumerate(results):
            result.rerank_score = float(scores[i])
            
            # Early exit if we find a very high confidence match
            if result.rerank_score >= self.early_exit_threshold:
                # Score remaining quickly at 0 and return
                for j in range(i + 1, len(results)):
                    results[j].rerank_score = 0.0
                break
        
        return results
    
    async def rerank_with_fallback(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int = 5
    ) -> List[SearchResult]:
        """
        Rerank with fallback to hybrid scores on timeout.
        
        Ensures we always return results even if reranking times out.
        """
        try:
            reranked = await asyncio.wait_for(
                self.rerank(query, results, top_k),
                timeout=self.timeout_ms / 1000.0
            )
            return reranked
        except asyncio.TimeoutError:
            # Fallback to hybrid scores
            sorted_results = sorted(
                results[:top_k],
                key=lambda x: x.hybrid_score,
                reverse=True
            )
            return sorted_results
