"""Hybrid search result merger using RRF fusion."""

import asyncio
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

from .dense_search import DenseSearcher
from .sparse_search import SparseSearcher
from ..config import get_settings


@dataclass
class SearchResult:
    """A single search result."""
    doc_id: str
    content: str
    dense_score: float = 0.0
    sparse_score: float = 0.0
    hybrid_score: float = 0.0
    rerank_score: Optional[float] = None
    
    @property
    def final_score(self) -> float:
        """Get final ranking score."""
        if self.rerank_score is not None:
            return self.rerank_score
        return self.hybrid_score


class HybridMerger:
    """
    Merges dense and sparse search results using Reciprocal Rank Fusion.
    
    RRF provides robust fusion without requiring score calibration.
    """
    
    def __init__(
        self,
        dense_searcher: Optional[DenseSearcher] = None,
        sparse_searcher: Optional[SparseSearcher] = None,
        k: int = 60  # RRF parameter
    ):
        """Initialize hybrid merger."""
        self.settings = get_settings()
        self.dense_searcher = dense_searcher or DenseSearcher()
        self.sparse_searcher = sparse_searcher or SparseSearcher()
        self.k = k  # RRF constant
    
    async def search(
        self, 
        query: str, 
        top_k: int = 50
    ) -> List[SearchResult]:
        """
        Perform hybrid search with RRF fusion.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of SearchResult objects sorted by hybrid score
        """
        # Execute dense and sparse searches in parallel
        dense_task = asyncio.create_task(
            self.dense_searcher.search(query, top_k)
        )
        sparse_task = asyncio.create_task(
            self.sparse_searcher.search(query, top_k)
        )
        
        dense_results, sparse_results = await asyncio.gather(
            dense_task, sparse_task
        )
        
        # Merge using RRF
        return self._rrf_fusion(dense_results, sparse_results, top_k)
    
    def _rrf_fusion(
        self,
        dense_results: List[Tuple[str, str, float]],
        sparse_results: List[Tuple[str, str, float]],
        top_k: int
    ) -> List[SearchResult]:
        """
        Fuse results using Reciprocal Rank Fusion.
        
        RRF score = sum(1 / (k + rank)) for each list
        """
        # Build result objects indexed by doc_id
        results: Dict[str, SearchResult] = {}
        
        # Process dense results
        for rank, (doc_id, content, score) in enumerate(dense_results):
            if doc_id not in results:
                results[doc_id] = SearchResult(
                    doc_id=doc_id,
                    content=content,
                    dense_score=score
                )
            else:
                results[doc_id].dense_score = score
        
        # Process sparse results  
        for rank, (doc_id, content, score) in enumerate(sparse_results):
            if doc_id not in results:
                results[doc_id] = SearchResult(
                    doc_id=doc_id,
                    content=content,
                    sparse_score=score
                )
            else:
                results[doc_id].sparse_score = score
        
        # Calculate RRF scores
        # Get ranks for each result type
        dense_ranks = {doc_id: rank for rank, (doc_id, _, _) in enumerate(dense_results)}
        sparse_ranks = {doc_id: rank for rank, (doc_id, _, _) in enumerate(sparse_results)}
        
        for doc_id, result in results.items():
            rrf_score = 0.0
            
            if doc_id in dense_ranks:
                rrf_score += 1.0 / (self.k + dense_ranks[doc_id])
            
            if doc_id in sparse_ranks:
                rrf_score += 1.0 / (self.k + sparse_ranks[doc_id])
            
            result.hybrid_score = rrf_score
        
        # Sort by hybrid score and return top-k
        sorted_results = sorted(
            results.values(),
            key=lambda x: x.hybrid_score,
            reverse=True
        )
        
        return sorted_results[:top_k]
    
    async def search_with_prefetch(
        self,
        final_query: str,
        prefetch_results: Optional[List[SearchResult]] = None,
        top_k: int = 50
    ) -> List[SearchResult]:
        """
        Hybrid search with optional speculative prefetch results.
        
        If prefetch results are available, merge them with new results
        to avoid redundant retrieval.
        """
        new_results = await self.search(final_query, top_k)
        
        if not prefetch_results:
            return new_results
        
        # Merge prefetch and new results, preferring new results
        merged = {r.doc_id: r for r in prefetch_results}
        for r in new_results:
            merged[r.doc_id] = r  # Overwrite with new scores
        
        # Re-sort
        sorted_results = sorted(
            merged.values(),
            key=lambda x: x.hybrid_score,
            reverse=True
        )
        
        return sorted_results[:top_k]
