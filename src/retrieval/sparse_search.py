"""Sparse BM25 search for keyword matching."""

import asyncio
import pickle
from pathlib import Path
from typing import List, Optional, Tuple
import re

from rank_bm25 import BM25Okapi

from ..config import get_settings


class SparseSearcher:
    """
    BM25 keyword search for exact term matching.
    
    Complements dense search for error codes, table references,
    and other exact-match scenarios.
    """
    
    def __init__(self, index_path: Optional[str] = None):
        """Initialize sparse searcher."""
        self.settings = get_settings()
        self.index_path = Path(index_path or self.settings.index_dir)
        
        self._bm25 = None
        self._documents = []
        self._doc_ids = []
        self._tokenized_docs = []
    
    def load_index(self) -> bool:
        """Load BM25 index from disk."""
        try:
            index_file = self.index_path / "bm25.pkl"
            
            if not index_file.exists():
                return False
            
            with open(index_file, "rb") as f:
                data = pickle.load(f)
                self._documents = data["documents"]
                self._doc_ids = data["doc_ids"]
                self._tokenized_docs = data["tokenized_docs"]
                self._bm25 = BM25Okapi(self._tokenized_docs)
            
            return True
            
        except Exception as e:
            print(f"Error loading BM25 index: {e}")
            return False
    
    def build_index(
        self, 
        documents: List[str], 
        doc_ids: Optional[List[str]] = None
    ) -> None:
        """
        Build BM25 index from documents.
        
        Args:
            documents: List of document chunks
            doc_ids: Optional document IDs
        """
        self._documents = documents
        self._doc_ids = doc_ids or [str(i) for i in range(len(documents))]
        
        # Tokenize documents
        self._tokenized_docs = [self._tokenize(doc) for doc in documents]
        
        # Build BM25 index
        self._bm25 = BM25Okapi(self._tokenized_docs)
        
        # Save to disk
        self._save_index()
    
    def _save_index(self) -> None:
        """Save index to disk."""
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        index_file = self.index_path / "bm25.pkl"
        
        with open(index_file, "wb") as f:
            pickle.dump({
                "documents": self._documents,
                "doc_ids": self._doc_ids,
                "tokenized_docs": self._tokenized_docs
            }, f)
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25."""
        # Lowercase and split on non-alphanumeric
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        
        # Keep technical terms intact (e.g., error codes)
        # but also split compound terms
        expanded = []
        for token in tokens:
            expanded.append(token)
            # Split camelCase and snake_case
            if '_' in token:
                expanded.extend(token.split('_'))
            # Keep numbers with letters together (e.g., "raid5")
        
        return expanded
    
    async def search(
        self, 
        query: str, 
        top_k: int = 50
    ) -> List[Tuple[str, str, float]]:
        """
        Search for documents matching query.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of (doc_id, document, score) tuples
        """
        if self._bm25 is None:
            if not self.load_index():
                return []
        
        # Tokenize query
        query_tokens = self._tokenize(query)
        
        # Get BM25 scores
        loop = asyncio.get_event_loop()
        scores = await loop.run_in_executor(
            None,
            lambda: self._bm25.get_scores(query_tokens)
        )
        
        # Get top-k indices
        top_indices = scores.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only non-zero scores
                results.append((
                    self._doc_ids[idx],
                    self._documents[idx],
                    float(scores[idx])
                ))
        
        return results
