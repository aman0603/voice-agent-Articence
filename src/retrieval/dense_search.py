"""Dense vector search using FAISS and sentence transformers."""

import asyncio
import pickle
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np

from sentence_transformers import SentenceTransformer

from ..config import get_settings


class DenseSearcher:
    """
    Dense vector search using FAISS for fast similarity search.
    
    Optimized for low-latency retrieval with HNSW index.
    """
    
    def __init__(
        self, 
        model_name: Optional[str] = None,
        index_path: Optional[str] = None
    ):
        """Initialize dense searcher."""
        self.settings = get_settings()
        self.model_name = model_name or self.settings.embedding_model
        self.index_path = Path(index_path or self.settings.index_dir)
        
        self._model = None
        self._index = None
        self._documents = []
        self._doc_ids = []
        
    @property
    def model(self) -> SentenceTransformer:
        """Lazy load embedding model."""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model
    
    def load_index(self) -> bool:
        """Load FAISS index from disk."""
        try:
            import faiss
            
            index_file = self.index_path / "dense.index"
            docs_file = self.index_path / "documents.pkl"
            
            if not index_file.exists() or not docs_file.exists():
                return False
            
            self._index = faiss.read_index(str(index_file))
            
            with open(docs_file, "rb") as f:
                data = pickle.load(f)
                self._documents = data["documents"]
                self._doc_ids = data["doc_ids"]
            
            return True
            
        except Exception as e:
            print(f"Error loading index: {e}")
            return False
    
    def build_index(
        self, 
        documents: List[str], 
        doc_ids: Optional[List[str]] = None
    ) -> None:
        """
        Build FAISS index from documents.
        
        Args:
            documents: List of document chunks
            doc_ids: Optional document IDs
        """
        import faiss
        
        self._documents = documents
        self._doc_ids = doc_ids or [str(i) for i in range(len(documents))]
        
        # Generate embeddings
        embeddings = self.model.encode(
            documents, 
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Build HNSW index for speed
        dim = embeddings.shape[1]
        self._index = faiss.IndexHNSWFlat(dim, 32)  # 32 neighbors
        self._index.hnsw.efConstruction = 200
        self._index.add(embeddings)
        
        # Set search params
        self._index.hnsw.efSearch = 64  # Balance speed/accuracy
        
        # Save to disk
        self._save_index()
    
    def _save_index(self) -> None:
        """Save index to disk."""
        import faiss
        
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        index_file = self.index_path / "dense.index"
        docs_file = self.index_path / "documents.pkl"
        
        faiss.write_index(self._index, str(index_file))
        
        with open(docs_file, "wb") as f:
            pickle.dump({
                "documents": self._documents,
                "doc_ids": self._doc_ids
            }, f)
    
    async def search(
        self, 
        query: str, 
        top_k: int = 50
    ) -> List[Tuple[str, str, float]]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of (doc_id, document, score) tuples
        """
        if self._index is None:
            if not self.load_index():
                return []
        
        # Encode query
        loop = asyncio.get_event_loop()
        query_embedding = await loop.run_in_executor(
            None,
            lambda: self.model.encode([query], convert_to_numpy=True)
        )
        
        import faiss
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = await loop.run_in_executor(
            None,
            lambda: self._index.search(query_embedding, top_k)
        )
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:  # Valid index
                results.append((
                    self._doc_ids[idx],
                    self._documents[idx],
                    float(score)
                ))
        
        return results
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a single text."""
        return self.model.encode([text], convert_to_numpy=True)[0]
