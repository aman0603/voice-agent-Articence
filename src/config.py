"""Configuration management for Voice RAG Agent."""

from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment."""
    
    # API Keys
    gemini_api_key: str = Field(default="", description="Google Gemini API key")
    open_router_key: str = Field(default="", alias="OPEN_ROUTER", description="OpenRouter API key")
    nvidia_api_key: str = Field(default="", description="NVIDIA API key")
    
    # Model Settings
    whisper_model: str = Field(default="base", description="Whisper model size")
    piper_voice: str = Field(default="en_US-lessac-medium", description="Piper voice model")
    
    # Embedding & Reranking
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Sentence transformer model for embeddings"
    )
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Cross-encoder model for reranking"
    )
    
    # Retrieval Settings
    top_k_retrieval: int = Field(default=50, description="Number of docs for initial retrieval")
    top_k_rerank: int = Field(default=5, description="Number of docs after reranking")
    chunk_size: int = Field(default=512, description="Document chunk size in characters")
    chunk_overlap: int = Field(default=50, description="Overlap between chunks")
    
    # Latency Targets (ms)
    target_ttfb: int = Field(default=800, description="Target time to first byte in ms")
    rerank_timeout: int = Field(default=150, description="Max reranking time in ms")
    
    # Paths
    data_dir: str = Field(default="data", description="Data directory path")
    manuals_dir: str = Field(default="data/manuals", description="Manuals directory")
    index_dir: str = Field(default="data/index", description="Index directory")
    
    # Server
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    
    # Voice Optimization
    max_sentence_words: int = Field(default=15, description="Max words per sentence for TTS")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
