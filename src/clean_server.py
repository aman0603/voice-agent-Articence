"""Simplified FastAPI server for Voice RAG Agent (text queries only)."""

import asyncio
import time
import uuid
from typing import List, Optional
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
import json

from .config import get_settings
from .asr.intent_detector import IntentDetector
from .query.context_manager import ContextManager
from .query.rewriter import QueryRewriter
from .voice.optimizer import VoiceOptimizer
from .voice.phonetic import PhoneticConverter
from .generation.filler_generator import FillerGenerator
from .voice.kokoro_tts import KokoroTTS
import base64


@dataclass
class SimpleResult:
    """Result from simplified pipeline."""
    text_response: str
    ttfb_ms: float = 0
    total_latency_ms: float = 0
    documents_used: List[str] = field(default_factory=list)
    intent_domain: str = ""
    query_rewritten: str = ""


class SimplePipeline:
    """Simplified pipeline for text-only queries (no whisper/LLM API required)."""
    
    def __init__(self):
        self.settings = get_settings()
        self.intent_detector = IntentDetector()
        self.context_manager = ContextManager()
        self.query_rewriter = QueryRewriter(self.context_manager)
        self.voice_optimizer = VoiceOptimizer()
        self.phonetic = PhoneticConverter()
        self.filler = FillerGenerator()
        
        # Try to load retrieval components
        self.retrieval_available = False
        self.llm_available = False
        
        try:
            from .retrieval import HybridMerger, CrossEncoderReranker
            self.hybrid_search = HybridMerger()
            self.reranker = CrossEncoderReranker()
            self.retrieval_available = True
        except Exception as e:
            print(f"Retrieval not available: {e}")
        
        try:
            #add nvidia api key check
            if self.settings.nvidia_api_key:
                from .generation.nvidia_client import NvidiaClient
                self.llm = NvidiaClient()
                self.llm_available = True
                self.llm_available = True
                print("LLM enabled: NVIDIA")
            # Fallback to Gemini if API key is configured
            elif self.settings.gemini_api_key and self.settings.gemini_api_key != "your-gemini-api-key-here":
                from .generation import LLMClient
                self.llm = LLMClient()
                self.llm_available = True
                print("LLM enabled: Gemini")
            else:
                print("LLM disabled - no API key configured (using demo mode)")
        except Exception as e:
            print(f"LLM not available: {e}")
        
        
        # Initialize Kokoro TTS for high-quality voice output
        try:
            # self.tts = KokoroTTS(voice="af_bella")
            self.tts = None  # Disabled for Browser TTS mode
            print("Kokoro TTS disabled (Browser Mode)")
        except Exception as e:
            print(f"Kokoro TTS unavailable: {e}")
            self.tts = None
        
        # Preload models to avoid cold start latency
        self._warmup_models()
    
    def _warmup_models(self):
        """Preload models to eliminate cold start latency."""
        if not self.retrieval_available:
            return
        
        print("\n🔥 Warming up models...")
        
        try:
            # 1. Warmup embedding model (dense search)
            print("  ⏳ Loading sentence transformer...")
            _ = self.hybrid_search.dense_searcher.model
            print("  ✓ Sentence transformer ready")
            
            # 2. Warmup reranker model
            print("  ⏳ Loading cross-encoder reranker...")
            self.reranker.load_model()
            print("  ✓ Cross-encoder ready")
            
            # 3. Load dense index
            if self.hybrid_search.dense_searcher._index is None:
                print("  ⏳ Loading dense index...")
                self.hybrid_search.dense_searcher.load_index()
                print("  ✓ Dense index loaded")
            
            # 4. Load sparse index
            if self.hybrid_search.sparse_searcher._bm25 is None:
                print("  ⏳ Loading sparse index...")
                self.hybrid_search.sparse_searcher.load_index()
                print("  ✓ Sparse index loaded")
            
            print("✅ Model warmup complete! Ready for fast queries.\n")
            
        except Exception as e:
            print(f"⚠️  Warmup failed (models will load on first query): {e}\n")
    
    async def process_query(self, query: str, session_id: str = "default") -> SimpleResult:
        """Process a text query."""
        start_time = time.time()
        
        # Intent detection
        intent = self.intent_detector.detect(query)
        intent_domain = intent.domain if intent else "general"
        
        # Query rewriting
        rewrite_result = await self.query_rewriter.rewrite(query, session_id, use_llm=False)
        rewritten_query = rewrite_result.rewritten
        
        self.context_manager.add_user_message(session_id, query, rewritten_query)
        
        # Generate response
        response = None
        documents = []
        ttfb = 0
        
        if self.llm_available and self.retrieval_available:
            # Full pipeline - try LLM, fallback to demo on error
            try:
                results = await self.hybrid_search.search(rewritten_query, top_k=20)
                top_docs = await self.reranker.rerank_with_fallback(
                    rewritten_query, results, top_k=5
                )
                
                # Measure TTFB - time until we get response
                llm_start = time.time()
                response = await self.llm.generate(rewritten_query, top_docs)
                ttfb = (time.time() - llm_start) * 1000  # LLM response time
                
                documents = [doc.doc_id for doc in top_docs]
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "rate" in error_str.lower():
                    print(f"Rate Limited - using demo response")
                else:
                    print(f"LLM Error: {e} - using demo response")
                response = None  # Will trigger demo below
        
        # Fall back to error message if no LLM response
        if not response:
            ttfb = (time.time() - start_time) * 1000
            response = self._generate_error_response()
            documents = []
        
        # Voice optimization removed - browser TTS doesn't need it
        # Just use the response as-is
        
        total_time = (time.time() - start_time) * 1000
        
        # Update context
        self.context_manager.add_assistant_message(session_id, response, [intent_domain])
        
        return SimpleResult(
            text_response=response,
            ttfb_ms=ttfb,
            total_latency_ms=total_time,
            documents_used=documents,
            intent_domain=intent_domain,
            query_rewritten=rewritten_query
        )
    
    async def stream_query(self, query: str, session_id: str = "default"):
        """
        Stream query response - yields chunks for real-time TTS.
        
        Yields JSON objects with text chunks and metadata.
        """
        print("🔍 DEBUG: clean_server FIXED VERSION (CLEAN SLATE)")  # Debug marker
        start_time = time.time()
        
        # Intent detection
        intent = self.intent_detector.detect(query)
        intent_domain = intent.domain if intent else "general"
        
        # Query rewriting
        rewrite_result = await self.query_rewriter.rewrite(query, session_id, use_llm=False)
        rewritten_query = rewrite_result.rewritten
        
        self.context_manager.add_user_message(session_id, query, rewritten_query)
        
        # First, yield a filler while we process
        filler = self.filler.get_filler(intent.query_type if intent else "general")
        yield {
            "type": "filler",
            "text": filler,
            "intent_domain": intent_domain,
            "query_rewritten": rewritten_query
        }
        
        # Now stream the actual response
        ttfb_recorded = False
        ttfb = 0
        full_text = ""
        sentence_buffer = ""
        llm_succeeded = False  # Track if LLM generated any output
        
        if self.llm_available and self.retrieval_available:
            try:
                # Retrieval - reduced from 20 to 10 for faster reranking
                results = await self.hybrid_search.search(rewritten_query, top_k=20)
                top_docs = await self.reranker.rerank_with_fallback(
                    rewritten_query, results, top_k=5
                )
                
                llm_start = time.time()
                
                # Stream LLM response
                async for chunk in self.llm.generate_streaming(rewritten_query, top_docs):
                    if chunk.text:
                        llm_succeeded = True  # Mark that LLM produced output
                        if not ttfb_recorded:
                            ttfb = (time.time() - llm_start) * 1000
                            ttfb_recorded = True
                            yield {"type": "ttfb", "ttfb_ms": ttfb}
                        
                        full_text += chunk.text
                        sentence_buffer += chunk.text
                        
                        # Check for complete sentences
                        sentences = self._split_sentences(sentence_buffer)
                        if len(sentences) > 1:
                            # Yield complete sentences with Kokoro TTS audio
                            for sentence in sentences[:-1]:
                                yield {
                                    "type": "sentence",
                                    "text": sentence,
                                    "is_final": False
                                }
                                
                                # Generate Kokoro TTS audio for this sentence
                                if self.tts:
                                    try:
                                        audio_bytes = await self.tts.synthesize(sentence)
                                        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
                                        yield {
                                            "type": "audio",
                                            "audio_base64": audio_base64,
                                            "format": "wav"
                                        }
                                    except Exception as e:
                                        print(f"TTS synthesis error: {e}")
                            
                            sentence_buffer = sentences[-1]
                
                # Yield any remaining text
                if sentence_buffer.strip():
                    yield {
                        "type": "sentence",
                        "text": sentence_buffer,
                        "is_final": True
                    }
                    
                    # Generate final Kokoro TTS audio
                    if self.tts:
                        try:
                            audio_bytes = await self.tts.synthesize(sentence_buffer)
                            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
                            yield {
                                "type": "audio",
                                "audio_base64": audio_base64,
                                "format": "wav"
                            }
                        except Exception as e:
                            print(f"TTS synthesis error: {e}")
                
            except Exception as e:
                import traceback
                full_trace = traceback.format_exc()
                print(f"LLM Streaming Error: {e}\n{full_trace}")
                # Also yield it for visibility
                yield {
                    "type": "error_debug",
                    "message": f"Server Error: {e}",
                    "traceback": full_trace
                }
                llm_succeeded = False  # Mark failure
                llm_succeeded = False  # Mark failure
        
        # NO demo fallback - if LLM fails, just show error
        if not llm_succeeded:
            # Just yield an error message
            yield {
                "type": "sentence",
                "text": "I'm having trouble connecting to the knowledge base. Please try again.",
                "is_final": True
            }
        
        
        total_time = (time.time() - start_time) * 1000
        
        # Final metadata
        yield {
            "type": "done",
            "total_latency_ms": total_time,
            "ttfb_ms": ttfb
        }
        
        # Update context
        self.context_manager.add_assistant_message(session_id, full_text, [intent_domain])
    
    def _split_sentences(self, text: str) -> list:
        """Split text into sentences."""
        import re
        # Split on sentence-ending punctuation followed by space or end
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    async def _synthesize_audio_chunk(self, text: str):
        """Synthesize audio for a text chunk asynchronously."""
        # This method is effectively unused now but kept for compatibility just in case
        return None
    
    def _generate_error_response(self) -> str:
        """Generate a simple error response."""
        return "I'm having trouble connecting to the knowledge base. Please try again."


# Global pipeline instance
pipeline: SimplePipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global pipeline
    pipeline = SimplePipeline()
    yield


app = FastAPI(
    title="Voice RAG Agent",
    description="Zero-Latency Voice Knowledge-Base RAG System",
    version="0.1.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TextQueryRequest(BaseModel):
    """Request for text-based query."""
    query: str
    session_id: str = "default"


class TextQueryResponse(BaseModel):
    """Response for text-based query."""
    response: str
    ttfb_ms: float
    total_latency_ms: float
    documents_used: list
    intent_domain: Optional[str] = None
    query_rewritten: Optional[str] = None


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy", 
        "service": "voice-rag-agent",
        "retrieval_available": pipeline.retrieval_available if pipeline else False,
        "llm_available": pipeline.llm_available if pipeline else False
    }


@app.get("/frontend")
async def serve_frontend():
    """Serve the frontend HTML."""
    import os
    frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "index.html")
    if os.path.exists(frontend_path):
        return FileResponse(frontend_path)
    raise HTTPException(status_code=404, detail="Frontend not found")


@app.post("/query/text", response_model=TextQueryResponse)
async def text_query(request: TextQueryRequest):
    """
    Process a text query.
    
    Returns the voice-optimized response and latency metrics.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    result = await pipeline.process_query(
        request.query,
        request.session_id
    )
    
    return TextQueryResponse(
        response=result.text_response,
        ttfb_ms=result.ttfb_ms,
        total_latency_ms=result.total_latency_ms,
        documents_used=result.documents_used,
        intent_domain=result.intent_domain,
        query_rewritten=result.query_rewritten
    )


@app.post("/query/stream")
async def stream_query(request: TextQueryRequest):
    """
    Stream query response with Server-Sent Events.
    
    Yields sentences as they are generated for real-time TTS.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    async def generate_events():
        async for chunk in pipeline.stream_query(request.query, request.session_id):
            yield f"data: {json.dumps(chunk)}\n\n"
    
    return StreamingResponse(
        generate_events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.get("/metrics")
async def get_metrics():
    """Get pipeline status."""
    return {
        "retrieval_available": pipeline.retrieval_available if pipeline else False,
        "llm_available": pipeline.llm_available if pipeline else False,
        "components": {
            "intent_detector": True,
            "query_rewriter": True,
            "voice_optimizer": True,
            "phonetic_converter": True
        }
    }


def main():
    """Run the server."""
    import uvicorn
    settings = get_settings()
    
    uvicorn.run(
        "src.clean_server:app",
        host=settings.host,
        port=settings.port,
        reload=True
    )


if __name__ == "__main__":
    main()
