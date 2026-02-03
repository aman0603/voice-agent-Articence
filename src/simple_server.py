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
                print("LLM enabled: NVIDIA")
            # Prefer OpenRouter (free tier)
            elif self.settings.open_router_key:
                from .generation.openrouter_client import OpenRouterClient
                self.llm = OpenRouterClient()
                self.llm_available = True
                print("LLM enabled: OpenRouter (Trinity model)")
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
        
        
        # TTS disabled - using browser-based TTS instead
        # All local TTS options (Piper, Edge-TTS, pyttsx3, Coqui) failed on Windows
        self.tts = None
    
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
        
        # Fall back to demo response if no LLM response
        if not response:
            ttfb = (time.time() - start_time) * 1000
            response = self._generate_demo_response(query, intent)
            documents = []
        
        # Voice optimization
        optimized = self.voice_optimizer.optimize(response)
        optimized = self.phonetic.convert(optimized)
        
        total_time = (time.time() - start_time) * 1000
        
        # Update context
        self.context_manager.add_assistant_message(session_id, optimized, [intent_domain])
        
        return SimpleResult(
            text_response=optimized,
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
                        if not ttfb_recorded:
                            ttfb = (time.time() - llm_start) * 1000
                            ttfb_recorded = True
                            yield {"type": "ttfb", "ttfb_ms": ttfb}
                        
                        full_text += chunk.text
                        sentence_buffer += chunk.text
                        
                        # Check for complete sentences
                        sentences = self._split_sentences(sentence_buffer)
                        if len(sentences) > 1:
                            # Yield complete sentences (skip voice optimization for browser TTS)
                            for sentence in sentences[:-1]:
                                # Browser TTS doesn't need phonetic conversion - yield raw text
                                yield {
                                    "type": "sentence",
                                    "text": sentence,
                                    "is_final": False
                                }
                                
                                # Immediately synthesize and yield audio (don't defer)
                                # Edge-TTS is fast (~100-200ms), so per-sentence is fine
                                if self.tts:
                                    audio_result = await self._synthesize_audio_chunk(optimized)
                                    if audio_result:
                                        yield audio_result
                            sentence_buffer = sentences[-1]
                
                # Yield any remaining text
                if sentence_buffer.strip():
                    # Browser TTS doesn't need voice optimization
                    yield {
                        "type": "sentence",
                        "text": sentence_buffer,
                        "is_final": True
                    }
                    
                    # Synthesize audio for final sentence
                    if self.tts:
                        audio_result = await self._synthesize_audio_chunk(optimized)
                        if audio_result:
                            yield audio_result
                
            except Exception as e:
                print(f"LLM Streaming Error: {e}")
                # Fall through to demo
                full_text = ""
        
        # Demo fallback if no streaming happened
        if not full_text:
            ttfb = (time.time() - start_time) * 1000
            yield {"type": "ttfb", "ttfb_ms": ttfb}
            
            response = self._generate_demo_response(query, intent)
            sentences = self._split_sentences(response)
            
            for i, sentence in enumerate(sentences):
                optimized = self.voice_optimizer.optimize(sentence)
                optimized = self.phonetic.convert(optimized)
                yield {
                    "type": "sentence",
                    "text": optimized,
                    "is_final": i == len(sentences) - 1
                }
                await asyncio.sleep(0.05)  # Small delay for demo
            
            full_text = response
        
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
        try:
            import base64
            audio_bytes = await self.tts.synthesize(text)
            if audio_bytes:
                b64_audio = base64.b64encode(audio_bytes).decode('utf-8')
                return {
                    "type": "audio",
                    "data": b64_audio,
                    "text": text
                }
        except Exception as e:
            print(f"TTS synthesis error: {e}")
        return None
    
    def _generate_demo_response(self, query: str, intent) -> str:
        """Generate a demo response based on intent detection."""
        query_lower = query.lower()
        
        if intent and intent.domain == "power":
            if "redundancy" in query_lower or "modes" in query_lower:
                return """There are three power redundancy modes available on Dell PowerEdge servers. 
                First, Grid Redundancy Mode divides power supplies into two grids. 
                Second, Power Supply Redundancy Mode allows remaining supplies to take over if one fails. 
                Third, No Redundancy Mode shares load with no backup. 
                Most production environments use the first or second mode."""
            elif "led" in query_lower or "amber" in query_lower or "blinking" in query_lower:
                return """An amber blinking LED on the power supply indicates a warning condition. 
                First, check if the power cable is properly connected. 
                Second, verify the power outlet is functioning. 
                Third, ensure the power supply is fully seated. 
                If the problem persists, the power supply may need replacement."""
            else:
                return """For power supply issues, check the LED indicators first. 
                A solid green light means the power supply is operating normally. 
                An amber light indicates a problem that needs attention. 
                Would you like me to help troubleshoot a specific issue?"""
        
        elif intent and intent.domain == "storage":
            if "raid" in query_lower and ("5" in query_lower or "five" in query_lower):
                return """To configure RAID 5, you'll need at least three drives. 
                First, access the BIOS and go to Device Settings. 
                Then select Controller Management and choose Create Virtual Disk. 
                Select RAID 5 and choose your drives. 
                For the stripe size, 64 kilobytes works well for most workloads."""
            elif "replace" in query_lower or "failed" in query_lower or "drive" in query_lower:
                return """To replace a failed drive, first identify it by the amber LED. 
                Hot-swap capable drives can be replaced while the server is running. 
                Pull the drive release latch and slide out the failed drive. 
                Insert the new drive until it clicks into place. 
                The RAID controller will automatically rebuild the array."""
            else:
                return """Dell PowerEdge servers support multiple RAID configurations. 
                RAID 0 provides maximum performance but no redundancy. 
                RAID 1 mirrors two drives for redundancy. 
                RAID 5 and 6 provide a balance of performance and protection. 
                What specific storage configuration would you like to learn about?"""
        
        elif intent and intent.domain == "network":
            return """For network configuration, start by checking the NIC LEDs. 
            A solid green link light means the connection is active. 
            If the link light is off, check the cable and switch port. 
            For 25 gigabit ports, ensure you're using compatible SFP28 modules."""
        
        elif intent and intent.domain == "memory":
            return """Memory should be installed following the population rules. 
            Start with slot A1, then A2, A3, and A4. 
            Always use matching DDR4 ECC memory within a channel. 
            If you see memory errors, check the System Event Log for details."""
        
        else:
            return f"""I understand you're asking about: {query}. 
            I can help with Dell PowerEdge server troubleshooting. 
            Ask me about power supply issues, RAID configuration, network setup, or memory installation. 
            What specific topic would you like to explore?"""


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
    Events:
      - filler: Initial filler while processing
      - ttfb: Time to first byte recorded
      - sentence: A complete sentence ready for TTS
      - done: Stream complete with final metrics
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
        "src.simple_server:app",
        host=settings.host,
        port=settings.port,
        reload=True
    )


if __name__ == "__main__":
    main()
