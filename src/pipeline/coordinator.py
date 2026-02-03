"""Pipeline coordinator for speculative execution."""

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import AsyncIterator, List, Optional

from ..asr import StreamingASR, TranscriptEvent, IntentDetector
from ..query import QueryRewriter, ContextManager
from ..retrieval import HybridMerger, CrossEncoderReranker, SearchResult
from ..generation import LLMClient, FillerGenerator
from ..voice import VoiceOptimizer, Pyttsx3TTSClient
from ..config import get_settings
from .metrics import LatencyMetrics, MetricsTracker


@dataclass
class PipelineResult:
    """Result from pipeline execution."""
    text_response: str
    audio_chunks: List[bytes] = field(default_factory=list)
    metrics: Optional[LatencyMetrics] = None
    documents_used: List[str] = field(default_factory=list)


class PipelineCoordinator:
    """
    Coordinates speculative RAG pipeline execution.
    
    Key optimizations:
    - Parallel ASR + intent detection
    - Speculative retrieval from partial transcripts
    - Filler generation during processing
    - Streaming LLM + TTS overlap
    """
    
    def __init__(self):
        """Initialize pipeline coordinator."""
        self.settings = get_settings()
        
        # Initialize components
        self.asr = StreamingASR()
        self.intent_detector = IntentDetector()
        self.context_manager = ContextManager()
        self.query_rewriter = QueryRewriter(self.context_manager)
        self.hybrid_search = HybridMerger()
        self.reranker = CrossEncoderReranker()
        self.llm = LLMClient()
        self.filler = FillerGenerator()
        self.voice_optimizer = VoiceOptimizer()
        self.tts = Pyttsx3TTSClient()
        
        # Metrics
        self.metrics_tracker = MetricsTracker()
    
    async def process_audio_stream(
        self,
        audio_stream: AsyncIterator[bytes],
        session_id: str
    ) -> AsyncIterator[bytes]:
        """
        Process streaming audio and yield audio response chunks.
        
        This is the main entry point for the speculative pipeline.
        """
        request_id = str(uuid.uuid4())
        metrics = self.metrics_tracker.create_metrics(request_id)
        
        # Stage 1: Streaming ASR with early intent detection
        metrics.mark('asr_start')
        
        final_transcript = ""
        speculative_results: List[SearchResult] = []
        intent = None
        filler_task = None
        
        async for event in self.asr.transcribe_stream(audio_stream):
            if event.type.value == "partial":
                if metrics.asr_first_partial is None:
                    metrics.mark('asr_first_partial')
                
                # Try intent detection on partial
                if intent is None:
                    new_intent = self.intent_detector.detect(event.text)
                    if self.intent_detector.should_trigger_prefetch(new_intent):
                        intent = new_intent
                        metrics.mark('intent_detected')
                        
                        # Trigger speculative retrieval
                        prefetch_query = self.intent_detector.get_prefetch_query(intent)
                        metrics.mark('prefetch_started')
                        
                        speculative_results = await self.hybrid_search.search(
                            prefetch_query, 
                            top_k=20
                        )
                        metrics.mark('prefetch_complete')
                        
                        # Start filler generation
                        filler_text = self.filler.get_filler(
                            query_type=intent.query_type,
                            domain=intent.domain
                        )
                        filler_task = asyncio.create_task(
                            self._generate_filler_audio(filler_text, metrics)
                        )
            
            elif event.type.value == "final":
                final_transcript = event.text
                metrics.mark('asr_end')
                break
        
        if not final_transcript:
            return
        
        # Yield filler audio first if available
        if filler_task:
            filler_audio = await filler_task
            if filler_audio:
                metrics.mark('filler_tts_ready')
                yield filler_audio
        
        # Stage 2: Query rewriting (parallel with filler)
        metrics.mark('query_rewrite_start')
        rewritten = await self.query_rewriter.rewrite(final_transcript, session_id)
        metrics.mark('query_rewrite_end')
        
        # Update context
        self.context_manager.add_user_message(
            session_id, 
            final_transcript,
            rewritten.rewritten
        )
        
        # Stage 3: Final retrieval (merge with speculative if available)
        metrics.mark('retrieval_start')
        final_results = await self.hybrid_search.search_with_prefetch(
            rewritten.rewritten,
            prefetch_results=speculative_results if speculative_results else None,
            top_k=50
        )
        metrics.mark('hybrid_complete')
        
        # Stage 4: Reranking with timeout
        metrics.mark('rerank_start')
        top_docs = await self.reranker.rerank_with_fallback(
            rewritten.rewritten,
            final_results,
            top_k=self.settings.top_k_rerank
        )
        metrics.mark('rerank_end')
        
        # Stage 5: LLM streaming generation
        metrics.mark('llm_start')
        
        response_chunks = []
        first_chunk = True
        
        async for chunk in self.llm.generate_streaming(rewritten.rewritten, top_docs):
            if first_chunk and chunk.text:
                metrics.mark('llm_first_token')
                first_chunk = False
            
            response_chunks.append(chunk.text)
            
            # Yield TTS audio as LLM generates
            if chunk.text and len(chunk.text) > 20:  # Buffer a bit
                optimized = self.voice_optimizer.optimize(chunk.text)
                if optimized:
                    metrics.mark('tts_start')
                    audio = await self.tts.synthesize(optimized)
                    if metrics.tts_first_byte is None:
                        metrics.mark('tts_first_byte')
                    yield audio
            
            if chunk.is_complete:
                metrics.mark('llm_complete')
        
        # Synthesize any remaining text
        remaining_text = "".join(response_chunks)
        if remaining_text:
            # optimized = self.voice_optimizer.optimize(remaining_text)
            # audio = await self.tts.synthesize(optimized)
            pass # TTS disabled
            yield audio
        
        metrics.mark('tts_complete')
        
        # Update context with response
        full_response = "".join(response_chunks)
        topics = [doc.doc_id for doc in top_docs[:3]]
        self.context_manager.add_assistant_message(session_id, full_response, topics)
    
    async def process_text_query(
        self,
        query: str,
        session_id: str = "default"
    ) -> PipelineResult:
        """
        Process a text query (for testing without audio).
        
        Returns full result with text and optional audio.
        """
        request_id = str(uuid.uuid4())
        metrics = self.metrics_tracker.create_metrics(request_id)
        
        # Detect intent
        intent = self.intent_detector.detect(query)
        if intent:
            metrics.mark('intent_detected')
        
        # Query rewriting
        metrics.mark('query_rewrite_start')
        rewritten = await self.query_rewriter.rewrite(query, session_id)
        metrics.mark('query_rewrite_end')
        
        self.context_manager.add_user_message(session_id, query, rewritten.rewritten)
        
        # Retrieval
        metrics.mark('retrieval_start')
        results = await self.hybrid_search.search(rewritten.rewritten, top_k=50)
        metrics.mark('hybrid_complete')
        
        # Reranking
        metrics.mark('rerank_start')
        top_docs = await self.reranker.rerank_with_fallback(
            rewritten.rewritten, results, top_k=self.settings.top_k_rerank
        )
        metrics.mark('rerank_end')
        
        # LLM generation
        metrics.mark('llm_start')
        response = await self.llm.generate(rewritten.rewritten, top_docs)
        metrics.mark('llm_complete')
        
        optimized = response  # Use response directly
        
        # Optional TTS
        audio_chunks = []
        if self.tts.is_available():
            metrics.mark('tts_start')
            audio = await self.tts.synthesize(response)
            audio_chunks.append(audio)
            metrics.mark('tts_complete')
        
        # Update context
        topics = [doc.doc_id for doc in top_docs[:3]]
        self.context_manager.add_assistant_message(session_id, response, topics)
        
        return PipelineResult(
            text_response=optimized,
            audio_chunks=audio_chunks,
            metrics=metrics,
            documents_used=[doc.doc_id for doc in top_docs]
        )
    
    async def _generate_filler_audio(
        self, 
        filler_text: str,
        metrics: LatencyMetrics
    ) -> Optional[bytes]:
        """Generate filler audio asynchronously."""
        metrics.mark('filler_start')
        try:
            if self.tts.is_available():
                audio = await self.tts.synthesize(filler_text)
                return audio
        except Exception:
            pass
        return None
    
    def get_metrics_summary(self) -> dict:
        """Get metrics summary for monitoring."""
        return {
            "stats": self.metrics_tracker.get_stats(),
            "component_breakdown": self.metrics_tracker.get_component_breakdown()
        }
