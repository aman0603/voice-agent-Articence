"""FastAPI server for Voice RAG Agent."""

import asyncio
from typing import AsyncIterator
from contextlib import asynccontextmanager
import uuid

from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .config import get_settings
from .pipeline import PipelineCoordinator


# Global coordinator instance
coordinator: PipelineCoordinator = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global coordinator
    coordinator = PipelineCoordinator()
    yield
    # Cleanup if needed


app = FastAPI(
    title="Voice RAG Agent",
    description="Zero-Latency Voice Knowledge-Base RAG System",
    version="0.1.0",
    lifespan=lifespan
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


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "healthy", "service": "voice-rag-agent"}


@app.get("/metrics")
async def get_metrics():
    """Get pipeline metrics."""
    if coordinator is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    return coordinator.get_metrics_summary()


@app.post("/query/text", response_model=TextQueryResponse)
async def text_query(request: TextQueryRequest):
    """
    Process a text query (for testing without audio).
    
    Returns the voice-optimized response and latency metrics.
    """
    if coordinator is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    result = await coordinator.process_text_query(
        request.query,
        request.session_id
    )
    
    return TextQueryResponse(
        response=result.text_response,
        ttfb_ms=result.metrics.ttfb or 0,
        total_latency_ms=result.metrics.total_latency or 0,
        documents_used=result.documents_used
    )


@app.websocket("/ws/voice")
async def voice_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for streaming voice input/output.
    
    Protocol:
    - Client sends binary audio chunks (16kHz, 16-bit PCM)
    - Server responds with binary audio chunks (WAV)
    - Text messages for control: {"action": "start_session"/"end_session"}
    """
    await websocket.accept()
    session_id = str(uuid.uuid4())
    
    try:
        while True:
            message = await websocket.receive()
            
            if "bytes" in message:
                # Audio data - create async iterator
                async def audio_stream() -> AsyncIterator[bytes]:
                    yield message["bytes"]
                    try:
                        while True:
                            msg = await asyncio.wait_for(
                                websocket.receive(),
                                timeout=5.0
                            )
                            if "bytes" in msg:
                                yield msg["bytes"]
                            elif "text" in msg:
                                import json
                                data = json.loads(msg["text"])
                                if data.get("action") == "end_turn":
                                    break
                    except asyncio.TimeoutError:
                        pass
                
                # Process and stream response
                async for audio_chunk in coordinator.process_audio_stream(
                    audio_stream(),
                    session_id
                ):
                    await websocket.send_bytes(audio_chunk)
                
                # Signal end of response
                await websocket.send_json({"status": "complete"})
            
            elif "text" in message:
                import json
                data = json.loads(message["text"])
                
                if data.get("action") == "end_session":
                    break
                elif data.get("action") == "ping":
                    await websocket.send_json({"status": "pong"})
    
    except Exception as e:
        await websocket.send_json({"error": str(e)})
    finally:
        await websocket.close()


def main():
    """Run the server."""
    import uvicorn
    settings = get_settings()
    
    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        reload=True
    )


if __name__ == "__main__":
    main()
