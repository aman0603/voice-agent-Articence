"""ASR module - Streaming speech recognition with Whisper."""

# Lazy imports to avoid requiring whisper at import time
from .intent_detector import IntentDetector, Intent

# These require whisper, import only when needed
def get_streaming_asr():
    """Get StreamingASR class (requires whisper)."""
    from .streaming_asr import StreamingASR
    return StreamingASR

def get_transcript_event():
    """Get TranscriptEvent class."""
    from .streaming_asr import TranscriptEvent
    return TranscriptEvent

__all__ = ["IntentDetector", "Intent", "get_streaming_asr", "get_transcript_event"]
