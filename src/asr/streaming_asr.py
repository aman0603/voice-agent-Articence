"""Streaming ASR using Whisper with VAD for real-time transcription."""

import asyncio
import numpy as np
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional
from enum import Enum
import whisper
import webrtcvad
from ..config import get_settings


class TranscriptType(Enum):
    """Type of transcript event."""
    PARTIAL = "partial"
    FINAL = "final"


@dataclass
class TranscriptEvent:
    """Event emitted during transcription."""
    text: str
    type: TranscriptType
    confidence: float = 1.0
    timestamp_ms: int = 0
    is_end_of_speech: bool = False


@dataclass
class AudioBuffer:
    """Circular buffer for audio samples."""
    samples: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    sample_rate: int = 16000
    max_duration_sec: float = 30.0
    
    def append(self, chunk: np.ndarray) -> None:
        """Append audio chunk to buffer."""
        self.samples = np.concatenate([self.samples, chunk])
        max_samples = int(self.max_duration_sec * self.sample_rate)
        if len(self.samples) > max_samples:
            self.samples = self.samples[-max_samples:]
    
    def get_audio(self) -> np.ndarray:
        """Get current audio buffer."""
        return self.samples
    
    def clear(self) -> None:
        """Clear the buffer."""
        self.samples = np.array([], dtype=np.float32)


class StreamingASR:
    """
    Streaming ASR using Whisper with Voice Activity Detection.
    
    Emits partial transcripts as audio arrives, enabling speculative
    query execution before speech completion.
    """
    
    def __init__(self, model_name: Optional[str] = None):
        """Initialize streaming ASR."""
        settings = get_settings()
        self.model_name = model_name or settings.whisper_model
        self.model = None
        self.vad = webrtcvad.Vad(2)  # Aggressiveness 0-3
        self.sample_rate = 16000
        self.frame_duration_ms = 30  # VAD frame duration
        self.silence_threshold_ms = 500  # End of speech detection
        
    def load_model(self) -> None:
        """Load Whisper model (lazy loading for faster startup)."""
        if self.model is None:
            self.model = whisper.load_model(self.model_name)
    
    async def transcribe_stream(
        self, 
        audio_stream: AsyncIterator[bytes],
        emit_partials: bool = True,
        partial_interval_ms: int = 300
    ) -> AsyncIterator[TranscriptEvent]:
        """
        Transcribe streaming audio with partial results.
        
        Args:
            audio_stream: Async iterator of audio bytes (16kHz, 16-bit PCM)
            emit_partials: Whether to emit partial transcripts
            partial_interval_ms: Interval between partial transcripts
            
        Yields:
            TranscriptEvent with partial or final transcripts
        """
        self.load_model()
        
        buffer = AudioBuffer(sample_rate=self.sample_rate)
        last_partial_time = 0
        silence_frames = 0
        frames_for_silence = self.silence_threshold_ms // self.frame_duration_ms
        
        async for chunk in audio_stream:
            # Convert bytes to numpy array
            audio_chunk = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
            buffer.append(audio_chunk)
            
            # Check VAD for end of speech detection
            try:
                is_speech = self.vad.is_speech(chunk, self.sample_rate)
                if not is_speech:
                    silence_frames += 1
                else:
                    silence_frames = 0
            except Exception:
                silence_frames = 0
            
            current_time = len(buffer.samples) * 1000 // self.sample_rate
            
            # Emit partial transcript
            if emit_partials and (current_time - last_partial_time) >= partial_interval_ms:
                audio = buffer.get_audio()
                if len(audio) > self.sample_rate * 0.5:  # At least 0.5 sec
                    result = await self._transcribe_audio(audio)
                    if result.strip():
                        yield TranscriptEvent(
                            text=result,
                            type=TranscriptType.PARTIAL,
                            timestamp_ms=current_time
                        )
                        last_partial_time = current_time
            
            # Check for end of speech
            if silence_frames >= frames_for_silence and len(buffer.samples) > 0:
                audio = buffer.get_audio()
                result = await self._transcribe_audio(audio)
                yield TranscriptEvent(
                    text=result,
                    type=TranscriptType.FINAL,
                    timestamp_ms=current_time,
                    is_end_of_speech=True
                )
                buffer.clear()
                silence_frames = 0
        
        # Final transcription for remaining audio
        if len(buffer.samples) > 0:
            audio = buffer.get_audio()
            result = await self._transcribe_audio(audio)
            yield TranscriptEvent(
                text=result,
                type=TranscriptType.FINAL,
                timestamp_ms=len(buffer.samples) * 1000 // self.sample_rate,
                is_end_of_speech=True
            )
    
    async def _transcribe_audio(self, audio: np.ndarray) -> str:
        """Transcribe audio buffer using Whisper."""
        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self.model.transcribe(
                audio,
                language="en",
                fp16=False,
                task="transcribe"
            )
        )
        return result["text"].strip()
    
    async def transcribe_file(self, audio_path: str) -> str:
        """Transcribe an audio file (for testing)."""
        self.load_model()
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self.model.transcribe(audio_path, language="en", fp16=False)
        )
        return result["text"].strip()
