"""Kokoro TTS client for high-quality voice synthesis."""

import io
import asyncio
from typing import AsyncIterator
import numpy as np

import soundfile as sf
from kokoro import KPipeline


class KokoroTTS:
    """
    Kokoro TTS client - 82M parameter open-weight TTS model.
    
    Features:
    - High-quality natural voice
    - Fast inference
    - Apache licensed
    - Multiple voice options
    """
    
    def __init__(self, voice: str = "af_bella", lang_code: str = "a"):
        """
        Initialize Kokoro TTS.
        
        Args:
            voice: Voice to use (af_heart, af_sky, am_adam, am_michael, etc.)
            lang_code: Language code ('a' for general)
        """
        self.voice = voice
        self.lang_code = lang_code
        self._pipeline = None
    
    def load_model(self):
        """Load Kokoro pipeline."""
        if self._pipeline is None:
            print(f"Loading Kokoro TTS ({self.voice})...")
            self._pipeline = KPipeline(lang_code=self.lang_code)
            print("✅ Kokoro TTS loaded")
    
    async def synthesize(self, text: str) -> bytes:
        """
        Synthesize text to speech (non-streaming).
        
        Args:
            text: Text to synthesize
            
        Returns:
            Complete audio as bytes (WAV format)
        """
        self.load_model()
        
        # Run in executor
        loop = asyncio.get_event_loop()
        
        def _synthesize():
            """Generate complete audio."""
            generator = self._pipeline(text, voice=self.voice)
            
            # Collect all audio chunks
            audio_chunks = []
            for _, _, audio in generator:
                audio_chunks.append(audio)
            
            # Concatenate
            if not audio_chunks:
                return b""
                
            full_audio = np.concatenate(audio_chunks)
            
            # Convert to WAV bytes
            buffer = io.BytesIO()
            sf.write(buffer, full_audio, 24000, format='WAV')
            return buffer.getvalue()
        
        audio_bytes = await loop.run_in_executor(None, _synthesize)
        return audio_bytes


# Alias for compatibility
TTSClient = KokoroTTS
