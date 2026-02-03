"""pyttsx3 TTS client for local speech synthesis."""

import asyncio
import io
import wave
from pathlib import Path
from typing import AsyncIterator, Optional

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None

from ..config import get_settings


class Pyttsx3TTSClient:
    """
    Local TTS client using pyttsx3 (Windows SAPI voices).
    
    Advantages:
    - Pure Python, no compilation needed
    - Uses native Windows voices
    - Works offline
    - Fast synthesis
    - No external API calls
    """
    
    def __init__(
        self, 
        voice: Optional[str] = None
    ):
        """Initialize pyttsx3 TTS client."""
        self.settings = get_settings()
        self.engine = None
        
        if pyttsx3 is None:
            print("pyttsx3 not installed. Install with: uv pip install pyttsx3")
            return
        
        try:
            self.engine = pyttsx3.init()
            
            # Configure voice properties
            self.engine.setProperty('rate', 175)  # Speed (default is 200)
            self.engine.setProperty('volume', 0.9)  # Volume (0.0 to 1.0)
            
            # Try to use a good quality voice
            voices = self.engine.getProperty('voices')
            if voices:
                # Prefer Microsoft David or Zira (natural voices)
                preferred = None
                for v in voices:
                    if 'David' in v.name or 'Zira' in v.name or 'Mark' in v.name:
                        preferred = v
                        break
                
                if preferred:
                    self.engine.setProperty('voice', preferred.id)
                    print(f"TTS voice: {preferred.name}")
                else:
                    print(f"TTS voice: {voices[0].name}")
                    
        except Exception as e:
            print(f"Failed to initialize pyttsx3: {e}")
            self.engine = None

    async def synthesize(self, text: str) -> bytes:
        """Synthesize speech from text using pyttsx3."""
        if not self.engine or not text.strip():
            return b""

        try:
            # Save to WAV file temporarily
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                output_path = tmp.name
            
            # Run synthesis in thread pool to avoid blocking
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._sync_synthesize, text, output_path)
            
            # Read the WAV file
            if os.path.exists(output_path):
                with open(output_path, 'rb') as f:
                    audio_data = f.read()
                
                # Cleanup
                try:
                    os.unlink(output_path)
                except:
                    pass
                
                return audio_data
            
            return b""
                
        except Exception as e:
            print(f"pyttsx3 synthesis failed: {e}")
            return b""
    
    def _sync_synthesize(self, text: str, output_path: str):
        """Synchronous synthesis helper."""
        self.engine.save_to_file(text, output_path)
        self.engine.runAndWait()

    async def synthesize_streaming(self, text_iterator: AsyncIterator[str]) -> AsyncIterator[bytes]:
        """
        Stream audio bytes from a text stream.
        """
        if not self.engine:
            return

        # Synthesize sentence-by-sentence
        async for text_chunk in text_iterator:
            if not text_chunk.strip():
                continue
            
            audio_bytes = await self.synthesize(text_chunk)
            if audio_bytes:
                yield audio_bytes

    def is_available(self) -> bool:
        return self.engine is not None


# Keep EdgeTTSClient as fallback
class EdgeTTSClient:
    """Edge TTS client (fallback)."""
    
    def __init__(self, voice: Optional[str] = None):
        try:
            import edge_tts
            self.voice_name = voice or "en-US-AriaNeural"
            self.available = True
        except ImportError:
            self.available = False
    
    async def synthesize(self, text: str) -> bytes:
        if not self.available or not text.strip():
            return b""
        
        try:
            import edge_tts
            import io
            
            communicate = edge_tts.Communicate(text, self.voice_name)
            audio_buffer = io.BytesIO()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_buffer.write(chunk["data"])
            
            return audio_buffer.getvalue()
        except Exception as e:
            print(f"Edge TTS error: {e}")
            return b""
    
    def is_available(self) -> bool:
        return self.available
