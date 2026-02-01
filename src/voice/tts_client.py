"""Piper TTS client for local speech synthesis."""

import asyncio
import subprocess
import tempfile
from pathlib import Path
from typing import AsyncIterator, Optional, List
import wave
import io

from ..config import get_settings


class PiperTTSClient:
    """
    Local TTS client using Piper for fast speech synthesis.
    
    Piper is optimized for low-latency local inference.
    """
    
    def __init__(
        self, 
        voice: Optional[str] = None,
        model_dir: Optional[str] = None
    ):
        """Initialize Piper TTS client."""
        self.settings = get_settings()
        self.voice = voice or self.settings.piper_voice
        self.model_dir = Path(model_dir) if model_dir else Path("models/piper")
        
        self._piper_path = None
        self._model_path = None
    
    def _find_piper(self) -> Optional[Path]:
        """Find piper executable."""
        # Check common locations
        candidates = [
            Path("piper"),
            Path("piper.exe"),
            Path("./piper/piper"),
            Path("./piper/piper.exe"),
        ]
        
        for candidate in candidates:
            if candidate.exists():
                return candidate
        
        # Try to find in PATH
        try:
            result = subprocess.run(
                ["where" if self.settings.host == "windows" else "which", "piper"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                return Path(result.stdout.strip())
        except Exception:
            pass
        
        return None
    
    async def synthesize(self, text: str) -> bytes:
        """
        Synthesize speech from text.
        
        Args:
            text: Text to synthesize
            
        Returns:
            WAV audio bytes
        """
        try:
            # Try using piper-tts Python package first
            from piper import PiperVoice
            
            voice = PiperVoice.load(str(self.model_dir / f"{self.voice}.onnx"))
            
            # Synthesize to in-memory buffer
            audio_buffer = io.BytesIO()
            
            with wave.open(audio_buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(22050)
                
                for audio_bytes in voice.synthesize_stream_raw(text):
                    wav_file.writeframes(audio_bytes)
            
            return audio_buffer.getvalue()
            
        except ImportError:
            # Fall back to subprocess if piper package not installed
            return await self._synthesize_subprocess(text)
    
    async def _synthesize_subprocess(self, text: str) -> bytes:
        """Synthesize using piper subprocess."""
        piper_path = self._find_piper()
        if not piper_path:
            raise RuntimeError("Piper executable not found")
        
        # Create temp file for output
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            output_path = tmp.name
        
        try:
            # Run piper
            process = await asyncio.create_subprocess_exec(
                str(piper_path),
                '--model', str(self.model_dir / f"{self.voice}.onnx"),
                '--output_file', output_path,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            await process.communicate(input=text.encode('utf-8'))
            
            # Read output
            with open(output_path, 'rb') as f:
                return f.read()
                
        finally:
            # Cleanup
            Path(output_path).unlink(missing_ok=True)
    
    async def synthesize_streaming(
        self, 
        text_chunks: List[str]
    ) -> AsyncIterator[bytes]:
        """
        Synthesize speech from text chunks, yielding audio as it's ready.
        
        For streaming output where we don't want to wait for full synthesis.
        """
        for chunk in text_chunks:
            if chunk.strip():
                audio = await self.synthesize(chunk)
                yield audio
    
    async def synthesize_with_timing(
        self, 
        text: str
    ) -> tuple[bytes, float]:
        """
        Synthesize speech and return with timing info.
        
        Returns:
            Tuple of (audio_bytes, duration_seconds)
        """
        import time
        
        start = time.time()
        audio = await self.synthesize(text)
        synthesis_time = time.time() - start
        
        # Calculate audio duration from WAV
        try:
            with io.BytesIO(audio) as audio_buffer:
                with wave.open(audio_buffer, 'rb') as wav_file:
                    frames = wav_file.getnframes()
                    rate = wav_file.getframerate()
                    duration = frames / rate
        except Exception:
            duration = len(text.split()) / 2.5  # Estimate from word count
        
        return audio, duration
    
    def is_available(self) -> bool:
        """Check if Piper TTS is available."""
        try:
            from piper import PiperVoice
            return True
        except ImportError:
            return self._find_piper() is not None
