"""TTS module — Kokoro wrapper."""
import os
import io
import asyncio
from pathlib import Path
import numpy as np
import soundfile as sf
from kokoro import KPipeline

# Default 'e' for Spanish in Kokoro
TTS_LANG = os.getenv("TTS_LANG", "e")
# e_isabella is one of the Spanish voices in Kokoro
TTS_VOICE = os.getenv("TTS_VOICE", "e_isabella")
TTS_SPEED = float(os.getenv("TTS_SPEED", "1.0"))

_pipeline: KPipeline | None = None

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        print(f"[TTS] Loading Kokoro model (lang={TTS_LANG})...")
        _pipeline = KPipeline(lang_code=TTS_LANG)
        print("[TTS] Model loaded.")
    return _pipeline

async def synthesize(text: str, voice: str | None = None, speed: float | None = None) -> bytes:
    """Convert text to WAV audio bytes using Kokoro."""
    v = voice or TTS_VOICE
    s = speed or TTS_SPEED

    def _generate() -> bytes:
        pipeline = get_pipeline()
        
        # Generator yields (graphemes, phonemes, audio)
        # Using split_pattern to handle multi-line/long text without throwing errors
        generator = pipeline(text, voice=v, speed=s, split_pattern=r'\n+')
        
        audio_segments = []
        sample_rate = 24000
        
        for _, _, audio in generator:
            if audio is not None:
                audio_segments.append(audio)
                
        if not audio_segments:
            # Fallback empty audio
            audio_segments.append(np.zeros(24000, dtype=np.float32))
            
        final_audio = np.concatenate(audio_segments)
        
        # Ensure audio data type is suitable for soundfile
        if final_audio.dtype != np.float32:
            final_audio = final_audio.astype(np.float32)
            
        out = io.BytesIO()
        sf.write(out, final_audio, sample_rate, format='WAV', subtype='PCM_16')
        return out.getvalue()
        
    return await asyncio.to_thread(_generate)

async def list_spanish_voices() -> list[str]:
    """List available Spanish voices for Kokoro."""
    return ["e_isabella", "e_carmen", "e_alejandra", "e_mateo", "e_luis"]
