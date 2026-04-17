"""TTS module — Kokoro (hexgrad/Kokoro-82M) wrapper."""
import asyncio
import io
import os

import numpy as np
import soundfile as sf
from kokoro import KPipeline

# GTX 1060 = SM 6.1, incompatible with PyTorch >= 2.0 CUDA → force CPU
TTS_DEVICE = os.getenv("TTS_DEVICE", "cpu")
TTS_LANG = os.getenv("TTS_LANG", "e")          # 'e' = Spanish
TTS_VOICE = os.getenv("TTS_VOICE", "ef_dora")  # ef_dora, em_alex, em_santa
TTS_SPEED = float(os.getenv("TTS_SPEED", "1.0"))
SAMPLE_RATE = 24000

_pipeline: KPipeline | None = None


def get_pipeline() -> KPipeline:
    global _pipeline
    if _pipeline is None:
        print(f"[TTS] Loading Kokoro (lang={TTS_LANG}, device={TTS_DEVICE})...")
        _pipeline = KPipeline(
            lang_code=TTS_LANG,
            repo_id="hexgrad/Kokoro-82M",
            device=TTS_DEVICE,
        )
        print("[TTS] Kokoro loaded.")
    return _pipeline


def _synthesize_sync(text: str, voice: str, speed: float) -> bytes:
    pipeline = get_pipeline()
    segments = []
    for _, _, audio in pipeline(text, voice=voice, speed=speed, split_pattern=r"\n+"):
        if audio is not None:
            segments.append(audio)

    if not segments:
        segments.append(np.zeros(SAMPLE_RATE, dtype=np.float32))

    final = np.concatenate(segments)
    if final.dtype != np.float32:
        final = final.astype(np.float32)

    out = io.BytesIO()
    sf.write(out, final, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return out.getvalue()


async def synthesize(text: str, voice: str | None = None, speed: float | None = None) -> bytes:
    """Convert text to WAV audio bytes using Kokoro."""
    v = voice or TTS_VOICE
    s = speed or TTS_SPEED
    return await asyncio.to_thread(_synthesize_sync, text, v, s)


async def list_spanish_voices() -> list[str]:
    """Available Spanish voices in Kokoro-82M."""
    return ["ef_dora", "em_alex", "em_santa"]
