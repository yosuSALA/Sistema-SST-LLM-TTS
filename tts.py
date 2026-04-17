"""TTS module — edge-tts wrapper."""
import os
import tempfile
from pathlib import Path
import edge_tts

TTS_VOICE = os.getenv("TTS_VOICE", "es-MX-DaliaNeural")
TTS_RATE = os.getenv("TTS_RATE", "+0%")


async def synthesize(text: str, voice: str | None = None, rate: str | None = None) -> bytes:
    """Convert text to MP3 audio bytes."""
    v = voice or TTS_VOICE
    r = rate or TTS_RATE

    communicate = edge_tts.Communicate(text, v, rate=r)

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        tmp_path = f.name

    try:
        await communicate.save(tmp_path)
        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        Path(tmp_path).unlink(missing_ok=True)


async def list_spanish_voices() -> list[str]:
    """List available Spanish voices from edge-tts."""
    voices = await edge_tts.list_voices()
    return [v["ShortName"] for v in voices if v["Locale"].startswith("es")]
