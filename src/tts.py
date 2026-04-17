"""TTS module — Piper TTS wrapper (local, Spanish, CPU-fast)."""
import asyncio
import io
import os
import wave
from pathlib import Path

from piper.voice import PiperVoice

# Model path — defaults to bundled es_MX voice
_DEFAULT_MODEL = os.getenv(
    "TTS_MODEL_PATH",
    str(Path(__file__).parent.parent / "models/piper/es/es_MX/claude/high/es_MX-claude-high.onnx"),
)
TTS_SPEED = float(os.getenv("TTS_SPEED", "1.0"))

_voice: PiperVoice | None = None


def get_voice() -> PiperVoice:
    global _voice
    if _voice is None:
        model_path = _DEFAULT_MODEL
        print(f"[TTS] Loading Piper voice from {model_path}")
        _voice = PiperVoice.load(model_path)
        print(f"[TTS] Loaded. Sample rate: {_voice.config.sample_rate}Hz")
    return _voice


def _synthesize_sync(text: str) -> bytes:
    """Blocking synthesis — run in thread via asyncio.to_thread."""
    voice = get_voice()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_out:
        wav_out.setnchannels(1)
        wav_out.setsampwidth(2)  # 16-bit PCM
        first = True
        for chunk in voice.synthesize(text):
            if first:
                wav_out.setframerate(chunk.sample_rate)
                first = False
            wav_out.writeframes(chunk.audio_int16_bytes)
    return buf.getvalue()


async def synthesize(text: str, **kwargs) -> bytes:
    """Convert text to WAV audio bytes using Piper TTS."""
    return await asyncio.to_thread(_synthesize_sync, text)


async def list_spanish_voices() -> list[str]:
    """List available TTS voices (static — Piper uses local model files)."""
    models_dir = Path(__file__).parent.parent / "models/piper"
    voices = []
    if models_dir.exists():
        for f in models_dir.rglob("*.onnx"):
            if not f.name.endswith(".json"):
                voices.append(f.stem)
    return voices if voices else ["es_MX-claude-high"]
