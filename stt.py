"""STT module — faster-whisper wrapper."""
import os
import tempfile
from pathlib import Path
from faster_whisper import WhisperModel

_model: WhisperModel | None = None


def get_model() -> WhisperModel:
    global _model
    if _model is None:
        model_size = os.getenv("WHISPER_MODEL", "base")
        device = os.getenv("WHISPER_DEVICE", "cpu")
        compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
        print(f"[STT] Loading whisper model={model_size} device={device} compute={compute_type}")
        _model = WhisperModel(model_size, device=device, compute_type=compute_type)
        print("[STT] Model loaded.")
    return _model


def transcribe_bytes(audio_bytes: bytes, file_ext: str = ".ogg") -> str:
    """Transcribe audio bytes. Returns text string."""
    model = get_model()

    with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name

    try:
        segments, info = model.transcribe(
            tmp_path,
            beam_size=3,
            language="es",  # Spanish default; None = auto-detect
            vad_filter=True,
        )
        text = " ".join(s.text.strip() for s in segments).strip()
        print(f"[STT] Detected lang={info.language} prob={info.language_probability:.2f}")
        return text
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def transcribe_file(path: str | Path) -> str:
    """Transcribe from file path. Returns text string."""
    with open(path, "rb") as f:
        ext = Path(path).suffix or ".ogg"
        return transcribe_bytes(f.read(), ext)
