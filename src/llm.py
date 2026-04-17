"""LLM module — Ollama client."""
import os
import httpx

SYSTEM_PROMPT = """Eres un asistente de voz inteligente. Respondes de forma concisa y natural.
Tus respuestas serán convertidas a audio, así que:
- Evita usar markdown, asteriscos, o símbolos especiales
- Sé breve: máximo 3 oraciones por respuesta
- Habla en español
"""

OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma-4b-pro:latest")


async def chat(messages: list[dict], *, stream: bool = False) -> str:
    """Send messages to Ollama. Returns response text."""
    full_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages

    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(
            f"{OLLAMA_BASE}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": full_messages,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 256,
                },
            },
        )
        r.raise_for_status()
        data = r.json()
        return data["message"]["content"].strip()


async def simple_ask(text: str) -> str:
    """Single turn Q&A. Convenience wrapper."""
    return await chat([{"role": "user", "content": text}])
