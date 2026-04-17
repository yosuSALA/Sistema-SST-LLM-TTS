"""Local web interface — STT → LLM → TTS pipeline."""
import asyncio
import io
import os
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from dotenv import load_dotenv

load_dotenv()

from src import llm, stt, tts

app = FastAPI(title="AI Lab Local", docs_url=None, redoc_url=None)

# ── API endpoints ─────────────────────────────────────────────────────────────

@app.post("/api/ask")
async def api_ask(text: str = Form(...)):
    """Text → LLM → text response."""
    response = await llm.simple_ask(text)
    return {"text": response}


@app.post("/api/speak")
async def api_speak(text: str = Form(...)):
    """Text → TTS → WAV audio bytes."""
    audio = await tts.synthesize(text)
    return Response(content=audio, media_type="audio/wav")


@app.post("/api/chat")
async def api_chat(text: str = Form(...)):
    """Text → LLM → TTS → WAV (full pipeline, no STT)."""
    response_text = await llm.simple_ask(text)
    audio = await tts.synthesize(response_text)
    return Response(
        content=audio,
        media_type="audio/wav",
        headers={"X-Response-Text": response_text.encode("utf-8").decode("latin-1")},
    )


@app.post("/api/voice")
async def api_voice(audio: UploadFile = File(...)):
    """Audio file → STT → LLM → TTS → WAV (full pipeline)."""
    audio_bytes = await audio.read()
    ext = Path(audio.filename).suffix if audio.filename else ".ogg"

    # STT
    transcript = await asyncio.to_thread(stt.transcribe_bytes, audio_bytes, ext)
    if not transcript:
        raise HTTPException(400, "Could not transcribe audio")

    # LLM
    response_text = await llm.simple_ask(transcript)

    # TTS
    audio_out = await tts.synthesize(response_text)

    return Response(
        content=audio_out,
        media_type="audio/wav",
        headers={
            "X-Transcript": transcript.encode("utf-8").decode("latin-1"),
            "X-Response-Text": response_text.encode("utf-8").decode("latin-1"),
        },
    )


@app.get("/api/health")
async def api_health():
    return {
        "status": "ok",
        "model": os.getenv("OLLAMA_MODEL", "gemma-4b-pro:latest"),
        "tts_engine": "piper",
        "tts_model": os.path.basename(os.getenv("TTS_MODEL_PATH", "es_MX-claude-high")),
        "whisper_model": os.getenv("WHISPER_MODEL", "base"),
    }


# ── Web UI ────────────────────────────────────────────────────────────────────

HTML = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI Lab Local</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: system-ui, sans-serif;
    background: #0f0f0f;
    color: #e0e0e0;
    height: 100vh;
    display: flex;
    flex-direction: column;
  }
  header {
    padding: 12px 20px;
    background: #1a1a2e;
    border-bottom: 1px solid #333;
    display: flex;
    align-items: center;
    gap: 10px;
  }
  header h1 { font-size: 1rem; color: #7b9fff; }
  #status { font-size: 0.75rem; color: #666; margin-left: auto; }
  #chat {
    flex: 1;
    overflow-y: auto;
    padding: 16px 20px;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }
  .msg {
    max-width: 75%;
    padding: 10px 14px;
    border-radius: 12px;
    font-size: 0.9rem;
    line-height: 1.5;
  }
  .msg.user {
    background: #1e3a5f;
    align-self: flex-end;
    border-bottom-right-radius: 4px;
  }
  .msg.bot {
    background: #1e1e2e;
    align-self: flex-start;
    border-bottom-left-radius: 4px;
    border: 1px solid #333;
  }
  .msg .label {
    font-size: 0.7rem;
    color: #666;
    margin-bottom: 4px;
  }
  .msg .transcript {
    font-size: 0.75rem;
    color: #888;
    font-style: italic;
    margin-bottom: 6px;
  }
  .msg audio { margin-top: 8px; width: 100%; }
  .thinking {
    align-self: flex-start;
    color: #555;
    font-size: 0.85rem;
    padding: 6px 14px;
    animation: pulse 1.2s ease-in-out infinite;
  }
  @keyframes pulse { 0%,100%{opacity:0.4} 50%{opacity:1} }

  footer {
    padding: 12px 16px;
    background: #111;
    border-top: 1px solid #222;
    display: flex;
    gap: 8px;
    align-items: flex-end;
  }
  #input-text {
    flex: 1;
    background: #1e1e1e;
    border: 1px solid #333;
    border-radius: 8px;
    color: #e0e0e0;
    padding: 10px 14px;
    font-size: 0.9rem;
    resize: none;
    min-height: 42px;
    max-height: 120px;
    outline: none;
  }
  #input-text:focus { border-color: #7b9fff; }
  button {
    background: #2a4a8a;
    color: #fff;
    border: none;
    border-radius: 8px;
    padding: 10px 16px;
    cursor: pointer;
    font-size: 0.85rem;
    transition: background 0.2s;
    white-space: nowrap;
  }
  button:hover { background: #3a5aaa; }
  button:disabled { background: #333; color: #555; cursor: not-allowed; }
  button#rec-btn { background: #5a1a1a; }
  button#rec-btn.recording { background: #aa2222; animation: pulse 0.8s infinite; }
  button#rec-btn:hover:not(.recording) { background: #7a2a2a; }
  .audio-player {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-top: 10px;
    background: #2a2a3e;
    border-radius: 8px;
    padding: 8px 12px;
    border: 1px solid #444;
  }
  .play-btn {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    background: #4a7aff;
    border: none;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 14px;
    flex-shrink: 0;
    transition: background 0.2s, transform 0.1s;
    padding: 0;
    color: white;
  }
  .play-btn:hover { background: #6a9aff; transform: scale(1.05); }
  .play-btn:disabled { background: #333; cursor: not-allowed; }
  .audio-progress {
    flex: 1;
    height: 4px;
    background: #444;
    border-radius: 2px;
    overflow: hidden;
    cursor: pointer;
  }
  .audio-progress-bar {
    height: 100%;
    background: #4a7aff;
    width: 0%;
    transition: width 0.1s linear;
    border-radius: 2px;
  }
  .audio-time {
    font-size: 0.72rem;
    color: #888;
    white-space: nowrap;
    min-width: 35px;
    text-align: right;
  }
</style>
</head>
<body>
<header>
  <span>🤖</span>
  <h1>AI Lab Local</h1>
  <span id="status">cargando...</span>
</header>
<div id="chat"></div>
<footer>
  <textarea id="input-text" placeholder="Escribe un mensaje... (Enter = enviar)" rows="1"></textarea>
  <button id="send-btn" onclick="sendText()">Enviar</button>
  <button id="rec-btn" onclick="toggleRecord()">🎤 Grabar</button>
</footer>

<script>
const chat = document.getElementById('chat');
const input = document.getElementById('input-text');
const sendBtn = document.getElementById('send-btn');
const recBtn = document.getElementById('rec-btn');
let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;

// ── Unlock audio context on first interaction ────────────────────────────────
let audioUnlocked = false;
function unlockAudio() {
  if (audioUnlocked) return;
  const ctx = new (window.AudioContext || window.webkitAudioContext)();
  const buf = ctx.createBuffer(1, 1, 22050);
  const src = ctx.createBufferSource();
  src.buffer = buf;
  src.connect(ctx.destination);
  src.start(0);
  audioUnlocked = true;
  document.removeEventListener('click', unlockAudio);
  document.removeEventListener('keydown', unlockAudio);
}
document.addEventListener('click', unlockAudio);
document.addEventListener('keydown', unlockAudio);

// ── Health check ────────────────────────────────────────────────────────────
fetch('/api/health').then(r => r.json()).then(d => {
  document.getElementById('status').textContent = `${d.model} · ${d.tts_engine}:${d.tts_model}`;
}).catch(() => {
  document.getElementById('status').textContent = 'error connecting';
});

// ── Keyboard shortcut ────────────────────────────────────────────────────────
input.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendText();
  }
});

// ── Text pipeline ────────────────────────────────────────────────────────────
async function sendText() {
  const text = input.value.trim();
  if (!text) return;

  addMessage('user', text);
  input.value = '';
  setLoading(true);

  const thinking = addThinking();

  try {
    const fd = new FormData();
    fd.append('text', text);
    const res = await fetch('/api/chat', { method: 'POST', body: fd });
    if (!res.ok) throw new Error(await res.text());

    const responseText = decodeURIComponent(escape(
      res.headers.get('X-Response-Text') || '...'
    ));
    const rawBlob = await res.blob();
    const blob = new Blob([rawBlob], { type: 'audio/wav' });
    thinking.remove();
    addBotMessage(responseText, blob);
  } catch (err) {
    thinking.remove();
    addMessage('bot', `Error: ${err.message}`);
  } finally {
    setLoading(false);
  }
}

// ── Voice recording ──────────────────────────────────────────────────────────
async function toggleRecord() {
  if (isRecording) {
    mediaRecorder.stop();
    return;
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    audioChunks = [];
    mediaRecorder = new MediaRecorder(stream);
    mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
    mediaRecorder.onstop = async () => {
      stream.getTracks().forEach(t => t.stop());
      isRecording = false;
      recBtn.textContent = '🎤 Grabar';
      recBtn.classList.remove('recording');
      await sendVoice();
    };
    mediaRecorder.start();
    isRecording = true;
    recBtn.textContent = '⏹ Detener';
    recBtn.classList.add('recording');
  } catch (err) {
    alert('No se pudo acceder al micrófono: ' + err.message);
  }
}

async function sendVoice() {
  if (!audioChunks.length) return;
  const blob = new Blob(audioChunks, { type: 'audio/webm' });

  addMessage('user', '[Nota de voz]');
  setLoading(true);
  const thinking = addThinking();

  try {
    const fd = new FormData();
    fd.append('audio', blob, 'voice.webm');
    const res = await fetch('/api/voice', { method: 'POST', body: fd });
    if (!res.ok) throw new Error(await res.text());

    const transcript = res.headers.get('X-Transcript') || '';
    const responseText = decodeURIComponent(escape(
      res.headers.get('X-Response-Text') || '...'
    ));
    const rawBlob = await res.blob();
    const audioBlob = new Blob([rawBlob], { type: 'audio/wav' });
    thinking.remove();

    // Update the [Nota de voz] message with transcript
    const msgs = chat.querySelectorAll('.msg.user');
    const last = msgs[msgs.length - 1];
    if (last && transcript) {
      last.querySelector('.content').textContent = `🎤 "${transcript}"`;
    }
    addBotMessage(responseText, audioBlob);
  } catch (err) {
    thinking.remove();
    addMessage('bot', `Error: ${err.message}`);
  } finally {
    setLoading(false);
  }
}

// ── DOM helpers ──────────────────────────────────────────────────────────────
function addMessage(role, text) {
  const div = document.createElement('div');
  div.className = `msg ${role}`;
  div.innerHTML = `<div class="label">${role === 'user' ? 'Tú' : 'IA'}</div><div class="content">${escHtml(text)}</div>`;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
  return div;
}

function addBotMessage(text, audioBlob) {
  const div = document.createElement('div');
  div.className = 'msg bot';
  const audioUrl = URL.createObjectURL(audioBlob);

  // Build DOM manually to attach events
  const label = document.createElement('div');
  label.className = 'label';
  label.textContent = 'IA';

  const content = document.createElement('div');
  content.className = 'content';
  content.textContent = text;

  // Custom audio player
  const audio = new Audio();
  audio.preload = 'auto';
  audio.src = audioUrl;
  const player = document.createElement('div');
  player.className = 'audio-player';

  const playBtn = document.createElement('button');
  playBtn.className = 'play-btn';
  playBtn.innerHTML = '▶';
  playBtn.title = 'Reproducir';

  const progressWrap = document.createElement('div');
  progressWrap.className = 'audio-progress';
  const progressBar = document.createElement('div');
  progressBar.className = 'audio-progress-bar';
  progressWrap.appendChild(progressBar);

  const timeLabel = document.createElement('span');
  timeLabel.className = 'audio-time';
  timeLabel.textContent = '0:00';

  player.appendChild(playBtn);
  player.appendChild(progressWrap);
  player.appendChild(timeLabel);

  // Events
  playBtn.onclick = () => {
    if (audio.paused) { audio.play(); playBtn.innerHTML = '⏸'; }
    else { audio.pause(); playBtn.innerHTML = '▶'; }
  };
  audio.ontimeupdate = () => {
    if (audio.duration) {
      progressBar.style.width = (audio.currentTime / audio.duration * 100) + '%';
      const s = Math.floor(audio.currentTime);
      timeLabel.textContent = `${Math.floor(s/60)}:${String(s%60).padStart(2,'0')}`;
    }
  };
  audio.onended = () => { playBtn.innerHTML = '▶'; progressBar.style.width = '0%'; };
  audio.onerror = (e) => { console.error('Audio error:', e); playBtn.innerHTML = '❌'; playBtn.title = 'Error de audio'; };
  audio.oncanplay = () => { playBtn.disabled = false; };
  audio.onloadedmetadata = () => {
    const dur = Math.floor(audio.duration);
    timeLabel.textContent = `0:00 / ${Math.floor(dur/60)}:${String(dur%60).padStart(2,'0')}`;
  };
  progressWrap.onclick = (e) => {
    const rect = progressWrap.getBoundingClientRect();
    audio.currentTime = ((e.clientX - rect.left) / rect.width) * audio.duration;
  };

  div.appendChild(label);
  div.appendChild(content);
  div.appendChild(player);
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;

  // Autoplay — if blocked by browser policy, pulse the play button
  audio.play().then(() => {
    playBtn.innerHTML = '⏸';
  }).catch(() => {
    playBtn.innerHTML = '▶';
    playBtn.style.animation = 'pulse 1s ease-in-out 3';
    playBtn.style.background = '#22aa44';
    setTimeout(() => { playBtn.style.animation = ''; playBtn.style.background = ''; }, 3000);
  });
}

function addThinking() {
  const div = document.createElement('div');
  div.className = 'thinking';
  div.textContent = 'Pensando...';
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
  return div;
}

function escHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function setLoading(v) {
  sendBtn.disabled = v;
  recBtn.disabled = v && !isRecording;
}
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def root():
    return HTML


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    print(f"[SERVER] Starting at http://{host}:{port}")
    uvicorn.run("server:app", host=host, port=port, reload=False)
