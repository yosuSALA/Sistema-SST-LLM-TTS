"""Discord bot — STT → LLM → TTS pipeline."""
import asyncio
import io
import os
import tempfile
from pathlib import Path

import discord
from discord.ext import commands
from dotenv import load_dotenv

load_dotenv()

import llm
import stt
import tts

# ── Bot setup ────────────────────────────────────────────────────────────────

intents = discord.Intents.default()
intents.message_content = True
intents.messages = True

bot = commands.Bot(command_prefix="!", intents=intents)

# ── Events ───────────────────────────────────────────────────────────────────

@bot.event
async def on_ready():
    print(f"[BOT] Logged in as {bot.user} (ID: {bot.user.id})")
    print(f"[BOT] Ollama model: {os.getenv('OLLAMA_MODEL', 'gemma-4b-pro:latest')}")
    print(f"[BOT] TTS voice: {os.getenv('TTS_VOICE', 'es-MX-DaliaNeural')}")
    print("[BOT] Ready. Send a voice message or use !ask / !habla")


@bot.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return

    # ── Voice message handling (automatic) ───────────────────────────────
    voice_attachment = _get_voice_attachment(message)
    if voice_attachment:
        await _handle_voice_message(message, voice_attachment)
        return

    await bot.process_commands(message)


def _get_voice_attachment(message: discord.Message) -> discord.Attachment | None:
    """Return first audio attachment if message is a voice note."""
    for att in message.attachments:
        if att.content_type and att.content_type.startswith("audio/"):
            return att
        # Discord voice messages have waveform data
        if hasattr(att, "waveform") and att.waveform:
            return att
        # Fallback: ogg/mp3/wav files
        if att.filename.lower().endswith((".ogg", ".mp3", ".wav", ".m4a", ".webm")):
            return att
    return None


async def _handle_voice_message(message: discord.Message, attachment: discord.Attachment):
    """Full pipeline: audio → STT → LLM → TTS → Discord."""
    async with message.channel.typing():
        try:
            # 1. Download audio
            audio_bytes = await attachment.read()
            ext = Path(attachment.filename).suffix or ".ogg"

            # 2. STT
            await message.add_reaction("👂")
            transcript = await asyncio.to_thread(stt.transcribe_bytes, audio_bytes, ext)

            if not transcript:
                await message.reply("No pude entender el audio. ¿Puedes repetirlo?")
                return

            print(f"[STT] '{transcript}'")
            await message.remove_reaction("👂", bot.user)

            # 3. LLM
            await message.add_reaction("🤔")
            response_text = await llm.simple_ask(transcript)
            print(f"[LLM] '{response_text}'")
            await message.remove_reaction("🤔", bot.user)

            # 4. TTS
            await message.add_reaction("🔊")
            audio_response = await tts.synthesize(response_text)
            await message.remove_reaction("🔊", bot.user)

            # 5. Reply with transcript + audio
            await message.reply(
                f"**Escuché:** {transcript}\n**Respuesta:** {response_text}",
                file=discord.File(
                    io.BytesIO(audio_response),
                    filename="respuesta.mp3",
                    description="Respuesta de voz",
                ),
            )

        except Exception as e:
            print(f"[ERR] Pipeline error: {e}")
            await message.reply(f"Error en el pipeline: `{e}`")


# ── Commands ─────────────────────────────────────────────────────────────────

@bot.command(name="ask")
async def cmd_ask(ctx: commands.Context, *, text: str):
    """Text query → LLM → text response. Usage: !ask <pregunta>"""
    async with ctx.typing():
        try:
            response = await llm.simple_ask(text)
            await ctx.reply(response)
        except Exception as e:
            await ctx.reply(f"Error LLM: `{e}`")


@bot.command(name="habla")
async def cmd_habla(ctx: commands.Context, *, text: str):
    """Text → TTS audio. Usage: !habla <texto>"""
    async with ctx.typing():
        try:
            audio_bytes = await tts.synthesize(text)
            await ctx.reply(
                file=discord.File(
                    io.BytesIO(audio_bytes),
                    filename="audio.mp3",
                )
            )
        except Exception as e:
            await ctx.reply(f"Error TTS: `{e}`")


@bot.command(name="pregunta")
async def cmd_pregunta(ctx: commands.Context, *, text: str):
    """Full pipeline: text → LLM → TTS audio. Usage: !pregunta <texto>"""
    async with ctx.typing():
        try:
            response_text = await llm.simple_ask(text)
            audio_bytes = await tts.synthesize(response_text)
            await ctx.reply(
                f"**Respuesta:** {response_text}",
                file=discord.File(
                    io.BytesIO(audio_bytes),
                    filename="respuesta.mp3",
                ),
            )
        except Exception as e:
            await ctx.reply(f"Error: `{e}`")


@bot.command(name="voces")
async def cmd_voces(ctx: commands.Context):
    """List available Spanish TTS voices."""
    async with ctx.typing():
        voices = await tts.list_spanish_voices()
        voice_list = "\n".join(voices)
        await ctx.reply(f"**Voces disponibles en español:**\n```\n{voice_list}\n```")


@bot.command(name="ping")
async def cmd_ping(ctx: commands.Context):
    """Health check."""
    latency = round(bot.latency * 1000)
    await ctx.reply(f"Pong! Latencia: {latency}ms | Modelo: {os.getenv('OLLAMA_MODEL', 'N/A')}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    token = os.getenv("DISCORD_TOKEN")
    if not token:
        raise ValueError("DISCORD_TOKEN not set. Copy .env.example → .env and add your token.")
    bot.run(token)
