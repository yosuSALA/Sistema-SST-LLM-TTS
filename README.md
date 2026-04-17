# Sistema SST-LLM-TTS

Pipeline completo de voz artificial (Voz a Texto -> LLM -> Texto a Voz) ahora optimizado para servidores con tarjetas de video GTX 1060 de 6GB, usando localmente Ollama, Faster-Whisper, y el sintetizador **Kokoro**.

## Solución de Errores Comunes (GTX 1060)

El problema más recurrente al ejecutar este tipo de modelos en una GTX 1060 (Arquitectura Pascal, 6GB VRAM) es el desbordamiento de memoria de video (OOM - Out of Memory) o incompatibilidad con cálculos nativos en `float16`. 

Para evitar bloqueos:

**1. Usar `int8` para Whisper:**  
La tarjeta GTX 1060 procesa muy bien en `int8`, mitigando el uso excesivo de VRAM mientras corre en la GPU haciéndolo mucho más rápido que en un CPU. Hemos configurado `.env.example` y el código para que asigne automáticamente `cuda` e `int8`. 

**2. Gestionar la VRAM de Ollama y el Modelo LLM:**  
El modelo por defecto en `.env.example` puede competir por la memoria de la tarjeta contra Kokoro y Whisper. Si todavía tienes cuelgues por falta de VRAM, considera probar un modelo más pequeño en Ollama (ej: `qwen2:1.5b` o `phi3`).

**3. Cambio a Kokoro TTS:**  
Hemos reemplazado la integración online de `edge-tts` por **Kokoro TTS**, que se ejecuta de forma 100% nativa. El sistema descargará el modelo de Kokoro durante el primer intento de generar voz.

## Instrucciones de Instalación para el Servidor Local

### 1. Actualizar Dependencias

Las librerías de Kokoro (`kokoro` y `soundfile`) han sido agregadas y actualizadas. Borra o actualiza tu entorno e instala:

```bash
pip install -r requirements.txt
```

### 2. Configurar Entorno (.env)

Debes crear tu archivo `.env` basándote en el archivo actualizado `.env.example`. Modifica el `.env` para asegurar que el uso de VRAM y la voz de Kokoro sean correctas.

```bash
copy .env.example .env
```

Asegúrate de que contenga esta parte crucial:
```ini
WHISPER_DEVICE=cuda
WHISPER_COMPUTE_TYPE=int8
TTS_VOICE=e_isabella
TTS_SPEED=1.0
```
*Nota: La voz por defecto para español en Kokoro es `e_isabella` (puedes optar por `e_carmen`, `e_alejandra`, `e_mateo`, o `e_luis`).*

### 3. Ejecutar los Servicios

Para ejecutar la **Interfaz Web** (modo prueba):
```bash
python server.py
# Ingresa en http://localhost:8000
```
Al correrlo por primera vez verás en consola que Kokoro y Whisper descargarán sus archivos de modelos. Esto tomará algo de tiempo. 

Para ejecutar el **Bot de Discord** (si requiere testeo final):
```bash
python main.py
```

## Cambios Realizados

- Se ajustaron scripts de servidor (`server.py`) y del bot (`main.py`) para consumir y servir archivos `.wav` de manera directa gracias a la eficiencia de Kokoro.
- Código reescrito de `tts.py` para instanciar bajo demanda y retener el modelo (KPipeline de Kokoro en memoria).
- Librerías ajustadas en el `requirements.txt`.
