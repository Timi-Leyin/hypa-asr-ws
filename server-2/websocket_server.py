#!/usr/bin/env python3
"""
WebSocket Transcription Server
Uses faster-whisper with a local CTranslate2 model.

Design:
  - Client sends a complete 3-second WAV chunk.
  - Server decodes it, transcribes it, replies with the caption.
  - Backpressure: one transcription at a time per client; extra chunks
    are dropped with a {"type": "busy"} reply so the client can skip.
"""
import asyncio
import io
import json
import logging
import wave
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from time import perf_counter
from typing import Optional

import numpy as np
import websockets

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("transcription_server")

# ── Config ─────────────────────────────────────────────────────────────────────
USE_CONVERTED_MODEL = True          # True → faster-whisper (CT2), False → HuggingFace transformers
CT2_MODEL_PATH      = "./wspr_small_ct2"
HF_MODEL_ID         = "hypaai/wspr_small_2025-11-11_12-12-17"
MODEL_PATH          = CT2_MODEL_PATH if USE_CONVERTED_MODEL else HF_MODEL_ID
HOST                = "0.0.0.0"
PORT                = 8765
SAMPLE_RATE         = 16_000
SILENCE_RMS         = 0.01         # below this → treat as silence
MIN_AUDIO_SECS      = 0.3           # ignore chunks shorter than this
MAX_WORKERS         = 10              # reduced - H200 is fast enough

# ── Model parameters ───────────────────────────────────────────────────────────
TEMPERATURE         = 0.3           # 0.0 = greedy/deterministic (fastest)
BEAM_SIZE           = 1             # 1 = greedy search (fastest), >1 = beam search
REPETITION_PENALTY  = 1.0           # 1.0 = disabled (no overhead)


# ── Audio decoding ─────────────────────────────────────────────────────────────

def decode_wav(raw: bytes, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Decode WAV bytes → float32 mono array at *target_sr*."""
    with io.BytesIO(raw) as buf, wave.open(buf, "rb") as wf:
        ch  = wf.getnchannels()
        sw  = wf.getsampwidth()
        fr  = wf.getframerate()
        nf  = wf.getnframes()
        pcm = wf.readframes(nf)

        scale = 32_768.0 if sw == 2 else 2_147_483_648.0
        dtype = np.int16 if sw == 2 else np.int32
        arr   = np.frombuffer(pcm, dtype=dtype).astype(np.float32) / scale

        if ch == 2:
            arr = arr.reshape(-1, 2).mean(axis=1)
        if fr != target_sr:
            import librosa
            arr = librosa.resample(arr, orig_sr=fr, target_sr=target_sr)
        return arr


# ── Transcription server ───────────────────────────────────────────────────────

class TranscriptionServer:

    def __init__(self):
        self._use_ct2 = USE_CONVERTED_MODEL
        log.info("Loading model from %s …", MODEL_PATH)

        # Detect GPU availability
        import torch
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._torch = torch
        log.info("Using device: %s", self._device)

        if self._use_ct2:
            from faster_whisper import WhisperModel
            self.model = WhisperModel(
                CT2_MODEL_PATH,
                device=self._device,
                compute_type="int8_float16" if self._device == "cuda" else "int8",
                cpu_threads=4,
            )
        else:
            from transformers import WhisperProcessor, WhisperForConditionalGeneration
            self._processor = WhisperProcessor.from_pretrained(HF_MODEL_ID)
            self._hf_model  = WhisperForConditionalGeneration.from_pretrained(HF_MODEL_ID)
            self._hf_model.to(self._device)
            self._hf_model.eval()

        self._executor = ThreadPoolExecutor(
            max_workers=MAX_WORKERS, thread_name_prefix="whisper",
        )
        self._connections: set = set()
        log.info("Model ready.")

    # ── Inference (blocking – runs in thread pool) ──────────────────────────────

    def _transcribe(self, audio: np.ndarray, language: str) -> Optional[str]:
        try:
            if self._use_ct2:
                return self._transcribe_ct2(audio, language)
            else:
                return self._transcribe_hf(audio, language)
        except Exception as e:
            log.error("Transcription error: %s", e)
            return None

    def _transcribe_ct2(self, audio: np.ndarray, language: str) -> Optional[str]:
        segments, _ = self.model.transcribe(
            audio,
            language=language,
            beam_size=BEAM_SIZE,
            temperature=TEMPERATURE,
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
            vad_filter=True,              # Skip silent portions
            vad_parameters=dict(min_silence_duration_ms=250),
        )
        parts = [s.text.strip() for s in segments if s.text.strip()]
        return " ".join(parts) or None

    def _transcribe_hf(self, audio: np.ndarray, language: str) -> Optional[str]:
        inputs = self._processor(
            audio, sampling_rate=SAMPLE_RATE, return_tensors="pt",
        )
        input_features = inputs.input_features.to(self._device)

        gen_kwargs = {
            "num_beams":          BEAM_SIZE,
            "temperature":        TEMPERATURE if TEMPERATURE > 0 else 1.0,
            "do_sample":          TEMPERATURE > 0,
            "repetition_penalty": REPETITION_PENALTY,
            "language":           language,
            "task":               "transcribe",
        }

        with self._torch.no_grad():
            predicted_ids = self._hf_model.generate(
                input_features, **gen_kwargs,
            )

        text = self._processor.batch_decode(
            predicted_ids, skip_special_tokens=True,
        )
        result = " ".join(t.strip() for t in text if t.strip())
        return result or None

    # ── Per-client handler ─────────────────────────────────────────────────────

    async def handle_client(self, websocket) -> None:
        cid  = id(websocket)
        loop = asyncio.get_running_loop()
        self._connections.add(websocket)
        log.info("Client connected  [%d]  (total: %d)", cid, len(self._connections))

        is_busy      = False
        config       = {"language": "en"}
        current_task = None

        async def send(payload: dict) -> None:
            try:
                await websocket.send(json.dumps(payload))
            except websockets.exceptions.ConnectionClosed:
                pass

        await send({
            "type":      "connected",
            "message":   "Transcription server ready",
            "model":     MODEL_PATH,
            "config":    config,
            "timestamp": _now(),
        })

        try:
            async for message in websocket:

                # ── Binary: 3-second audio chunk ───────────────────────────────
                if isinstance(message, bytes):
                    if is_busy:
                        await send({"type": "busy"})
                        continue

                    if len(message) < 44:          # smaller than WAV header
                        continue

                    try:
                        audio = decode_wav(message)
                    except Exception as e:
                        log.warning("Decode failed [%d]: %s", cid, e)
                        continue

                    duration = len(audio) / SAMPLE_RATE
                    if duration < MIN_AUDIO_SECS:
                        continue

                    # Silence gate
                    if np.sqrt(np.mean(audio ** 2)) < SILENCE_RMS:
                        log.info("Silence detected [%d]", cid)
                        await send({"type": "silence"})
                        continue

                    is_busy = True
                    t0 = perf_counter()

                    try:
                        current_task = loop.run_in_executor(
                            self._executor, self._transcribe, audio, config["language"],
                        )
                        text    = await current_task
                        elapsed = perf_counter() - t0

                        if text:
                            await send({
                                "type":            "caption",
                                "text":            text,
                                "duration":        round(duration, 2),
                                "processing_time": round(elapsed, 3),
                                "rtf":             round(elapsed / duration, 3),
                                "timestamp":       _now(),
                            })
                            log.info("[%d] %.1fs audio → %.2fs | %s",
                                     cid, duration, elapsed, text)
                        else:
                            await send({"type": "silence"})
                    except asyncio.CancelledError:
                        log.info("Transcription cancelled [%d]", cid)
                        break
                    finally:
                        is_busy      = False
                        current_task = None

                # ── Text: control messages ─────────────────────────────────────
                elif isinstance(message, str):
                    try:
                        data = json.loads(message)
                    except json.JSONDecodeError:
                        continue

                    msg_type = data.get("type")

                    if msg_type == "config":
                        if "language" in data:
                            config["language"] = data["language"]
                        await send({"type": "config_updated", "config": config})

                    elif msg_type == "ping":
                        await send({"type": "pong", "timestamp": _now()})

                    elif msg_type == "clear_buffer":
                        await send({"type": "buffer_cleared"})

        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception as e:
            log.exception("Unhandled error [%d]: %s", cid, e)
        finally:
            if current_task and not current_task.done():
                current_task.cancel()
            self._connections.discard(websocket)
            log.info("Client disconnected [%d]  (total: %d)",
                     cid, len(self._connections))

    # ── Server entry point ─────────────────────────────────────────────────────

    async def start(self) -> None:
        log.info("=" * 55)
        log.info("  TRANSCRIPTION SERVER  ws://%s:%d", HOST, PORT)
        log.info("  Model:  %s", MODEL_PATH)
        log.info("  Mode:   3 s chunks · no sliding window")
        log.info("=" * 55)

        async with websockets.serve(
            self.handle_client,
            HOST,
            PORT,
            max_size=10 * 1024 * 1024,
            ping_interval=20,
            ping_timeout=10,
        ):
            await asyncio.Future()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Main ───────────────────────────────────────────────────────────────────────

async def _main() -> None:
    server = TranscriptionServer()
    await server.start()


if __name__ == "__main__":
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        log.info("Server stopped.")