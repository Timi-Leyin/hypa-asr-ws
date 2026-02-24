#!/usr/bin/env python3
"""
Transcription engine – model loading, audio decoding, and inference.

Extracted from websocket_server.py so it can be imported by handler.py
(RunPod serverless) without pulling in server-startup logic.
"""
import io
import json
import logging
import threading
import wave
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from time import perf_counter
from typing import Optional

import asyncio
import numpy as np

try:
    import websockets
except ImportError:
    websockets = None  # handled at runtime

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("transcriber")

# ── Config ─────────────────────────────────────────────────────────────────────
USE_CONVERTED_MODEL = False          # True → faster-whisper (CT2), False → HuggingFace transformers
CT2_MODEL_PATH      = "./wspr_small_ct2"
HF_MODEL_ID         = "hypaai/wspr_small_2025-11-11_12-12-17"
MODEL_PATH          = CT2_MODEL_PATH if USE_CONVERTED_MODEL else HF_MODEL_ID
HOST                = "0.0.0.0"
WS_PORT             = 8765
HTTP_PORT           = 8766
SAMPLE_RATE         = 16_000
SILENCE_RMS         = 0.01          # < val → treat as silence
MIN_AUDIO_SECS      = 0.3
MAX_WORKERS         = 3

# ── Model parameters ───────────────────────────────────────────────────────────
TEMPERATURE         = 0.3
BEAM_SIZE           = 1
REPETITION_PENALTY  = 1.0


# ── Helpers ────────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
        dtype = np.int16  if sw == 2 else np.int32
        arr   = np.frombuffer(pcm, dtype=dtype).astype(np.float32) / scale

        if ch == 2:
            arr = arr.reshape(-1, 2).mean(axis=1)
        if fr != target_sr:
            import librosa
            arr = librosa.resample(arr, orig_sr=fr, target_sr=target_sr)
        return arr


# ── Transcription engine ───────────────────────────────────────────────────────

class TranscriptionServer:

    def __init__(self, shutdown_event: Optional[threading.Event] = None):
        self._use_ct2 = USE_CONVERTED_MODEL
        self._shutdown_event = shutdown_event
        log.info("Loading model from %s …", MODEL_PATH)

        # Detect GPU availability (CUDA > MPS > CPU)
        import torch
        self._torch = torch

        if torch.cuda.is_available():
            self._device = "cuda"
        elif torch.backends.mps.is_available():
            self._device = "mps"  # Apple Silicon GPU
        else:
            self._device = "cpu"

        log.info("Using device: %s", self._device)

        if self._use_ct2:
            from faster_whisper import WhisperModel
            # faster-whisper doesn't support MPS, fall back to CPU
            ct2_device = "cuda" if self._device == "cuda" else "cpu"
            if self._device == "mps":
                log.warning(
                    "faster-whisper doesn't support MPS, using CPU. "
                    "Set USE_CONVERTED_MODEL=False for Apple GPU."
                )
            self.model = WhisperModel(
                CT2_MODEL_PATH,
                device=ct2_device,
                compute_type="int8_float16" if ct2_device == "cuda" else "int8",
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

    def _transcribe(self, audio: np.ndarray, config: dict) -> Optional[str]:
        try:
            if self._use_ct2:
                return self._transcribe_ct2(audio, config)
            else:
                return self._transcribe_hf(audio, config)
        except Exception as e:
            log.error("Transcription error: %s", e)
            return None

    def _transcribe_ct2(self, audio: np.ndarray, config: dict) -> Optional[str]:
        segments, _ = self.model.transcribe(
            audio,
            beam_size=config["beam_size"],
            temperature=config["temperature"],
            condition_on_previous_text=False,
            no_speech_threshold=config["no_speech_threshold"],
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=250),
        )
        parts = [s.text.strip() for s in segments if s.text.strip()]
        return " ".join(parts) or None

    def _transcribe_hf(self, audio: np.ndarray, config: dict) -> Optional[str]:
        inputs = self._processor(
            audio, sampling_rate=SAMPLE_RATE, return_tensors="pt",
        )
        input_features = inputs.input_features.to(self._device)

        temp = config["temperature"]
        gen_kwargs = {
            "num_beams":          config["beam_size"],
            "temperature":        temp if temp > 0 else 1.0,
            "do_sample":          temp > 0,
            "repetition_penalty": config["repetition_penalty"],
            "top_p":              config["top_p"] if temp > 0 else 1.0,
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

    # ── Per-client WebSocket handler ───────────────────────────────────────────

    async def handle_client(self, websocket) -> None:
        import websockets.exceptions

        cid  = id(websocket)
        loop = asyncio.get_running_loop()
        self._connections.add(websocket)
        log.info("Client connected  [%d]  (total: %d)", cid, len(self._connections))

        is_busy      = False
        config       = {
            "temperature":         TEMPERATURE,
            "beam_size":           BEAM_SIZE,
            "repetition_penalty":  REPETITION_PENALTY,
            "top_p":               0.9,
            "no_speech_threshold": 0.6,
        }
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
                            self._executor, self._transcribe, audio, config.copy(),
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
                        _schema = {
                            "temperature":         float,
                            "beam_size":           int,
                            "repetition_penalty":  float,
                            "top_p":               float,
                            "no_speech_threshold": float,
                        }
                        for key, cast in _schema.items():
                            if key in data:
                                config[key] = cast(data[key])
                        log.info("Config updated [%d]: %s", cid, config)
                        await send({"type": "config_updated", "config": config})

                    elif msg_type == "ping":
                        await send({"type": "pong", "timestamp": _now()})

                    elif msg_type == "clear_buffer":
                        await send({"type": "buffer_cleared"})

                    elif msg_type == "shutdown":
                        log.info("Shutdown requested by client [%d]", cid)
                        await send({"type": "shutdown_ack", "timestamp": _now()})
                        if self._shutdown_event is not None:
                            self._shutdown_event.set()
                        break

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
