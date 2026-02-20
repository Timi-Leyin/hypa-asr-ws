import base64
import io
import wave
from typing import Optional

import numpy as np
import runpod


MODEL_ID = "hypaai/wspr_small_2025-11-11_12-12-17"
SAMPLE_RATE = 16_000
TEMPERATURE = 0.3
BEAM_SIZE = 1
REPETITION_PENALTY = 1.0

_MODEL = None
_PROCESSOR = None
_DEVICE = None
_TORCH = None


def _load_model():
    global _MODEL, _PROCESSOR, _DEVICE, _TORCH
    if _MODEL is not None:
        return

    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    _TORCH = torch
    if torch.cuda.is_available():
        _DEVICE = "cuda"
    elif torch.backends.mps.is_available():
        _DEVICE = "mps"
    else:
        _DEVICE = "cpu"

    _PROCESSOR = WhisperProcessor.from_pretrained(MODEL_ID)
    _MODEL = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
    _MODEL.to(_DEVICE)
    _MODEL.eval()


def _decode_wav(raw: bytes, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    with io.BytesIO(raw) as buf, wave.open(buf, "rb") as wf:
        ch = wf.getnchannels()
        sw = wf.getsampwidth()
        fr = wf.getframerate()
        nf = wf.getnframes()
        pcm = wf.readframes(nf)

        scale = 32_768.0 if sw == 2 else 2_147_483_648.0
        dtype = np.int16 if sw == 2 else np.int32
        arr = np.frombuffer(pcm, dtype=dtype).astype(np.float32) / scale

        if ch == 2:
            arr = arr.reshape(-1, 2).mean(axis=1)
        if fr != target_sr:
            import librosa
            arr = librosa.resample(arr, orig_sr=fr, target_sr=target_sr)
        return arr


def _decode_base64_audio(audio_b64: str) -> bytes:
    if "," in audio_b64:
        audio_b64 = audio_b64.split(",", 1)[1]
    return base64.b64decode(audio_b64)


def transcribe_base64_audio(audio_b64: str, language: str = "en") -> Optional[str]:
    _load_model()
    audio_bytes = _decode_base64_audio(audio_b64)
    audio = _decode_wav(audio_bytes)

    inputs = _PROCESSOR(
        audio, sampling_rate=SAMPLE_RATE, return_tensors="pt",
    )
    input_features = inputs.input_features.to(_DEVICE)

    gen_kwargs = {
        "num_beams": BEAM_SIZE,
        "temperature": TEMPERATURE if TEMPERATURE > 0 else 1.0,
        "do_sample": TEMPERATURE > 0,
        "repetition_penalty": REPETITION_PENALTY,
        "language": language,
        "task": "transcribe",
    }

    with _TORCH.no_grad():
        predicted_ids = _MODEL.generate(input_features, **gen_kwargs)

    text = _PROCESSOR.batch_decode(predicted_ids, skip_special_tokens=True)
    result = " ".join(t.strip() for t in text if t.strip())
    return result or None


def handler(event):
    input_data = event.get("input", {})
    audio_b64 = input_data.get("audio_base64") or input_data.get("audio")
    if not audio_b64:
        return {"status": False, "error": "Missing 'audio_base64' in input."}

    language = input_data.get("language") or "en"
    try:
        text = transcribe_base64_audio(audio_b64, language=language)
        return {
            "status": True,
            "text": text or "",
            "language": language,
        }
    except Exception as exc:
        return {
            "status": False,
            "error": f"Transcription failed: {exc}",
        }


runpod.serverless.start({"handler": handler})