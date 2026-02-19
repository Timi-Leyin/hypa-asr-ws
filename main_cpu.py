import argparse
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor


MODEL_ID = "hypaai/wspr_small_2025-11-11_12-12-17"
BASE_MODEL_ID = "openai/whisper-small"


def load_model_and_processor(model_id: str, use_sampling: bool = False):
	use_mps = torch.backends.mps.is_available() and not use_sampling  # MPS has sampling issues
	device = "mps" if use_mps else "cpu"
	dtype = torch.float16 if use_mps else torch.float32

	processor = AutoProcessor.from_pretrained(model_id)
	model = AutoModelForSpeechSeq2Seq.from_pretrained(
		model_id,
		torch_dtype=dtype,
		low_cpu_mem_usage=True,
	).to(device)
	
	if use_sampling and not use_mps:
		print(f"⚠️  Using CPU for sampling (MPS has compatibility issues with temperature/top_p/top_k)")

	return model, processor, device


def transcribe_audio(
	audio_path: Path,
	model_id: str,
	language: Optional[str],
	task: str,
	temperature: float = 0.0,
	top_p: Optional[float] = None,
	top_k: Optional[int] = None,
):
	if not audio_path.exists() or not audio_path.is_file():
		raise FileNotFoundError(f"Audio file not found: {audio_path}")

	print(f"Loading audio from: {audio_path}")
	print(f"File size: {audio_path.stat().st_size / 1024:.1f} KB")
	
	use_sampling = temperature > 0
	model, processor, device = load_model_and_processor(model_id, use_sampling=use_sampling)

	# Load and process audio
	import librosa
	audio_array, sampling_rate = librosa.load(str(audio_path), sr=16000)
	duration = len(audio_array) / sampling_rate
	print(f"Audio duration: {duration:.2f} seconds")
	
	inputs = processor(
		audio_array,
		sampling_rate=16000,
		return_tensors="pt",
	)
	
	# Move inputs to device and match model dtype
	input_features = inputs.input_features.to(device).to(model.dtype)
	print(f"Input features shape: {input_features.shape}")

	print(f"Running transcription with task='{task}', language='{language or 'en'}'...")
	print(f"Generation params: temperature={temperature}, top_p={top_p}, top_k={top_k}")
	
	# Build generation kwargs
	generate_kwargs = {
		"language": language if language else "en",
		"task": task,
	}
	
	if temperature > 0:
		generate_kwargs["do_sample"] = True
		generate_kwargs["temperature"] = temperature
		if top_p is not None:
			generate_kwargs["top_p"] = top_p
		if top_k is not None:
			generate_kwargs["top_k"] = top_k
	else:
		generate_kwargs["do_sample"] = False
	
	# Generate transcription
	with torch.no_grad():
		predicted_ids = model.generate(
			input_features,
			**generate_kwargs,
		)

	# Decode the result
	print(f"Generated token IDs shape: {predicted_ids.shape}")
	print(f"Generated token IDs: {predicted_ids[0].tolist()}")
	
	transcription = processor.batch_decode(
		predicted_ids,
		skip_special_tokens=True,
	)[0]
	
	print(f"Raw result: {transcription}")
	return transcription.strip()


def parse_args():
	parser = argparse.ArgumentParser(
		description="Run local transcription with a Hugging Face Whisper model."
	)
	parser.add_argument("--audio", required=True, help="Path to local audio file")
	parser.add_argument(
		"--model",
		default=MODEL_ID,
		help=f"Hugging Face model id. Use '{BASE_MODEL_ID}' for the base model (default: {MODEL_ID})",
	)
	parser.add_argument(
		"--language",
		default=None,
		help="Optional language token (example: en, yo).",
	)
	parser.add_argument(
		"--task",
		choices=["transcribe", "translate"],
		default="transcribe",
		help="Whisper generation task",
	)
	parser.add_argument(
		"--temperature",
		type=float,
		default=0.0,
		help="Sampling temperature (0.0 = greedy, higher = more random). Try 0.2-0.8 for variety.",
	)
	parser.add_argument(
		"--top-p",
		type=float,
		default=None,
		help="Nucleus sampling threshold (0.0-1.0). Try 0.95.",
	)
	parser.add_argument(
		"--top-k",
		type=int,
		default=None,
		help="Top-k sampling (e.g., 50). Limits to top k tokens.",
	)
	return parser.parse_args()


def main():
	args = parse_args()
	print("\n🎤 Transcribing audio...\n")
	text = transcribe_audio(
		audio_path=Path(args.audio),
		model_id=args.model,
		language=args.language,
		task=args.task,
		temperature=args.temperature,
		top_p=args.top_p,
		top_k=args.top_k,
	)
	print("\n" + "=" * 60)
	print("TRANSCRIPTION:")
	print("=" * 60)
	print(text)
	print("=" * 60 + "\n")


if __name__ == "__main__":
	main()
