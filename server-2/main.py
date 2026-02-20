import argparse
from pathlib import Path
from typing import Optional
import time
import librosa
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor



#
#  user 1 ->
#  user 2->
# usrer1 

# 

MODEL_ID = "hypaai/wspr_small_2025-11-11_12-12-17"
BASE_MODEL_ID = "openai/whisper-small"


def load_model_and_processor(model_id: str, use_bettertransformer: bool = True):
	"""Load Whisper model and processor on CPU with optimizations"""
	device = "cpu"
	dtype = torch.float32
	
	print(f"\n{'='*60}")
	print(f"🖥️  Loading model on CPU...")
	print(f"{'='*60}\n")
	
	processor = AutoProcessor.from_pretrained(model_id)
	model = AutoModelForSpeechSeq2Seq.from_pretrained(
		model_id,
		torch_dtype=dtype,
		low_cpu_mem_usage=True,
		use_safetensors=True,
	).to(device)
	
	# Apply BetterTransformer for ~2x speedup
	if use_bettertransformer:
		try:
			from optimum.bettertransformer import BetterTransformer
			model = BetterTransformer.transform(model)
			print("✓ BetterTransformer enabled (~2x speedup)")
		except ImportError:
			try:
				# Try direct method (newer API)
				model = model.to_bettertransformer()
				print("✓ BetterTransformer enabled (~2x speedup)")
			except Exception as e:
				print(f"⚠️  BetterTransformer not available: {e}")
				print("   Install with: pip install optimum")
	else:
		print("✓ Model loaded successfully")
	
	return model, processor, device


def transcribe_audio(
	audio_path: Path,
	model,
	processor,
	device,
	language: Optional[str],
	task: str,
	num_beams: int = 1,
):
	"""Transcribe audio using a loaded model"""
	# Load and process audio
	audio_array, sampling_rate = librosa.load(str(audio_path), sr=16000)
	duration = len(audio_array) / sampling_rate
	
	inputs = processor(
		audio_array,
		sampling_rate=16000,
		return_tensors="pt",
	)
	
	# Move inputs to device
	input_features = inputs.input_features.to(device)
	
	# Build generation kwargs with optimizations
	generate_kwargs = {
		"language": language if language else "en",
		"task": task,
		"num_beams": num_beams,  # 1 = greedy (fastest), 5 = beam search (better quality)
		"do_sample": False,  # Deterministic for speed
	}
	
	# Generate transcription
	with torch.no_grad():
		predicted_ids = model.generate(
			input_features,
			**generate_kwargs,
		)
	
	# Decode the result
	transcription = processor.batch_decode(
		predicted_ids,
		skip_special_tokens=True,
	)[0]
	
	return transcription.strip(), duration


def parse_args():
	parser = argparse.ArgumentParser(
		description="Run local transcription with a Hugging Face Whisper model."
	)
	parser.add_argument("--audio", required=True, help="Path to local audio file")
	parser.add_argument(
		"--model",
		default=MODEL_ID,
		help=f"Hugging Face model id (default: {MODEL_ID})",
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
		"--num-beams",
		type=int,
		default=1,
		help="Beam search size (1=greedy/fastest, 5=better quality/slower)",
	)
	parser.add_argument(
		"--no-bettertransformer",
		action="store_true",
		help="Disable BetterTransformer optimization",
	)
	parser.add_argument(
		"--benchmark",
		action="store_true",
		help="Benchmark mode: Load model once and transcribe 3 times to measure inference speed.",
	)
	return parser.parse_args()


def benchmark_mode(args):
	"""Load model once and transcribe the same audio 3 times"""
	audio_path = Path(args.audio)
	if not audio_path.exists():
		raise FileNotFoundError(f"Audio file not found: {audio_path}")
	
	print("\n" + "=" * 60)
	print("📊 BENCHMARK MODE: Load once, transcribe 3 times")
	print("=" * 60 + "\n")
	
	# Time model loading
	print("⏳ Loading model...")
	load_start = time.time()
	model, processor, device = load_model_and_processor(
		args.model,
		use_bettertransformer=not args.no_bettertransformer
	)
	load_time = time.time() - load_start
	
	print(f"\n✓ Model loaded in {load_time:.2f}s\n")
	print("=" * 60)
	
	# Run 3 transcriptions
	transcription_times = []
	audio_duration = 0
	
	for i in range(1, 4):
		print(f"\n🎤 Transcription #{i}...")
		trans_start = time.time()
		text, audio_duration = transcribe_audio(
			audio_path=audio_path,
			model=model,
			processor=processor,
			device=device,
			language=args.language,
			task=args.task,
			num_beams=args.num_beams,
		)
		trans_time = time.time() - trans_start
		transcription_times.append(trans_time)
		
		print(f"✓ Transcription #{i} completed in {trans_time:.2f}s")
		if i == 1:
			print(f"Text: {text[:100]}{'...' if len(text) > 100 else ''}")
	
	# Calculate statistics
	total_time = load_time + sum(transcription_times)
	avg_trans_time = sum(transcription_times) / len(transcription_times)
	avg_rtf = avg_trans_time / audio_duration if audio_duration > 0 else 0
	
	# Print summary
	print("\n" + "=" * 60)
	print("📊 BENCHMARK RESULTS")
	print("=" * 60)
	print(f"⏱️  Model load time: {load_time:.2f}s")
	print(f"🎵 Audio duration: {audio_duration:.2f}s")
	print()
	for i, t in enumerate(transcription_times, 1):
		rtf = t / audio_duration if audio_duration > 0 else 0
		print(f"  Transcription #{i}: {t:.2f}s (RTF: {rtf:.2f}x)")
	print()
	print(f"📈 Average transcription time: {avg_trans_time:.2f}s")
	print(f"⚡ Average RTF: {avg_rtf:.2f}x {'(faster than real-time)' if avg_rtf < 1 else '(slower than real-time)'}")
	print(f"⏱️  Total time (load + 3x transcribe): {total_time:.2f}s")
	print(f"💡 Overhead per run if reloading: +{load_time:.2f}s")
	print("=" * 60 + "\n")


def main():
	args = parse_args()
	
	# Run benchmark mode if requested
	if args.benchmark:
		benchmark_mode(args)
		return
	
	# Normal single transcription mode
	audio_path = Path(args.audio)
	if not audio_path.exists():
		raise FileNotFoundError(f"Audio file not found: {audio_path}")
	
	print("\n🎤 Transcribing audio...\n")
	
	start_time = time.time()
	model, processor, device = load_model_and_processor(
		args.model,
		use_bettertransformer=not args.no_bettertransformer
	)
	
	text, audio_duration = transcribe_audio(
		audio_path=audio_path,
		model=model,
		processor=processor,
		device=device,
		language=args.language,
		task=args.task,
		num_beams=args.num_beams,
	)
	elapsed_time = time.time() - start_time
	rtf = elapsed_time / audio_duration if audio_duration > 0 else 0
	
	print("\n" + "=" * 60)
	print("TRANSCRIPTION:")
	print("=" * 60)
	print(text)
	print("=" * 60)
	print(f"⏱️  Processing time: {elapsed_time:.2f}s")
	print(f"🎵 Audio duration: {audio_duration:.2f}s")
	print(f"⚡ Real-time factor (RTF): {rtf:.2f}x {'(faster than real-time)' if rtf < 1 else '(slower than real-time)'}\n")


if __name__ == "__main__":
	main()
