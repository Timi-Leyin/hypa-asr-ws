"""
Optimized real-time transcription using faster-whisper
Install: pip install faster-whisper
"""
import argparse
from pathlib import Path
import time
from faster_whisper import WhisperModel


# Your custom HuggingFace model
HF_MODEL_ID = "hypaai/wspr_small_2025-11-11_12-12-17"

# NOTE: faster-whisper requires CTranslate2 format.
# To convert your model, run:
# pip install ctranslate2
# ct2-transformers-converter --model hypaai/wspr_small_2025-11-11_12-12-17 \
#     --output_dir ./wspr_small_ct2 --copy_files tokenizer.json preprocessor_config.json
# Then use: MODEL_PATH = "./wspr_small_ct2"

MODEL_PATH = "./wspr_small_ct2"  # Path to converted CT2 model


def load_model(model_path: str = MODEL_PATH, device: str = "cpu", compute_type: str = "int8"):
	"""
	Load faster-whisper model with optimizations
	
	Args:
		model_path: Path to your converted CT2 model
		device: cpu, cuda, or auto
		compute_type: int8, int8_float16, float16, float32
			- int8: Fastest, 4x smaller, slight quality loss
			- float32: Best quality, slower
	"""
	print(f"\n🚀 Loading custom model from {model_path}...")
	print(f"   Device: {device}, Compute: {compute_type}\n")
	start = time.time()
	
	model = WhisperModel(
		model_path,
		device=device,
		compute_type=compute_type,
		cpu_threads=4,  # Adjust based on your CPU cores
		num_workers=1,  # For parallel processing
	)
	
	load_time = time.time() - start
	print(f"✓ Model loaded in {load_time:.2f}s\n")
	return model, load_time


def transcribe_audio(model, audio_path: Path, language: str = "en"):
	"""Transcribe audio file"""
	print(f"📄 Transcribing: {audio_path.name}")
	
	start = time.time()
	
	# Transcribe with optimizations
	segments, info = model.transcribe(
		str(audio_path),
		language=language,
		beam_size=1,  # 1 = greedy (fastest), 5 = default (better quality)
		vad_filter=True,  # Skip silence - huge speedup!
		vad_parameters=dict(
			threshold=0.5,
			min_speech_duration_ms=250,
			min_silence_duration_ms=100,
		),
		condition_on_previous_text=False,  # Faster, less context
	)
	
	# Collect segments
	transcription = " ".join([segment.text for segment in segments])
	
	elapsed = time.time() - start
	audio_duration = info.duration
	rtf = elapsed / audio_duration if audio_duration > 0 else 0
	
	return transcription.strip(), elapsed, audio_duration, rtf


def benchmark_mode(model, audio_path: Path, language: str):
	"""Run benchmark: transcribe 3 times"""
	print("\n" + "="*60)
	print("📊 BENCHMARK MODE")
	print("="*60 + "\n")
	
	times = []
	text = ""
	audio_duration = 0
	
	for i in range(1, 4):
		print(f"🎤 Transcription #{i}...")
		text, elapsed, audio_duration, rtf = transcribe_audio(model, audio_path, language)
		times.append(elapsed)
		print(f"✓ Completed in {elapsed:.2f}s (RTF: {rtf:.2f}x)\n")
	
	avg_time = sum(times) / len(times)
	avg_rtf = avg_time / audio_duration if audio_duration > 0 else 0
	
	print("="*60)
	print("📊 RESULTS")
	print("="*60)
	print(f"🎵 Audio duration: {audio_duration:.2f}s")
	for i, t in enumerate(times, 1):
		print(f"  Run #{i}: {t:.2f}s (RTF: {t/audio_duration:.2f}x)")
	print(f"\n⚡ Average: {avg_time:.2f}s (RTF: {avg_rtf:.2f}x)")
	print(f"📝 Text: {text[:100]}...")
	print("="*60 + "\n")


def main():
	parser = argparse.ArgumentParser(description="Faster Whisper transcription with custom model")
	parser.add_argument("--audio", required=True, help="Audio file path")
	parser.add_argument("--model", default=MODEL_PATH, 
		help=f"Path to converted CT2 model (default: {MODEL_PATH})")
	parser.add_argument("--language", default="en", help="Language code")
	parser.add_argument("--compute-type", default="int8", 
		choices=["int8", "int8_float16", "float16", "float32"],
		help="Compute type (int8 fastest)")
	parser.add_argument("--benchmark", action="store_true", help="Benchmark mode")
	args = parser.parse_args()
	
	audio_path = Path(args.audio)
	if not audio_path.exists():
		raise FileNotFoundError(f"Audio not found: {audio_path}")
	
	# Load model once
	model, load_time = load_model(args.model, device="cpu", compute_type=args.compute_type)
	
	if args.benchmark:
		benchmark_mode(model, audio_path, args.language)
	else:
		text, elapsed, duration, rtf = transcribe_audio(model, audio_path, args.language)
		
		print("\n" + "="*60)
		print("TRANSCRIPTION")
		print("="*60)
		print(text)
		print("="*60)
		print(f"⏱️  Time: {elapsed:.2f}s")
		print(f"🎵 Duration: {duration:.2f}s")
		print(f"⚡ RTF: {rtf:.2f}x\n")


if __name__ == "__main__":
	main()
