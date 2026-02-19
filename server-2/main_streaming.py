"""
Real-time streaming transcription for live captions
Processes audio in chunks as it's captured from microphone

Install: pip install faster-whisper pyaudio numpy webrtcvad
"""
import argparse
import time
import wave
from pathlib import Path
import numpy as np
from faster_whisper import WhisperModel
import pyaudio
import webrtcvad


# Your custom model (must be converted to CT2 format first)
CT2_MODEL_PATH = "./wspr_small_ct2"

class StreamingTranscriber:
	"""Real-time audio transcription with VAD"""
	
	def __init__(self, model_path=CT2_MODEL_PATH, language="en", compute_type="int8"):
		print(f"Loading custom model: {model_path}")
		self.model = WhisperModel(model_path, device="cpu", compute_type=compute_type)
		self.language = language
		self.vad = webrtcvad.Vad(2)  # Aggressiveness: 0-3 (3 = most aggressive)
		
		# Audio settings
		self.sample_rate = 16000
		self.chunk_duration = 1.0  # Process 1-second chunks
		self.chunk_size = int(self.sample_rate * self.chunk_duration)
		
		print(f"✓ Streaming transcriber ready (model: {model_size})")
	
	def is_speech(self, audio_chunk):
		"""Check if audio chunk contains speech using VAD"""
		# Convert to 16-bit PCM for VAD
		audio_int16 = (audio_chunk * 32767).astype(np.int16)
		
		# VAD requires specific frame sizes (10, 20, or 30 ms)
		frame_duration_ms = 30
		frame_size = int(self.sample_rate * frame_duration_ms / 1000)
		
		# Check multiple frames
		speech_frames = 0
		total_frames = 0
		
		for i in range(0, len(audio_int16) - frame_size, frame_size):
			frame = audio_int16[i:i + frame_size].tobytes()
			try:
				if self.vad.is_speech(frame, self.sample_rate):
					speech_frames += 1
				total_frames += 1
			except:
				pass
		
		# Consider speech if >30% of frames contain speech
		return total_frames > 0 and (speech_frames / total_frames) > 0.3
	
	def transcribe_chunk(self, audio_data):
		"""Transcribe a single audio chunk"""
		if len(audio_data) < self.sample_rate * 0.1:  # Skip very short clips
			return ""
		
		# Save chunk to temp file (faster-whisper needs file path)
		temp_file = Path("/tmp/chunk.wav")
		with wave.open(str(temp_file), 'wb') as wf:
			wf.setnchannels(1)
			wf.setsampwidth(2)  # 16-bit
			wf.setframerate(self.sample_rate)
			audio_int16 = (audio_data * 32767).astype(np.int16)
			wf.writeframes(audio_int16.tobytes())
		
		# Transcribe
		segments, _ = self.model.transcribe(
			str(temp_file),
			language=self.language,
			beam_size=1,
			vad_filter=False,  # Already filtered
			condition_on_previous_text=False,
		)
		
		text = " ".join([seg.text for seg in segments])
		return text.strip()
	
	def process_microphone(self, duration_seconds=30):
		"""Capture and transcribe from microphone in real-time"""
		print(f"\n🎤 Recording for {duration_seconds} seconds...")
		print("Speak naturally, transcription will appear in real-time\n")
		print("="*60)
		
		audio = pyaudio.PyAudio()
		stream = audio.open(
			format=pyaudio.paFloat32,
			channels=1,
			rate=self.sample_rate,
			input=True,
			frames_per_buffer=self.chunk_size,
		)
		
		buffer = []
		start_time = time.time()
		
		try:
			while (time.time() - start_time) < duration_seconds:
				# Read audio chunk
				data = stream.read(self.chunk_size, exception_on_overflow=False)
				audio_chunk = np.frombuffer(data, dtype=np.float32)
				
				buffer.append(audio_chunk)
				
				# Process when we have enough audio (2-3 seconds for better context)
				if len(buffer) >= 2:
					combined_audio = np.concatenate(buffer)
					
					# Check if contains speech
					if self.is_speech(combined_audio):
						chunk_start = time.time()
						text = self.transcribe_chunk(combined_audio)
						chunk_time = time.time() - chunk_start
						
						if text:
							timestamp = time.time() - start_time
							print(f"[{timestamp:.1f}s | {chunk_time:.2f}s] {text}")
					
					# Keep last chunk for context
					buffer = [buffer[-1]]
		
		finally:
			stream.stop_stream()
			stream.close()
			audio.terminate()
			print("="*60)
			print("\n✓ Recording finished\n")
	
	def process_file(self, audio_path: Path):
		"""Process pre-recorded file in chunks (simulates streaming)"""
		print(f"\n📄 Processing file in streaming mode: {audio_path.name}\n")
		print("="*60)
		
		# Load full audio
		import librosa
		audio, sr = librosa.load(str(audio_path), sr=self.sample_rate)
		duration = len(audio) / self.sample_rate
		
		# Process in chunks
		buffer = []
		process_interval = int(self.sample_rate * 2)  # Process every 2 seconds
		
		for i in range(0, len(audio), self.chunk_size):
			chunk = audio[i:i + self.chunk_size]
			buffer.append(chunk)
			
			# Process accumulated buffer
			if len(buffer) >= 2 or i + self.chunk_size >= len(audio):
				combined = np.concatenate(buffer)
				
				if self.is_speech(combined):
					start = time.time()
					text = self.transcribe_chunk(combined)
					elapsed = time.time() - start
					
					if text:
						timestamp = i / self.sample_rate
						print(f"[{timestamp:.1f}s | {elapsed:.2f}s] {text}")
				
				buffer = [buffer[-1]] if len(buffer) > 1 else []
		
		print("="*60)
		print(f"\n✓ Processed {duration:.1f}s of audio\n")


def main():
	parser = argparse.ArgumentParser(description="Real-time streaming transcription")
	parser.add_argument("--audio", help="Audio file (if omitted, uses microphone)")
	parser.add_argument("--model", default=CT2_MODEL_PATH, 
		help=f"Path to CT2 model (default: {CT2_MODEL_PATH})")
	parser.add_argument("--language", default="en", help="Language code")
	parser.add_argument("--duration", type=int, default=30, 
		help="Recording duration in seconds (microphone mode)")
	args = parser.parse_args()
	
	# Initialize transcriber
	transcriber = StreamingTranscriber(
		model_path=args.model,
		language=args.language,
		compute_type="int8"
	)
	
	if args.audio:
		# Process file
		audio_path = Path(args.audio)
		if not audio_path.exists():
			raise FileNotFoundError(f"Audio not found: {audio_path}")
		transcriber.process_file(audio_path)
	else:
		# Live microphone
		transcriber.process_microphone(duration_seconds=args.duration)


if __name__ == "__main__":
	main()
