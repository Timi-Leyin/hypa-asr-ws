# Real-Time Optimization Guide

## Your Model
**ONLY this model is used:** `hypaai/wspr_small_2025-11-11_12-12-17`

All optimizations below use your custom model, not OpenAI's base models.

---

## Current Performance (baseline)
- Model: `hypaai/wspr_small_2025-11-11_12-12-17` via transformers
- Device: CPU
- Load time: **6.07s**
- Average RTF: **0.48x** (faster than real-time)
- First transcription: **5.19s** (RTF 0.86x)

## Problem for Live Captions
- ❌ 6s model loading too slow to start
- ❌ 5s first transcription (cold start)
- ❌ No streaming capability
- ✅ Average RTF good, but not consistent

---

## Optimization Paths

### ✅ **Path 1: Quick Wins (No Conversion) - READY NOW**
**Your current `main.py` is already optimized!**

```bash
# Test the optimizations
python main.py --audio ./test.wav --language en --benchmark

# With greedy decoding (fastest)
python main.py --audio ./test.wav --language en --num-beams 1 --benchmark
```

**What's optimized:**
- ✨ BetterTransformer (~2x speedup)
- ✨ Greedy decoding option (num_beams=1, fastest)
- ✨ torch.no_grad() (lower memory)
- ✨ Model kept loaded in benchmark mode

**Expected improvement: 2-3x faster**
- Load time: ~3-4s (from 6s)
- RTF: ~0.2-0.25x (from 0.48x)

---

### 🚀 **Path 2: Maximum Speed (Requires Conversion)**
**Convert your model to CTranslate2 for 5-10x speedup**

See **[MODEL_CONVERSION.md](MODEL_CONVERSION.md)** for full instructions.

**Quick start:**
```bash
# 1. Install conversion tool
pip install ctranslate2

# 2. Convert your model
ct2-transformers-converter \
    --model hypaai/wspr_small_2025-11-11_12-12-17 \
    --output_dir ./wspr_small_ct2 \
    --quantization int8

# 3. Run with faster-whisper
pip install faster-whisper
python main_faster.py --audio ./test.wav --benchmark
```

**Expected results:**
- Load time: **0.5-1s** (vs 6s) - 6x faster
- RTF: **0.1-0.15x** (vs 0.48x) - 3-4x faster
- Memory: 4x smaller (INT8 quantization)

---

## Key Optimizations Explained
## Key Optimizations Explained

### 1️⃣ **BetterTransformer**
**Already enabled in main.py!**

What it does:
- FastPath execution (C++ kernels)
- Flash Attention for transformers
- ~2x speedup on CPU

```bash
# Enabled by default
python main.py --audio ./test.wav --benchmark

# Disable if it causes issues
python main.py --audio ./test.wav --no-bettertransformer
```

---

### 2️⃣ **Greedy Decoding (num_beams=1)**
**Fastest generation strategy**

```bash
# Greedy decoding (fastest)
python main.py --audio ./test.wav --num-beams 1

# Beam search (better quality, slower)
python main.py --audio ./test.wav --num-beams 5
```

| Beams | Speed | Quality |
|-------|-------|---------|
| 1 | Fastest | Good |
| 3 | Balanced | Better |
| 5 | Slower | Best |

---

### 3️⃣ **Voice Activity Detection (VAD)**
**Available after model conversion to CT2**

Skips silence and background noise:
- 2-3x faster on typical speech
- Better accuracy (less hallucination)
- Lower latency

Used in `main_streaming.py` for real-time captions.

---

### 4️⃣ **INT8 Quantization**
**Available after CT2 conversion**

```bash
# Conversion with INT8
ct2-transformers-converter \
    --model hypaai/wspr_small_2025-11-11_12-12-17 \
    --output_dir ./wspr_small_ct2 \
    --quantization int8
```

Benefits:
- 4x smaller model size
- 2-4x faster inference
- Minimal quality loss (<5%)

---

### 5️⃣ **Keep Model Loaded**
**Already implemented in benchmark mode!**

Your benchmark shows the impact:
```
Transcription #1: 5.19s (includes warmup)
Transcription #2: 2.23s (warm)
Transcription #3: 1.38s (warm)
```

**For production:** Load model once at startup, reuse for all transcriptions.

---

## Testing Optimizations

### Test Current Setup (Optimized)
```bash
python main.py --audio ./test.wav --language en --benchmark
```

### Test With Different Beam Sizes
```bash
# Greedy (fastest)
python main.py --audio ./test.wav --num-beams 1 --benchmark

# Beam search (quality)
python main.py --audio ./test.wav --num-beams 5 --benchmark
```

### Test After Conversion (Maximum Speed)
```bash
# Convert model first (see MODEL_CONVERSION.md)
ct2-transformers-converter \
    --model hypaai/wspr_small_2025-11-11_12-12-17 \
    --output_dir ./wspr_small_ct2 \
    --quantization int8

# Run with faster-whisper
python main_faster.py --audio ./test.wav --benchmark
```

---

## Expected Performance Summary

| Setup | Load Time | RTF | Speedup |
|-------|-----------|-----|---------|
| **Current (baseline)** | 6.07s | 0.48x | 1x |
| **Optimized transformers** | ~3-4s | ~0.2x | 2-3x |
| **CT2 + INT8** | ~0.5-1s | ~0.1-0.15x | 5-10x |

---

## Quick Start Recommendations

### For Immediate Use (No Conversion)
```bash
# Use optimized main.py with greedy decoding
python main.py --audio ./test.wav --language en --num-beams 1 --benchmark
```

### For Maximum Performance (Requires Setup)
1. **Convert model** (one-time, ~5 minutes):
   ```bash
   pip install ctranslate2
   ct2-transformers-converter \
       --model hypaai/wspr_small_2025-11-11_12-12-17 \
       --output_dir ./wspr_small_ct2 \
       --quantization int8
   ```

2. **Install faster-whisper**:
   ```bash
   pip install faster-whisper
   ```

3. **Run**:
   ```bash
   python main_faster.py --audio ./test.wav --benchmark
   ```

---

## Production Setup Example

```python
# Load model once at startup
from faster_whisper import WhisperModel

model = WhisperModel(
    "./wspr_small_ct2",  # Your converted model
    device="cpu",
    compute_type="int8",
)

# Transcribe with optimizations
segments, info = model.transcribe(
    audio_path,
    language="en",
    beam_size=1,  # Greedy = fastest
    vad_filter=True,  # Skip silence
    condition_on_previous_text=False,  # Faster
)

text = " ".join([seg.text for seg in segments])
```

This will give you **true real-time performance** for live captions! 🚀
