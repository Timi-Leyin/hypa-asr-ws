# Converting Your Model for faster-whisper

## Why Convert?
- `faster-whisper` uses **CTranslate2** format (not HuggingFace transformers)
- Your model `hypaai/wspr_small_2025-11-11_12-12-17` needs conversion
- **5-10x speedup** after conversion

## Step-by-Step Conversion

### 1. Install conversion tool
```bash
source .venv/bin/activate
pip install ctranslate2 transformers
```

### 2. Convert your model
```bash
ct2-transformers-converter \
    --model hypaai/wspr_small_2025-11-11_12-12-17 \
    --output_dir ./wspr_small_ct2 \
    --copy_files tokenizer.json preprocessor_config.json \
    --quantization int8
```

**Parameters:**
- `--model`: Your HuggingFace model ID
- `--output_dir`: Where to save converted model
- `--copy_files`: Required tokenizer files
- `--quantization int8`: Enable INT8 for 4x smaller + 2-4x faster

### 3. Verify conversion
```bash
ls -lh ./wspr_small_ct2/
```

You should see:
```
model.bin              # Converted model weights
tokenizer.json         # Tokenizer
preprocessor_config.json
config.json
vocabulary.txt         # Optional
```

### 4. Test the converted model
```bash
python main_faster.py --audio ./test.wav --language en --benchmark
```

## Expected Results

### Before (transformers)
```
Load: 6.07s
RTF: 0.48x avg
```

### After (faster-whisper + CT2)
```
Load: ~0.5-1s     (6x faster load)
RTF: ~0.1-0.15x   (3-4x faster inference)
```

## Troubleshooting

### Error: "No module named 'ctranslate2'"
```bash
pip install ctranslate2
```

### Error: "Cannot find tokenizer.json"
The model might not have this file. Try without `--copy_files`:
```bash
ct2-transformers-converter \
    --model hypaai/wspr_small_2025-11-11_12-12-17 \
    --output_dir ./wspr_small_ct2 \
    --quantization int8
```

### Error during conversion
Some custom models may not be compatible. Fallback to optimized transformers:
```bash
# Use the optimized main.py instead
python main.py --audio ./test.wav --language en --benchmark
```

## Alternative: Optimize Current Setup (No Conversion)

If conversion doesn't work, use the optimized `main.py`:

```bash
# With BetterTransformer (~2x speedup)
python main.py --audio ./test.wav --language en --benchmark

# Greedy decoding (fastest)
python main.py --audio ./test.wav --language en --num-beams 1

# Disable BetterTransformer if it causes issues
python main.py --audio ./test.wav --language en --no-bettertransformer
```

**Optimizations applied:**
- ✅ BetterTransformer (~2x speedup)
- ✅ Greedy decoding (num_beams=1)
- ✅ torch.no_grad() (less memory)
- ✅ Model kept loaded (save 6s per run)

**Expected improvement: 2-3x faster** (without conversion)
