#!/usr/bin/env python3
"""
Model Conversion Script
Converts HuggingFace Whisper model to CTranslate2 format for faster inference
"""
import subprocess
import sys
from pathlib import Path
import time


# Configuration
HF_MODEL_ID = "hypaai/wspr_small_2025-11-11_12-12-17"
OUTPUT_DIR = "./wspr_small_ct2"
QUANTIZATION = "int8"  # Options: int8, int8_float16, float16, float32


def check_ctranslate2_installed():
    """Check if ctranslate2 is installed"""
    try:
        import ctranslate2
        print(f"✓ ctranslate2 is installed (version {ctranslate2.__version__})")
        return True
    except ImportError:
        print("❌ ctranslate2 is not installed")
        return False


def install_ctranslate2():
    """Install ctranslate2"""
    print("\n📦 Installing ctranslate2...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "ctranslate2"
        ])
        print("✓ ctranslate2 installed successfully\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install ctranslate2: {e}")
        return False


def convert_model():
    """Convert the HuggingFace model to CTranslate2 format"""
    output_path = Path(OUTPUT_DIR)
    
    # Check if already converted
    if output_path.exists() and (output_path / "model.bin").exists():
        print(f"\n⚠️  Converted model already exists at: {OUTPUT_DIR}")
        response = input("Do you want to reconvert? (y/N): ").strip().lower()
        if response != 'y':
            print("Skipping conversion.")
            return True
        print("\nRemoving existing directory...")
        import shutil
        shutil.rmtree(output_path)
    
    print("\n" + "="*60)
    print("🚀 CONVERTING MODEL TO CTRANSLATE2 FORMAT")
    print("="*60)
    print(f"Model: {HF_MODEL_ID}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Quantization: {QUANTIZATION}")
    print("="*60 + "\n")
    
    print("⏳ Starting conversion (this may take 2-5 minutes)...\n")
    
    start_time = time.time()
    
    try:
        # Run conversion command
        result = subprocess.run([
            "ct2-transformers-converter",
            "--model", HF_MODEL_ID,
            "--output_dir", OUTPUT_DIR,
            "--copy_files", "tokenizer.json", "preprocessor_config.json",
            "--quantization", QUANTIZATION,
        ], capture_output=True, text=True)
        
        # Show output
        if result.stdout:
            print(result.stdout)
        
        if result.returncode != 0:
            print(f"\n❌ Conversion failed!")
            if result.stderr:
                print(f"Error: {result.stderr}")
            
            # Try without --copy_files if it failed
            print("\n⚠️  Retrying without --copy_files...")
            result = subprocess.run([
                "ct2-transformers-converter",
                "--model", HF_MODEL_ID,
                "--output_dir", OUTPUT_DIR,
                "--quantization", QUANTIZATION,
            ], capture_output=True, text=True)
            
            if result.stdout:
                print(result.stdout)
            
            if result.returncode != 0:
                print(f"\n❌ Conversion failed again!")
                if result.stderr:
                    print(f"Error: {result.stderr}")
                return False
        
        elapsed = time.time() - start_time
        print(f"\n✓ Conversion completed in {elapsed:.1f} seconds!")
        return True
        
    except FileNotFoundError:
        print("❌ ct2-transformers-converter command not found!")
        print("   This usually means ctranslate2 wasn't installed correctly.")
        return False
    except Exception as e:
        print(f"❌ Conversion error: {e}")
        return False


def verify_conversion():
    """Verify the converted model files exist"""
    output_path = Path(OUTPUT_DIR)
    
    print("\n" + "="*60)
    print("🔍 VERIFYING CONVERSION")
    print("="*60)
    
    required_files = ["model.bin", "config.json"]
    optional_files = ["tokenizer.json", "preprocessor_config.json", "vocabulary.txt"]
    
    all_good = True
    
    # Check required files
    for filename in required_files:
        filepath = output_path / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"✓ {filename:<30} ({size_mb:.1f} MB)")
        else:
            print(f"❌ {filename:<30} MISSING!")
            all_good = False
    
    # Check optional files
    for filename in optional_files:
        filepath = output_path / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"✓ {filename:<30} ({size_mb:.2f} MB)")
        else:
            print(f"⚠️  {filename:<30} (optional)")
    
    # Calculate total size
    total_size = sum(f.stat().st_size for f in output_path.glob("*") if f.is_file())
    total_size_mb = total_size / (1024 * 1024)
    
    print("="*60)
    print(f"📦 Total model size: {total_size_mb:.1f} MB")
    print("="*60)
    
    return all_good


def show_next_steps():
    """Show what to do next"""
    print("\n" + "="*60)
    print("🎉 CONVERSION COMPLETE!")
    print("="*60)
    print("\n📋 Next Steps:\n")
    print("1. Install faster-whisper:")
    print("   pip install faster-whisper")
    print()
    print("2. Test your converted model:")
    print("   python main_faster.py --audio ./test.wav --benchmark")
    print()
    print("3. Expected performance:")
    print("   - Load time: ~0.5-1s (vs current 6s)")
    print("   - RTF: ~0.1-0.15x (vs current 0.44x)")
    print("   - 5-10x speedup!")
    print()
    print("="*60 + "\n")


def main():
    print("\n" + "="*60)
    print("🔄 MODEL CONVERSION TOOL")
    print("="*60)
    print(f"Converting: {HF_MODEL_ID}")
    print(f"To: {OUTPUT_DIR}")
    print("="*60 + "\n")
    
    # Step 1: Check/install ctranslate2
    if not check_ctranslate2_installed():
        response = input("Install ctranslate2 now? (Y/n): ").strip().lower()
        if response in ['', 'y', 'yes']:
            if not install_ctranslate2():
                print("\n❌ Failed to install ctranslate2. Exiting.")
                sys.exit(1)
        else:
            print("\n❌ ctranslate2 is required. Exiting.")
            sys.exit(1)
    
    # Step 2: Convert model
    if not convert_model():
        print("\n❌ Conversion failed. Please check the errors above.")
        sys.exit(1)
    
    # Step 3: Verify conversion
    if not verify_conversion():
        print("\n⚠️  Some files are missing, but the model might still work.")
    
    # Step 4: Show next steps
    show_next_steps()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Conversion interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
