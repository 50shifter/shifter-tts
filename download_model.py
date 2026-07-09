#!/usr/bin/env python3
"""
shifter-tts — Download models from HuggingFace Hub.

Usage:
    python download_model.py              # base model (voice cloning)
    python download_model.py custom       # custom voice version
    python download_model.py voice_design # Voice Design
    python download_model.py tokenizer    # tokenizer only (for training)

Notes:
    - Public models are downloaded for FREE without any key
    - If a model requires accepting a license, the terminal will open
    - Free HuggingFace account: https://huggingface.co/join
"""

import subprocess

subprocess.run(["cmd.exe", "/c", "chcp 65001"], shell=True)

import argparse
import os
import sys


MODELS = {
    "base":      ("Qwen/Qwen3-TTS-12Hz-1.7B-Base",     "Base model (voice cloning)"),
    "custom":    ("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", "Custom voice version"),
    "voice_design": ("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", "Voice Design (generate voice by description)"),
    "tokenizer": ("Qwen/Qwen3-TTS-Tokenizer-12Hz",     "12Hz tokenizer (for training)"),
}


def download_model(model_id: str, cache_dir: str = None):
    """Download model from HuggingFace."""
    print(f"Downloading model: {model_id}")
    
    if cache_dir is None:
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    
    try:
        from huggingface_hub import snapshot_download
        
        local_dir = snapshot_download(
            repo_id=model_id,
            cache_dir=cache_dir,
            resume_download=True,
        )
        
        print(f"Model saved to: {local_dir}")
        return local_dir
        
    except Exception as e:
        error_msg = str(e).lower()
        
        if "401" in error_msg or "unauthorized" in error_msg or "authentication" in error_msg:
            print()
            print("=" * 60)
            print("  WARNING: HuggingFace account required")
            print("=" * 60)
            print()
            print(f"  Model '{model_id}' requires accepting license.")
            print()
            print("  1. Create a free account:")
            print("     https://huggingface.co/join")
            print()
            print("  2. Accept the license on the model page:")
            print(f"     https://huggingface.co/{model_id}")
            print()
            print("  3. Login in terminal:")
            print("     huggingface-cli login")
            print()
            print("  Or use a token (if you already have an account):")
            print("     huggingface-cli login --token YOUR_TOKEN_HERE")
            print("=" * 60)
        elif "gated" in error_msg or "access denied" in error_msg:
            print()
            print("=" * 60)
            print("  WARNING: Model requires access (gated model)")
            print("=" * 60)
            print()
            print(f"  Go to model page and accept conditions:")
            print(f"  https://huggingface.co/{model_id}")
            print()
            print("  After accepting - run again.")
            print("=" * 60)
        else:
            print(f"Error downloading: {e}")
        
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Download shifter-tts models from HuggingFace")
    parser.add_argument(
        "model",
        choices=list(MODELS.keys()),
        default="base",
        help="Which model to download (default: base)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="HuggingFace cache path (default: ~/.cache/huggingface/hub)",
    )
    args = parser.parse_args()
    
    model_id = MODELS[args.model][0]
    description = MODELS[args.model][1]
    
    print(f"Model: {model_id}")
    print(f"Description: {description}")
    print(f"Page: https://huggingface.co/{model_id}")
    print()
    
    download_model(model_id, args.cache_dir)


if __name__ == "__main__":
    main()
