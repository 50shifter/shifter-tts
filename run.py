"""shifter-tts - Быстрый старт"""
import sys
import os
import subprocess

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
VENV_DIR = os.path.join(PROJECT_DIR, "venv")
DEFAULT_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
CACHE_DIR = os.path.expanduser("~/.cache/huggingface/hub")
CACHE_SUBDIR = "models--Qwen--Qwen3-TTS-12Hz-1.7B-Base"


def show(msg):
    """Show a message or progress bar"""
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()


def clear_line():
    sys.stdout.write("\r" + " " * 80 + "\r")
    sys.stdout.flush()


def get_venv_python():
    """Get venv python executable if venv exists, else sys.executable"""
    if os.path.exists(VENV_DIR):
        venv_python = os.path.join(VENV_DIR, "Scripts" if os.name == "nt" else "bin", "python.exe" if os.name == "nt" else "python")
        if os.path.exists(venv_python):
            return venv_python
    return sys.executable


def setup_venv():
    """Create venv if not exists"""
    if not os.path.exists(VENV_DIR):
        show("Creating virtual environment...")
        result = subprocess.run([sys.executable, "-m", "venv", VENV_DIR], capture_output=True)
        if result.returncode == 0:
            show("Virtual environment created")
            return True
        else:
            show("ERROR: Failed to create venv")
            return False
    else:
        show("Virtual environment already exists")
        return True


def install_dependencies():
    """Install from requirements.txt"""
    py_exe = get_venv_python()
    req_file = os.path.join(PROJECT_DIR, "requirements.txt")
    
    if os.path.exists(req_file):
        show("Installing dependencies...")
        result = subprocess.run([py_exe, "-m", "pip", "install", "-q", "-r", req_file], capture_output=True)
    else:
        show("Installing dependencies manually...")
        packages = [
            "torch>=2.1.0", "torchaudio>=2.1.0", "transformers>=4.40.0",
            "PyQt6", "PyQt6-Qt6", "librosa>=0.10.0", "soundfile>=0.12.1",
            "sounddevice>=0.4.6", "faster-whisper>=1.0.0", "einops>=0.7.0",
            "safetensors>=0.4.0", "huggingface-hub>=0.20.0", "accelerate>=0.25.0"
        ]
        result = subprocess.run([py_exe, "-m", "pip", "install",] + packages, capture_output=True)
    
    if result.returncode == 0:
        show("Dependencies installed successfully")
        return True
    else:
        show("WARNING: Some dependencies may not installed properly")
        show("Try: " + " ".join([py_exe, "-m", "pip", "install", "-r", req_file] if os.path.exists(req_file) else [py_exe, "-m", "pip", "install"] + packages))
        return True  # Continue anyway


def download_model():
    """Download model from HuggingFace"""
    model_dir = os.path.join(CACHE_DIR, "models--Qwen--Qwen3-TTS-12Hz-1.7B-Base")
    if os.path.exists(model_dir):
        show("Model already downloaded locally")
        return True
        
    show("Downloading model... (this may take while)")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=DEFAULT_MODEL_ID,
            cache_dir=CACHE_DIR,
            resume_download=True
        )
        show("Model downloaded successfully!")
        return True
    except Exception as e:
        show(f"ERROR: Failed to download model: {e}")
        show("Try: python download_model.py base")
        return True  # Continue and show error later


def main():
    print()
    print("====================================================")
    print("  shifter-tts - Быстрый старт")
    print("====================================================")
    print()
    
    if sys.version_info < (3, 10):
        show("ERROR: Python 3.10+ required!")
        input()
        sys.exit(1)
    
    if not setup_venv():
        input()
        sys.exit(1)
    
    if not install_dependencies():
        show("ERROR: Failed to install dependencies")
        input()
        sys.exit(1)
    
    if not download_model():
        show("WARNING: Could not download model automatically")
        show("Try: python download_model.py base")
    
    show("Starting shifter-tts...")
    py_exe = get_venv_python()
    subprocess.run([py_exe, "-m", "tts_app"])


if __name__ == "__main__":
    main()
