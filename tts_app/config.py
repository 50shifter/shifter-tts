"""Shared configuration for tts_app modules."""

import os

# ── Audio sample rates ────────────────────────────────────────────
PAUSE_SR = 24000
"""Частота дискретизации для генерации пауз (тишины)."""

# ── File extensions ───────────────────────────────────────────────
VIDEO_EXTENSIONS = {
    ".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm",
    ".m4v", ".mpg", ".mpeg", ".3gp", ".ogv", ".ts",
}

AUDIO_EXTENSIONS = {
    ".mp3", ".wav", ".ogg", ".flac", ".aac", ".wma", ".m4a", ".opus",
}

# ── FFmpeg search paths ──────────────────────────────────────────
def _get_ffmpeg_default_paths() -> list[str]:
    """Распространённые пути к ffmpeg.exe на Windows."""
    winget_base = (
        "C:\\Users\\User\\AppData\\Local\\Microsoft\\WinGet\\Packages\\"
        "Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\\"
        "ffmpeg-8.1.1-full_build\\bin\\ffmpeg.exe"
    )
    return [
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        winget_base,
        r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe",
    ]