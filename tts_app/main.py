"""Qwen3-TTS Studio — десктопное приложение для синтеза речи и транскрипции."""

import sys
import os

# Добавляем родительскую директорию (корень проекта) и текущую в путь
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gui.window import main


if __name__ == "__main__":
    main()
