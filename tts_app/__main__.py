"""Module entry point for `python -m tts_app`"""
import sys
import os

# Same path setup as main.py
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gui.window import main

main()
