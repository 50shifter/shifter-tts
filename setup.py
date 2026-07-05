"""shifter-tts — Установка пакета для pip."""

from setuptools import setup, find_packages


def read_requirements():
    with open("requirements.txt") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


setup(
    name="shifter-tts",
    version="0.1.0",
    description="Desktop TTS application based on Qwen3-TTS with voice cloning and calibration",
    author="",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=read_requirements(),
    entry_points={
        "console_scripts": [
            "shifter-tts=tts_app.main:main",
        ],
    },
)
